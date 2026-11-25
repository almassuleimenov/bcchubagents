"""Дедупликация дефектов всех слоёв (детерминированный + LLM) перед скорингом.

Одну и ту же проблему часто находят 2–3 источника: правило, QualityFilter и CrossCheck — с разными
формулировками и иногда на соседних строках («200 + ERROR» цитируют и из JSON-примера, и из описания 400).
Чистое сходство текста здесь ненадёжно: шаблонные дефекты про разные поля API («amount» vs «enabled»)
похожи на 97 %, а настоящие дубли — на 30–70 %. Поэтому решение принимается по **маркерам**:

* ``fields`` — имена полей/сущностей API (латинские идентификаторы: amount, cardId, Content-Type, Swagger);
* ``codes`` — HTTP-коды из рассуждения и правки;
* ``topics`` — тема дефекта из закрытой таксономии (контракт ошибок, мутирующий GET, 403↔404, типы, единицы денег…).

Слияние запрещено, если у дефектов разные поля API или разные темы, а также для двух
детерминированных находок (каждое правило и каждое место — отдельный дефект). Разрешено, если:
дефекты рядом (±3 строки или одно требование) и совпадает тема; цитаты перекрываются (или одно
требование) и текст похож (``token_set_ratio ≥ 65``); на расстоянии — при общей теме И общем поле / HTTP-коде / ID требования (две стороны одного противоречия).

Из группы остаётся самый содержательный дефект; серьёзность — максимальная в группе,
ссылки на стандарты объединяются, источники перечисляются в ``merged_rules``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from .analyzers.deterministic import STATUS_MENTION
from .grounding import normalize
from .models import SEVERITY_ORDER, AuditDefect

LINE_WINDOW = 3
TEXT_SIMILARITY = 65

TOPICS: dict[str, re.Pattern[str]] = {
    name: re.compile(rx, re.I)
    for name, rx in {
        "error-contract": r"result\W{0,4}error|(?:\b200\b|2xx)[^.]{0,80}(?:ошибк|error)|(?:ошибк\w*|error)[^.]{0,80}(?:\b200\b|2xx)",
        "get-mutation": r"\bget\b[^.]{0,80}(?:сохран|измен\w*\s+состояни|мутир|безопасн|кэшир)"
        r"|(?:сохран\w*|изменени\w*\s+состояни\w*)[^.]{0,80}\bget\b",
        "status-403-404": r"\b403\b[^.]{0,160}\b404\b|\b404\b[^.]{0,160}\b403\b|чуж\w*\s+карт",
        "type-mismatch": r"\b(?:integer|string|boolean|number|int|float)\b[^.]{0,140}\b(?:integer|string|boolean|number|int|float)\b"
        r"|несоответстви\w*\s+тип|тип\w*\s+данных",
        "money-units": r"тиын|минорн\w*\s+единиц",
        "content-type-body": r"content-type[^.]{0,160}тел|тел\w*[^.]{0,120}content-type",
        "undescribed-status": r"(?:не описан|не приведен|упоминается, но|отсутстви\w*\s+пример\w*)[^.]{0,80}\b[1-5]\d\d\b"
        r"|\b[1-5]\d\d\b[^.]{0,80}(?:не описан|упоминается, но|не приведен|отсутстви\w*\s+пример)",
        "missing-value": r"не заполнен|отсутстви\w*\s+ссылк|пуст\w*\s+(?:пол|значени)",
        "timezone": r"\butc\b|часов\w*\s+пояс|time\s?zone|asia/almaty",
        "race": r"гонк|race condition|параллельн\w*\s+(?:обработк|событи|запрос|сохранени)|конкурентн",
        "idempotency": r"идемпотент|idempotency",
        "atomicity": r"атомарн|самостоятельн\w*\s+утвержд",
        "passive": r"пассив",
        "vague": r"расплывчат|открыт\w*\s+перечн|«и далее»|и т\.\s?д",
        "acceptance-trace": r"критери\w*\s+при[её]мки[^.]{0,60}(?:ожида(?:ет|ют)\b|противореч)"
        r"|противореч\w*[^.]{0,80}критери\w*\s+при[её]мки",
        "pii": r"\bpan\b|персональн\w*\s+данн|pci\s?dss",
        "sensitive-query": r"query-строк|в\s+url|query-параметр",
        "boundary": r"граничн\w*\s+(?:значени|случа|сценари)",
        "rounding": r"округлен",
        "approval": r"согласован|подпис",
        "ranking": r"приоритет\w*\s*/?\s*стабильн|ранжирован",
    }.items()
}

DOC_ID = re.compile(r"(?<![\w-])(?:REQ|[А-ЯЁA-Z]{1,4})-\d+(?:\.\d+)*(?![\w-])")
LINE_REF = re.compile(r"(?:стр\.?|строк\w*|line|REQ-|п\.)\s*\d+", re.I)
FIELD = re.compile(r"(?<![\w-])[A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z][A-Za-z0-9_]*)*(?![\w-])")
FIELD_STOP = {
    # типы, методы, протоколы, общие слова API
    "integer", "string", "boolean", "number", "int", "float", "double", "decimal", "json", "application", "get", "post",
    "put", "patch", "delete", "rest", "http", "https", "api", "url", "uri", "bearer", "authorization", "curl", "dev",
    "prod", "data", "bit", "body", "request", "response", "query", "header", "headers", "type", "method", "path",
    "successful", "created", "forbidden", "found", "bad", "unauthorized", "internal", "server", "error", "ok", "not",
    "unprocessable", "entity", "conflict", "too", "many", "requests", "problem", "code", "status", "true", "false",
    "null", "description", "endpoint", "base", "the", "and", "for", "with", "fetch", "read", "load", "list", "id",
    # стандарты, единицы, служебное
    "rfc", "iso", "iec", "ieee", "incose", "gtwr", "ears", "pci", "dss", "utc", "kzt", "req", "set", "sla", "p95",
    "p99", "ms", "asia", "almaty", "minor", "units", "unit", "push", "bpmn", "sequence", "ui", "ux", "mvp", "qa",
    "po", "ba", "sa", "infosec", "iban", "pan", "tbd", "todo", "xxx", "etag", "if-match", "uuid", "ttl", "problem-json",
}


@dataclass(frozen=True)
class Signature:
    fields: frozenset[str]
    codes: frozenset[str]
    topics: frozenset[str]
    refs: frozenset[str] = frozenset()  # ID требований документа (Ф-2, REQ-009), упомянутые в дефекте


def signature(d: AuditDefect) -> Signature:
    talk = f"{d.reasoning} {d.recommendation}"
    fields = {
        m.group(0).lower()
        for m in FIELD.finditer(f"{d.exact_quote} {d.reasoning}")
        if len(m.group(0)) >= 3 and m.group(0).lower() not in FIELD_STOP
    }
    # номера строк и ID требований («стр. 171», «REQ-167») — не HTTP-коды
    talk_codes = LINE_REF.sub(" ", talk)
    codes = {m.group(1) for m in STATUS_MENTION.finditer(talk_codes)}
    topics = {name for name, rx in TOPICS.items() if rx.search(talk_codes)}
    refs = {m.group(0).upper() for m in DOC_ID.finditer(f"{d.exact_quote} {talk}")}
    if d.req_id not in ("SET", ""):
        refs.add(d.req_id)
    return Signature(frozenset(fields), frozenset(codes), frozenset(topics), frozenset(refs))


def _near(a: AuditDefect, b: AuditDefect, window: int = LINE_WINDOW) -> bool:
    """Одно требование или строки в пределах ±window (window=0 — цитаты перекрываются)."""
    if a.req_id == b.req_id and a.req_id not in ("SET", ""):
        return True
    return a.line_start - window <= b.line_end and b.line_start - window <= a.line_end


def _similar(a: AuditDefect, b: AuditDefect) -> int:
    return int(
        max(
            fuzz.token_set_ratio(normalize(a.reasoning), normalize(b.reasoning)),
            fuzz.token_set_ratio(normalize(a.recommendation), normalize(b.recommendation)),
        )
    )


def vetoed(a: AuditDefect, b: AuditDefect, sa: Signature, sb: Signature) -> bool:
    """Явно разные проблемы — сливать нельзя."""
    if a.source == b.source == "deterministic":
        return True  # детерминированный слой не дублирует сам себя: разные правила/места — разные дефекты
    if sa.fields and sb.fields and sa.fields.isdisjoint(sb.fields):
        return True
    if sa.topics and sb.topics and sa.topics.isdisjoint(sb.topics):
        return True
    return False


def is_duplicate(a: AuditDefect, b: AuditDefect, sa: Signature | None = None, sb: Signature | None = None) -> bool:
    sa = sa or signature(a)
    sb = sb or signature(b)
    if vetoed(a, b, sa, sb):
        return False
    shared_topic = bool(sa.topics & sb.topics)
    if shared_topic and _near(a, b):
        return True
    # без общей темы — только при перекрытии цитат / одном требовании и похожем тексте
    if _near(a, b, window=0) and _similar(a, b) >= TEXT_SIMILARITY:
        return True
    if _near(a, b):
        return False
    # одна проблема, процитированная из разных мест документа
    return shared_topic and bool((sa.fields & sb.fields) or (sa.codes & sb.codes) or (sa.refs & sb.refs))


def _richness(d: AuditDefect) -> tuple:
    # содержательность: длина обоснования и правки, затем подтверждённость и точность цитаты
    return (len(d.reasoning) + len(d.recommendation), d.confirmed_by_llm, d.grounding_status == "exact")


def _merge_refs(refs: list[str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        for part in re.split(r";\s*", ref):
            part = part.strip().rstrip(".")
            key = normalize(part)
            if part and key not in seen:
                seen.add(key)
                out.append(part)
    return "; ".join(out)


def _source_label(d: AuditDefect) -> str:
    return d.rule_id or {"llm_quality": "LLM QualityFilter", "llm_crosscheck": "LLM CrossCheck"}.get(d.source, d.source)


def merge_duplicates(defects: list[AuditDefect]) -> list[AuditDefect]:
    """Группирует дубли и возвращает по одному дефекту на группу (порядок — по первому вхождению)."""
    sigs = [signature(d) for d in defects]
    clusters: list[list[int]] = []
    for i, d in enumerate(defects):
        home = None
        for c in clusters:
            if any(vetoed(defects[j], d, sigs[j], sigs[i]) for j in c):
                continue
            if any(is_duplicate(defects[j], d, sigs[j], sigs[i]) for j in c):
                home = c
                break
        if home is None:
            clusters.append([i])
        else:
            home.append(i)

    out: list[AuditDefect] = []
    for c in clusters:
        group = [defects[j] for j in c]
        if len(group) == 1:
            out.append(group[0])
            continue
        best = max(group, key=_richness).model_copy(deep=True)
        best.severity = min((g.severity for g in group), key=lambda s: SEVERITY_ORDER[s])
        best.standard_ref = _merge_refs([best.standard_ref] + [g.standard_ref for g in group])
        best.confirmed_by_llm = any(g.confirmed_by_llm or g.source != "deterministic" for g in group)
        if not best.rule_id:
            best.rule_id = next((g.rule_id for g in group if g.rule_id), "")
        labels = list(dict.fromkeys(_source_label(g) for g in group))
        best.merged_rules = labels
        best.merged_count = len(group)
        out.append(best)
    return out
