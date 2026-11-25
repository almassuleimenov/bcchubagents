"""Детерминированный слой анализа (0 токенов LLM).

Три группы правил:

* **Лексика требования** (ISO 29148 §5.2.7, INCOSE GtWR v4): слабые слова, лазейки,
  открытые перечни, абсолюты, пассив без субъекта, голые числа, неатомарность,
  неизмеримые НФТ.
* **Структура набора** (IEEE 830 §4.3): уникальность ID, битые ссылки, отсутствие ID,
  ранжирование, версионность, заглушки, незаполненные чек-листы и согласования,
  покрытие критериями приёмки и граничными значениями.
* **Финтех-профиль kz-banking + контракт REST API**: GET для изменяющих операций,
  чувствительные данные в query, HTTP 200 с ошибкой, недостижимые/неописанные коды,
  отсутствие 409/422/429, несовпадение типов запрос/ответ, float для денег, тиын/тенге,
  часовой пояс (UTC+5 с 01.03.2024), идемпотентность, гонки «не более одного», PAN/IBAN.

Каждый кандидат несёт ДОСЛОВНУЮ цитату из блока исходника и проходит тот же
верификатор цитат, что и выводы LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..grounding import Grounder, normalize
from ..models import AuditDefect, Block, ParsedDocument, Requirement
from . import lexicon as L
from .nlp import context_quote, is_passive, tokens

MONEY_NAME = re.compile(r"amount|sum|fee|limit|balance|threshold|price|total|сумм|порог|лимит|баланс|комисси", re.I)
FLOAT_TYPE = re.compile(r"^\s*(float|double|number|real|decimal\(?\s*\)?|numeric)\s*$", re.I)
SENSITIVE_PARAM = re.compile(r"card|pan|iin|bin|iban|account|phone|passport|cvv|cvc|token|счет|карт|иин|паспорт", re.I)
MUTATION_WORD = re.compile(
    r"save|update|set|create|delete|remove|add|change|сохран|измен|удал|созда|замен|добав|установ", re.I
)
HTTP_METHOD = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b")
URL_PATH = re.compile(r"(?<![\w$])/[A-Za-z][\w\-./{}]*")
STATUS_DESCRIBED = re.compile(r"^\s*(?:HTTP\s*)?([1-5]\d\d)\b\.?")
STATUS_MENTION = re.compile(r"(?<![\d.,])(?<!\d\s)\b([1-5]\d\d)\b(?![\s ]?\d{3})(?!\s*(?:bit|бит|₸|тенге|тиын))")
OLD_TZ = re.compile(r"UTC\s*\+\s*0?6\b|GMT\s*\+\s*0?6\b|\+06:00|\bALMT\b|UTC\+6", re.I)
EXPLICIT_TZ = re.compile(r"UTC\s*\+\s*0?5\b|GMT\s*\+\s*0?5\b|\+05:00|Asia/Almaty|Asia/Qostanay|UTC\+5", re.I)
IMPLICIT_TZ = re.compile(
    r"(?:по|в|времени|время)\s+(?:г\.\s*)?(?:Алмат\w*|Астан\w*|Нур-Султан\w*)|местно\w* времени|local time", re.I
)
ONCE_PHRASE = re.compile(r"максимум один|не более одного|не больше одного|однократно|только один раз|единожды", re.I)
PAN_CANDIDATE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
IBAN_KZ = re.compile(r"\bKZ\d{2}[0-9A-Z]{3}[0-9A-Z]{13}\b")
RANGE = re.compile(r"от\s+(\d[\d  ]*)\s+до\s+(\d[\d  ]*)")
NFR_TIME = re.compile(r"в течение|не более|не позднее|секунд|минут|мс\b|ms\b|час", re.I)
NFR_COND = re.compile(r"p\d{2}|\d{2}(?:[.,]\d+)?\s*%|перцентил|percentile|rps|tps|в секунду|одновременн|нагрузк", re.I)
ID_REF = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{1,6})[-‑](\d{1,4})(?![\w-])")
PRIORITY_WORDS = re.compile(r"приоритет|priority|\bmust\b|\bshould\b|moscow|обязательн\w* для релиза|критичност", re.I)
VERSION_FILLED = re.compile(r"верси\w*\s*[:№]?\s*\d|\bv\d+\.\d+|редакци\w*\s*\d", re.I)
MISSING_VALUE = {"—", "-", "–", "", "нет данных", "n/a", "tbd"}
MISSING_KEYS = re.compile(r"url|swagger|верси|дата|автор|owner|владел|ответствен|эндпоинт|endpoint|prod", re.I)
ENV_MISSING = re.compile(r"\b(Prod|Test|Stage|Прод|Тест)\w*\s*:\s*[—–-]\s*(?=$|[.;|])", re.I)


@dataclass
class _Ctx:
    doc: ParsedDocument
    reqs: list[Requirement]
    profile: str
    grounder: Grounder
    out: list[AuditDefect] = field(default_factory=list)
    seen: set[tuple[str, int, str]] = field(default_factory=set)
    req_norm: dict[str, str] = field(default_factory=dict)

    def find_req(self, quote_norm: str, line: int) -> Requirement | None:
        best = None
        for r in self.reqs:
            if r.line_start <= line <= r.line_end:
                if quote_norm and quote_norm in self.req_norm[r.req_id]:
                    return r
                best = best or r
        return best

    def emit(
        self,
        rule_id: str,
        quote: str,
        reasoning: str,
        standard_ref: str,
        category: str,
        severity: str,
        recommendation: str,
        *,
        req: Requirement | None = None,
        scope: str | None = None,
        hint: tuple[int, int] | None = None,
    ) -> None:
        quote = quote.strip()
        if not quote:
            return
        g = self.grounder.ground(quote, hint or ((req.line_start, req.line_end) if req else None))
        if not g.found:
            return  # детерминированные цитаты берутся из исходника; не нашли — значит не выдумываем
        qn = normalize(quote)
        key = (rule_id, g.line_start, qn[:60])
        if key in self.seen:
            return
        self.seen.add(key)
        if req is None and scope != "set":
            req = self.find_req(qn, g.line_start)
        self.out.append(
            AuditDefect(
                defect_id="",
                req_id=req.req_id if req and scope != "set" else "SET",
                exact_quote=g.matched_text if g.status == "fuzzy" else quote,
                reasoning=reasoning,
                standard_ref=standard_ref,
                category=category,  # type: ignore[arg-type]
                severity=severity,  # type: ignore[arg-type]
                source="deterministic",
                recommendation=recommendation,
                scope="set" if scope == "set" or req is None else "requirement",  # type: ignore[arg-type]
                rule_id=rule_id,
                verified_location=g.location,
                grounding_score=g.score,
                grounding_status="exact" if g.status == "exact" else "fuzzy",
                line_start=g.line_start,
                line_end=g.line_end,
                page=g.page,
            )
        )


# ---------------------------------------------------------------------------
# 1. Лексика отдельного требования
# ---------------------------------------------------------------------------

_NUM_WORDS = {"один", "одна", "два", "две", "три", "четыре", "пять", "десять", "сто", "тысяча"}


def _lexical(ctx: _Ctx, r: Requirement) -> None:
    text = r.text
    toks = tokens(text)
    low = text.lower().replace("ё", "е")

    # слабые слова
    hits = 0
    for i, t in enumerate(toks):
        if t.lemma not in L.WEAK_LEMMAS or hits >= 3:
            continue
        if t.lemma in L.NUMERIC_EXEMPT:
            window = toks[i + 1 : i + 4]
            if any(w.tag == "NUMB" or w.lemma in _NUM_WORDS for w in window):
                continue
        hits += 1
        ctx.emit(
            "LEX-WEAK",
            context_quote(text, t.start, t.stop),
            f"Расплывчатый термин «{t.text}»: не задан измеримый критерий, разные читатели поймут его по-разному, "
            "проверить выполнение невозможно.",
            "ISO/IEC/IEEE 29148 §5.2.7 (vague terms), §5.2.5 unambiguous/verifiable; INCOSE GtWR R7",
            "ambiguity",
            "Major",
            f"Заменить «{t.text}» измеримым значением (число, единица, условие измерения).",
            req=r,
        )

    for phrase in L.LOOPHOLE_PHRASES:
        pos = low.find(phrase)
        if pos >= 0:
            ctx.emit(
                "LEX-LOOPHOLE",
                context_quote(text, pos, pos + len(phrase)),
                f"Лазейка «{phrase}» делает требование необязательным: исполнитель сам решает, выполнять ли его.",
                "ISO/IEC/IEEE 29148 §5.2.7 (loopholes); INCOSE GtWR R8",
                "verifiability",
                "Major",
                "Убрать оговорку либо явно описать условие, при котором требование действует (EARS «Если …, то …»).",
                req=r,
            )

    for phrase in L.OPEN_ENDED:
        pos = low.find(phrase)
        if pos >= 0:
            ctx.emit(
                "LEX-OPEN-LIST",
                context_quote(text, pos, pos + len(phrase)),
                f"Открытый перечень «{phrase}»: состав не определён, полноту реализации нельзя проверить.",
                "ISO/IEC/IEEE 29148 §5.2.7 (open-ended); INCOSE GtWR R15/R16",
                "completeness",
                "Major" if phrase != "и/или" else "Minor",
                "Перечислить все элементы закрыто; вместо «и/или» явно указать допустимые комбинации.",
                req=r,
            )

    for t in toks:
        if t.lemma in L.UNIVERSAL_LEMMAS:
            ctx.emit(
                "LEX-UNIVERSAL",
                context_quote(text, t.start, t.stop),
                f"Абсолютный квантор «{t.text}» практически непроверяем: требует доказательства для всех случаев.",
                "ISO/IEC/IEEE 29148 §5.2.7 (absolutes); INCOSE GtWR R26",
                "verifiability",
                "Minor",
                "Ограничить область действия (условия, перечень случаев) или заменить измеримым критерием.",
                req=r,
            )
            break

    # пассив без субъекта (только для нормативных видов требований)
    if r.kind in {"functional", "nonfunctional", "data", "integration", "business", "constraint"}:
        lemmas = {t.lemma for t in toks}
        if not (lemmas & L.ACTOR_LEMMAS):
            for t in toks:
                if is_passive(t):
                    ctx.emit(
                        "LEX-PASSIVE",
                        context_quote(text, t.start, t.stop),
                        f"Пассивная конструкция «{t.text}» без субъекта: не указано, какая система или роль "
                        "отвечает за действие — непонятно, кому реализовывать и что тестировать.",
                        "ISO/IEC/IEEE 29148 §5.2.4 (subject of requirement), §5.2.5 unambiguous; INCOSE GtWR R2",
                        "ambiguity",
                        "Minor",
                        "Переформулировать в активном залоге: «<Система> должна …».",
                        req=r,
                    )
                    break

    # голые числа без единиц
    if r.kind in {"functional", "nonfunctional", "data", "business", "constraint"}:
        for i, t in enumerate(toks):
            if t.tag != "NUMB" or not t.text.isdigit():
                continue
            if i > 0 and toks[i - 1].text in {"-", "‑", ":", "."}:
                continue
            window = toks[i + 1 : i + 8]
            if any(w.lemma in L.UNIT_WORDS or w.text in {"₸", "%"} or w.lemma in L.MONTHS for w in window):
                continue
            if i + 1 < len(toks) and toks[i + 1].tag == "NUMB":
                continue  # «1 000» — часть разряда
            ctx.emit(
                "LEX-BARE-NUMBER",
                context_quote(text, t.start, t.stop),
                f"Число «{t.text}» без единицы измерения: неясно, в чём оно выражено.",
                "ISO/IEC/IEEE 29148 §5.2.5 unambiguous; INCOSE GtWR R6 (units)",
                "ambiguity",
                "Minor",
                "Указать единицу измерения (₸, тиын, секунды, шт.) и точность.",
                req=r,
            )
            break

    # неизмеримое НФТ
    if r.kind == "nonfunctional" and NFR_TIME.search(text) and not NFR_COND.search(text):
        ctx.emit(
            "NFR-UNMEASURABLE",
            text,
            "Показатель времени задан без условий измерения: нет перцентиля (p95/p99), нагрузки (событий/с), "
            "точки начала/окончания отсчёта и поведения при деградации — нагрузочный тест нельзя спроектировать.",
            "ISO/IEC/IEEE 29148 §5.2.5 verifiable, §9.5.x performance; IEEE 830 §4.3.6",
            "verifiability",
            "Major",
            "Указать: перцентиль (например, p99 ≤ 60 с), пиковую нагрузку, точки замера и что считается нарушением.",
            req=r,
        )


def _atomicity(ctx: _Ctx, r: Requirement) -> None:
    if r.kind in {"acceptance", "api"} and not r.source_id.startswith(("Ф", "F", "FR", "NFR", "REQ", "BR")):
        return
    from .nlp import sentences

    body = re.sub(r"^\s*[A-ZА-ЯЁ]{1,6}[-‑]\d{1,4}\.\s*", "", r.text)
    sents = [s for s in sentences(body) if len(s[2].split()) >= 2]
    words = len(body.split())
    modal = sum(1 for t in tokens(body) if t.lemma in L.MODAL_LEMMAS)
    if len(sents) >= 2 or words > 45 or modal > 1:
        sev = "Major" if len(sents) >= 3 or words > 45 or modal > 1 else "Minor"
        ctx.emit(
            "STRUCT-NON-ATOMIC",
            r.text if len(r.text) <= 320 else r.text[:300].rsplit(" ", 1)[0],
            f"Требование {r.source_id or r.req_id} объединяет {max(len(sents), modal)} самостоятельных утверждения "
            f"({words} слов): их нельзя независимо трассировать, тестировать и менять.",
            "ISO/IEC/IEEE 29148 §5.2.5 singular; INCOSE GtWR R18 (single thought)",
            "singularity",
            sev,
            f"Разделить на атомарные требования с собственными ID (например, {r.source_id or r.req_id}.1, "
            f"{r.source_id or r.req_id}.2 …).",
            req=r,
        )


# ---------------------------------------------------------------------------
# 2. Структура набора (IEEE 830 §4.3)
# ---------------------------------------------------------------------------


def _placeholders(ctx: _Ctx) -> None:
    pats = [re.compile(p, re.I) for p in L.PLACEHOLDER_PATTERNS + [r"не проставлен\w*"]]
    for b in ctx.doc.blocks:
        if b.kind == "code":
            continue
        for rx in pats:
            m = rx.search(b.text)
            if not m:
                continue
            test_section = "тестир" in b.section_path.lower()
            ctx.emit(
                "STRUCT-PLACEHOLDER",
                context_quote(b.text, m.start(), m.end()),
                (
                    "Раздел тестирования не заполнен: требования не имеют тестовых сценариев/критериев проверки."
                    if test_section
                    else f"Заглушка «{m.group(0)}»: содержание раздела не определено, документ не завершён."
                ),
                "ISO/IEC/IEEE 29148 §5.2.6 complete (no TBD); IEEE 830 §4.3.3 complete",
                "verifiability" if test_section else "placeholder",
                "Major",
                "Заполнить раздел до передачи в разработку или явно зафиксировать срок и ответственного (TBD-реестр).",
                hint=(b.line_start, b.line_end),
            )
            break

        m = ENV_MISSING.search(b.text)
        if m:
            ctx.emit(
                "API-ENV-MISSING",
                b.text,
                f"Не указан адрес для среды «{m.group(1)}»: контракт нельзя развернуть/протестировать в целевом контуре.",
                "ISO/IEC/IEEE 29148 §5.2.6 complete; IEEE 830 §4.3.3",
                "completeness",
                "Major",
                "Указать URL для каждой среды или ссылку на реестр эндпоинтов.",
                hint=(b.line_start, b.line_end),
            )

    # «поле — значение» с пустым значением
    for b in ctx.doc.blocks:
        if b.kind == "table_row" and len(b.cells) == 2:
            key, val = b.cells[0].strip(), b.cells[1].strip().lower()
            if val in MISSING_VALUE and MISSING_KEYS.search(key):
                ctx.emit(
                    "STRUCT-MISSING-VALUE",
                    b.text,
                    f"Обязательное поле «{key}» не заполнено.",
                    "IEEE 830 §4.3.3 complete; ISO/IEC/IEEE 29148 §5.2.6",
                    "completeness",
                    "Minor" if "swagger" in key.lower() else "Major",
                    f"Заполнить поле «{key}» (для API — ссылка на OpenAPI/Swagger-спецификацию).",
                    scope="set",
                    hint=(b.line_start, b.line_end),
                )


def _tables(ctx: _Ctx) -> None:
    by_table: dict[int, list[Block]] = {}
    for b in ctx.doc.blocks:
        if b.table_id is not None:
            by_table.setdefault(b.table_id, []).append(b)
    for rows in by_table.values():
        header = next((r for r in rows if r.kind == "table_header"), None)
        body = [r for r in rows if r.kind == "table_row"]
        if not body:
            continue
        unfilled = [r for r in body if len(re.findall(r"Да\s*/\s*Нет", r.text, re.I)) >= 2]
        if len(unfilled) >= 3:
            target = next((r for r in unfilled if "протестир" in r.text.lower()), unfilled[0])
            ctx.emit(
                "STRUCT-CHECKLIST-UNFILLED",
                target.cells[0] if target.cells else target.text,
                f"Чек-лист ревью не пройден: в {len(unfilled)} из {len(body)} пунктов оставлены шаблонные варианты "
                "«Да / Нет» — нет подтверждения тестируемости, отката, согласования с ИБ и архитекторами.",
                "ISO/IEC/IEEE 29148 §5.2.6 able to be validated; IEEE 830 §4.3",
                "conformity",
                "Major",
                "Пройти чек-лист ревью и проставить отметки с комментариями до начала разработки.",
                scope="set",
                hint=(target.line_start, target.line_end),
            )
        empty = [r for r in body if not any(c.strip() for c in r.cells)]
        if header and empty and len(empty) == len(body):
            hdr = " ".join(header.cells).lower()
            if any(k in hdr for k in ("подпис", "фио", "согласу", "подразделен")):
                ctx.emit(
                    "STRUCT-NO-APPROVAL",
                    header.text,
                    "Блок согласования пуст: документ не согласован ни одним подразделением (ИБ, архитектура, бизнес).",
                    "ISO/IEC/IEEE 29148 §6.4 (requirements management, agreement); IEEE 830 §4.3",
                    "conformity",
                    "Major",
                    "Получить и зафиксировать согласования ответственных лиц до старта разработки.",
                    scope="set",
                    hint=(header.line_start, header.line_end),
                )


def _structure(ctx: _Ctx) -> None:
    ids: dict[str, Requirement] = {}
    for r in ctx.reqs:
        sid = r.source_id
        if not sid or sid.startswith(("п.", "HTTP")):
            continue
        if sid in ids:
            ctx.emit(
                "STRUCT-DUP-ID",
                r.text,
                f"Идентификатор {sid} повторяется (уже использован в стр. {ids[sid].line_start}).",
                "IEEE 830 §4.3.8 traceable; ISO/IEC/IEEE 29148 §5.2.5 (unique identifier)",
                "traceability",
                "Major",
                "Присвоить уникальный идентификатор.",
                req=r,
            )
        ids[sid] = r

    prefixes = {re.split(r"[-‑]", s)[0] for s in ids}
    for b in ctx.doc.blocks:
        for m in ID_REF.finditer(b.text):
            ref = f"{m.group(1)}-{m.group(2)}"
            if m.group(1) in prefixes and ref not in ids and ref.replace("-", "‑") not in ids:
                ctx.emit(
                    "STRUCT-BROKEN-REF",
                    context_quote(b.text, m.start(), m.end()),
                    f"Ссылка на несуществующее требование {ref}.",
                    "IEEE 830 §4.3.8 traceable",
                    "traceability",
                    "Major",
                    "Исправить ссылку или добавить требование.",
                    hint=(b.line_start, b.line_end),
                )

    normative = [r for r in ctx.reqs if r.kind in {"nonfunctional", "data", "integration", "ui", "process"}]
    no_id = [r for r in normative if not r.source_id]
    if ids and len(no_id) >= 2:
        first = next((r for r in no_id if r.kind == "nonfunctional"), no_id[0])
        ctx.emit(
            "STRUCT-NO-IDS",
            first.text,
            f"{len(no_id)} нормативных требований (НФТ, данные, интеграция, UI) не имеют идентификаторов, "
            f"в отличие от функциональных ({', '.join(sorted(ids)[:6])}). На них нельзя сослаться в тестах и задачах.",
            "IEEE 830 §4.3.8 traceable; ISO/IEC/IEEE 29148 §5.2.6",
            "traceability",
            "Minor",
            "Присвоить ID всем требованиям (НФ-1, Д-1, И-1, UI-1 …) и вести матрицу трассировки.",
            scope="set",
        )

    if ctx.reqs and not PRIORITY_WORDS.search(ctx.doc.text):
        anchor = next((b for b in ctx.doc.blocks if b.kind == "heading" and "функциональн" in b.text.lower()), None)
        anchor = anchor or next((b for b in ctx.doc.blocks if b.kind == "heading"), None)
        if anchor:
            ctx.emit(
                "STRUCT-NOT-RANKED",
                anchor.text,
                "Ни одно требование не имеет приоритета/стабильности — нельзя планировать MVP и управлять объёмом.",
                "IEEE 830 §4.3.5 ranked for importance and/or stability",
                "conformity",
                "Minor",
                "Добавить атрибут приоритета (MoSCoW) и стабильности к каждому требованию.",
                scope="set",
                hint=(anchor.line_start, anchor.line_end),
            )

    if not VERSION_FILLED.search(ctx.doc.text) and not any(
        d.rule_id == "STRUCT-PLACEHOLDER" and "верси" in d.exact_quote.lower() for d in ctx.out
    ):
        anchor = next((b for b in ctx.doc.blocks if b.kind == "heading"), None)
        if anchor:
            ctx.emit(
                "STRUCT-NO-VERSION",
                anchor.text,
                "У документа нет версии и даты: невозможно сослаться на согласованную редакцию.",
                "IEEE 830 §4.3.7 modifiable; ISO/IEC/IEEE 29148 §6.4",
                "conformity",
                "Major",
                "Добавить версию, дату и историю изменений.",
                scope="set",
                hint=(anchor.line_start, anchor.line_end),
            )


def _acceptance_coverage(ctx: _Ctx) -> None:
    acc = [r for r in ctx.reqs if r.kind == "acceptance"]
    func = [r for r in ctx.reqs if r.kind == "functional"]
    if func and acc and len(acc) < len(func):
        ctx.emit(
            "SET-ACCEPTANCE-COVERAGE",
            acc[0].text,
            f"Критериев приёмки {len(acc)} на {len(func)} функциональных требований: часть правил "
            "(в т.ч. отрицательные и граничные сценарии) не имеет проверяемого критерия.",
            "ISO/IEC/IEEE 29148 §5.2.6 able to be validated; IEEE 830 §4.3.6 verifiable",
            "verifiability",
            "Major",
            "Для каждого функционального требования добавить ≥1 позитивный и ≥1 негативный критерий приёмки "
            "в формате Given/When/Then.",
            scope="set",
            hint=(acc[0].line_start, acc[0].line_end),
        )
    acc_text = normalize(" ".join(r.text for r in acc)).replace(" ", "")
    for r in func:
        m = RANGE.search(r.text)
        if not m:
            continue
        lo, hi = (re.sub(r"\D", "", g) for g in m.groups())
        if acc and (lo not in acc_text or hi not in acc_text):
            lo_i, hi_i = int(lo), int(hi)
            fmt = lambda v: f"{v:,}".replace(",", " ")  # noqa: E731
            ctx.emit(
                "SET-BOUNDARY-UNTESTED",
                context_quote(r.text, m.start(), m.end()),
                f"Граничные значения диапазона ({fmt(lo_i)} и {fmt(hi_i)}) и значения за границами не покрыты "
                "критериями приёмки — правило валидации нельзя считать проверенным.",
                "ISO/IEC/IEEE 29148 §5.2.5 verifiable; ISTQB boundary value analysis",
                "verifiability",
                "Major",
                f"Добавить критерии приёмки для {fmt(lo_i - 1)}, {fmt(lo_i)}, {fmt(hi_i)}, {fmt(hi_i + 1)} "
                "и нецелого значения (ожидаемый текст ошибки под полем).",
                req=r,
            )


# ---------------------------------------------------------------------------
# 3. Контракт API и финтех-профиль
# ---------------------------------------------------------------------------


def _section_kind(b: Block) -> str:
    from ..segmentation import classify

    return classify(b.section_path)


def _api(ctx: _Ctx) -> None:
    blocks = ctx.doc.blocks
    api_blocks = [b for b in blocks if _section_kind(b) == "api" or "api" in b.section_path.lower()]
    if not api_blocks:
        return

    # --- методы и эндпоинты
    method_rows = [b for b in api_blocks if b.kind == "table_row" and len(b.cells) == 2 and HTTP_METHOD.fullmatch(b.cells[1].strip())]
    url_rows = [b for b in api_blocks if URL_PATH.search(b.text) and b.kind in ("table_row", "paragraph", "list_item")]
    urls = {m.group(0).rstrip(".") for b in url_rows for m in URL_PATH.finditer(b.text)}
    descr = " ".join(b.text for b in api_blocks if b.kind == "table_row" and b.cells and b.cells[0].lower().startswith(("description", "описание")))
    mutation = bool(MUTATION_WORD.search(" ".join(urls) + " " + descr))
    for mr in method_rows:
        method = mr.cells[1].strip().upper()
        if method == "GET" and mutation:
            ctx.emit(
                "API-GET-MUTATION",
                mr.text,
                f"Операция изменения состояния ({', '.join(sorted(urls)) or 'сохранение'}) объявлена методом GET. "
                "GET по RFC 9110 безопасный и кэшируемый: запросы могут повторяться прокси/браузером, оседать в "
                "логах и истории, что недопустимо для изменения настроек клиента.",
                "RFC 9110 §9.2.1 (safe methods); ISO/IEC/IEEE 29148 §5.2.5 correct; IEEE 830 §4.3.1",
                "api_contract",
                "Blocker",
                "Использовать PUT /cards/{cardId}/spending-alert (идемпотентная замена) или POST, параметры — в теле JSON.",
                hint=(mr.line_start, mr.line_end),
            )
        if mutation and method in {"GET", "POST", "PUT", "PATCH", "DELETE"} and not re.search(
            r"idempoten|идемпотент|x-request-id|request-id", ctx.doc.text, re.I
        ):
            replace_sem = re.search(r"повторн\w* сохранени\w* заменяет", ctx.doc.text, re.I)
            ctx.emit(
                "FIN-IDEMPOTENCY",
                mr.text,
                "Для изменяющего метода не определён механизм идемпотентности (Idempotency-Key / requestId) и "
                "поведение при повторе и параллельных запросах"
                + (" — «замена» настройки не защищает от гонки двух одновременных сохранений." if replace_sem else "."),
                "RFC 9110 §9.2.2 (idempotent methods); IETF draft-ietf-httpapi-idempotency-key-header; ISO/IEC/IEEE 29148 §5.2.5 complete",
                "fintech",
                "Minor" if replace_sem else "Major",
                "Ввести заголовок Idempotency-Key (UUID, TTL 24 ч) и правило last-write-wins/версионирование (ETag/If-Match).",
                hint=(mr.line_start, mr.line_end),
            )

    # --- параметры: чувствительные данные в query, float для денег, несовпадение типов
    params: dict[str, dict[str, tuple[str, Block]]] = {"request": {}, "response": {}}
    for b in api_blocks:
        if b.kind != "table_row" or len(b.cells) < 2:
            continue
        name = b.cells[0].strip().strip("`")
        ftype = b.cells[1].strip().lower()
        if not re.fullmatch(r"[A-Za-z_][\w.\-]*", name):
            continue
        sp = b.section_path.lower()
        side = "response" if ("ответ" in sp or "response" in sp) else "request"
        params[side].setdefault(name, (ftype, b))
        if side == "request" and "query" in sp and SENSITIVE_PARAM.search(name):
            ctx.emit(
                "API-SENSITIVE-QUERY",
                b.text,
                f"Идентификатор карты «{name}» передаётся в query-строке URL. URL попадают в журналы веб-серверов, "
                "прокси, APM и историю — для карточных данных это нарушает принцип минимизации и требования PCI DSS "
                "к защите данных держателя карт (если это PAN или обратимый токен).",
                "PCI DSS v4.0.1 Req. 3.3–3.5, 10.3 (логи); OWASP API Security Top 10 (API3:2023); ISO 29148 §5.2.5",
                "security",
                "Major",
                "Передавать идентификатор карты в path (непрозрачный токен карты) или в теле запроса; зафиксировать, "
                "что cardId — не PAN; запретить логирование query-строк.",
                hint=(b.line_start, b.line_end),
            )
        if MONEY_NAME.search(name) and FLOAT_TYPE.match(ftype):
            ctx.emit(
                "FIN-MONEY-FLOAT",
                b.text,
                f"Денежное поле «{name}» имеет тип {ftype}: двоичная плавающая точка даёт ошибки округления.",
                "ISO 4217 (minor units); IEEE 754 (ошибки двоичной плавающей точки); ISO/IEC/IEEE 29148 §5.2.5 correct",
                "fintech",
                "Critical",
                "Использовать целое число в минимальных единицах (тиын) или decimal-строку + код валюты ISO 4217.",
                hint=(b.line_start, b.line_end),
            )
    for name, (rtype, rb) in params["response"].items():
        if name in params["request"]:
            qtype, qb = params["request"][name]
            if qtype.split()[0] != rtype.split()[0]:
                ctx.emit(
                    "API-TYPE-MISMATCH",
                    rb.text,
                    f"Поле «{name}» в запросе имеет тип {qtype} (стр. {qb.line_start}), а в ответе — {rtype}: "
                    "клиенту приходится по-разному сериализовать одно и то же значение, растёт риск ошибок сравнения.",
                    "IEEE 830 §4.3.4 consistent; ISO/IEC/IEEE 29148 §5.2.6 consistent",
                    "api_contract",
                    "Major",
                    f"Унифицировать тип «{name}» в запросе и ответе (для суммы — integer в тиынах, для флага — boolean).",
                    hint=(rb.line_start, rb.line_end),
                )

    # --- коды ответов
    described: dict[str, Block] = {}
    for b in api_blocks:
        if b.kind in ("heading", "paragraph", "list_item"):
            m = STATUS_DESCRIBED.match(b.text)
            if m and len(b.text) < 140:
                described.setdefault(m.group(1), b)
    mentioned: dict[str, Block] = {}
    for b in api_blocks:
        if b.kind == "code" or (b.kind == "table_row" and len(b.cells) > 2):
            continue
        for m in STATUS_MENTION.finditer(b.text):
            code = m.group(1)
            if 200 <= int(code) <= 599:
                mentioned.setdefault(code, b)

    for b in api_blocks:
        if b.kind == "code" and re.search(r'"(?:result|status)"\s*:\s*"(?:ERROR|FAIL\w*)"|"error"\s*:', b.text, re.I):
            if re.search(r"\b2\d\d\b", b.section_path):
                ctx.emit(
                    "API-200-WITH-ERROR",
                    b.text,
                    "Ошибка бизнес-валидации возвращается с HTTP-статусом успеха (2xx) и result=ERROR, при этом "
                    "для неверных параметров уже объявлен 400. Клиент и мониторинг не смогут однозначно отличить "
                    "успех от отказа; два способа сообщить об одной ошибке противоречат друг другу.",
                    "IEEE 830 §4.3.4 consistent; RFC 9110 §15; ISO/IEC/IEEE 29148 §5.2.6 consistent",
                    "consistency",
                    "Blocker" if re.search(r"\b400\b", ctx.doc.text) else "Critical",
                    "Возвращать 400/422 с единым форматом ошибки (например, RFC 9457 problem+json: code, message, field); "
                    "200 — только при успешном сохранении.",
                    hint=(b.line_start, b.line_end),
                )

    for b in api_blocks:
        if b.kind == "heading":
            hm = STATUS_DESCRIBED.match(b.text)
            if not hm:
                continue
            code = hm.group(1)
            nxt = [x for x in api_blocks if x.line_start > b.line_start and x.section_path == b.section_path and x.kind != "heading"]
            for x in nxt[:2]:
                other = re.search(r"возвраща\w*\s+(\d{3})|returns?\s+(\d{3})", x.text, re.I)
                if other and (other.group(1) or other.group(2)) != code:
                    oc = other.group(1) or other.group(2)
                    ctx.emit(
                        "API-STATUS-CONTRADICTION",
                        x.text,
                        f"Раздел описывает код {code}, но в тексте указано, что возвращается {oc}: код {code} "
                        f"фактически недостижим, а {oc} отсутствует в перечне статусов.",
                        "IEEE 830 §4.3.4 consistent; ISO/IEC/IEEE 29148 §5.2.6 consistent",
                        "consistency",
                        "Major",
                        f"Оставить один код (для чужой карты обычно 404 — защита от перебора) и привести перечень статусов в соответствие.",
                        hint=(x.line_start, x.line_end),
                    )

    for code, b in sorted(mentioned.items()):
        if code not in described and code not in {"200"}:
            heading_has = any(code in d.text for d in api_blocks if d.kind == "heading")
            if heading_has:
                continue
            m = re.search(rf"\b{code}\b", b.text)
            ctx.emit(
                "API-STATUS-UNDESCRIBED",
                context_quote(b.text, m.start(), m.end()) if m else b.text,
                f"Код ответа {code} упоминается, но не описан в разделе статусов (нет условия и тела ответа).",
                "IEEE 830 §4.3.3 complete, §4.3.4 consistent",
                "api_contract",
                "Major",
                f"Описать условие возникновения и тело ответа для {code} либо убрать упоминание.",
                hint=(b.line_start, b.line_end),
            )

    if described:
        missing = [c for c in ("409", "422", "429") if c not in described and c not in mentioned]
        if missing:
            anchor = min(described.values(), key=lambda x: x.line_start)
            ctx.emit(
                "API-MISSING-CODES",
                anchor.text,
                f"Не описаны коды {', '.join(missing)}: конфликт параллельных изменений (409), бизнес-валидация "
                "суммы вне диапазона (422), превышение лимита запросов (429). Поведение клиента в этих случаях не определено.",
                "IEEE 830 §4.3.3 complete; RFC 9110 §15.5.10 (409), §15.5.21 (422); RFC 6585 §4 (429)",
                "api_contract",
                "Major",
                "Добавить в контракт 409/422/429 с телом ошибки и рекомендацией по повтору (Retry-After).",
                scope="set",
                hint=(anchor.line_start, anchor.line_end),
            )

    # --- Content-Type при отсутствии тела
    ct = next((b for b in api_blocks if re.search(r"content-type\s*:\s*application/json", b.text, re.I)), None)
    nobody = next((b for b in api_blocks if re.search(r"тело не используется|body is not used|без тела", b.text, re.I)), None)
    if ct and nobody:
        ctx.emit(
            "API-CONTENT-TYPE-NO-BODY",
            nobody.text,
            f"Указан Content-Type: application/json (стр. {ct.line_start}), но тело запроса не используется — "
            "противоречивое описание запроса.",
            "IEEE 830 §4.3.4 consistent",
            "consistency",
            "Minor",
            "Либо убрать Content-Type, либо перенести параметры в JSON-тело.",
            hint=(nobody.line_start, nobody.line_end),
        )

    # --- A↔B: часть А упоминает чтение настроек, а в API только изменяющий метод
    reads = next(
        (
            r
            for r in ctx.reqs
            if r.kind in {"integration", "functional", "ui"} and re.search(r"чита\w*|чтени\w*|получ\w* настро", r.text, re.I)
        ),
        None,
    )
    read_url = any(re.search(r"get|read|fetch|load|list", u, re.I) for u in urls)
    if reads and urls and mutation and not read_url:
        ctx.emit(
            "XCHK-READ-ENDPOINT-MISSING",
            context_quote(reads.text, 0, 1),
            f"Часть А требует чтения настроек, но в REST API описан только метод сохранения "
            f"({', '.join(sorted(urls))}). Экран не сможет показать текущий порог, состояние переключателя и сумму за месяц.",
            "ISO/IEC/IEEE 29148 §5.2.6 complete/consistent; IEEE 830 §4.3.3",
            "completeness",
            "Blocker",
            "Добавить GET /cards/{cardId}/spending-alert с полями threshold, enabled, spentThisMonth, currency.",
            req=reads,
        )


def _fintech(ctx: _Ctx) -> None:
    text = ctx.doc.text
    for b in ctx.doc.blocks:
        m = OLD_TZ.search(b.text)
        if m:
            ctx.emit(
                "FIN-TZ-OBSOLETE",
                context_quote(b.text, m.start(), m.end()),
                f"Указан устаревший часовой пояс «{m.group(0)}»: с 01.03.2024 вся территория Казахстана "
                "живёт по UTC+5 (Постановление Правительства РК). Границы суток/месяца будут смещены на час.",
                "Постановление Правительства РК о едином часовом поясе UTC+5 с 01.03.2024; IANA tz Asia/Almaty; ISO/IEC/IEEE 29148 §5.2.5 correct",
                "correctness",
                "Critical",
                "Использовать часовой пояс Asia/Almaty (UTC+5) и хранить время в UTC.",
                hint=(b.line_start, b.line_end),
            )
    if not EXPLICIT_TZ.search(text):
        for b in ctx.doc.blocks:
            m = IMPLICIT_TZ.search(b.text)
            if m:
                ctx.emit(
                    "FIN-TZ-IMPLICIT",
                    context_quote(b.text, m.start(), m.end()),
                    f"Время задано выражением «{m.group(0)}» без явного часового пояса. Фразу можно прочитать и как "
                    "место совершения покупки; не указан UTC-сдвиг (с 01.03.2024 Казахстан — UTC+5), из-за чего "
                    "покупки около полуночи на границе месяца могут попасть не в тот месяц.",
                    "ISO/IEC/IEEE 29148 §5.2.5 unambiguous; ISO 8601 (смещение UTC); IANA tz Asia/Almaty (UTC+5)",
                    "ambiguity",
                    "Major",
                    "Писать «по времени Asia/Almaty (UTC+5)» и указать, какая метка времени операции используется.",
                    hint=(b.line_start, b.line_end),
                )

    units = [b for b in ctx.doc.blocks if re.search(r"тиын", b.text, re.I)]
    if units and re.search(r"тенге|₸|\bKZT\b", text):
        b = units[0]
        if re.search(r"тенге|₸", b.text):
            ctx.emit(
                "FIN-MONEY-UNITS",
                b.text,
                "Суммы хранятся в разных единицах: покупки — в тиынах, порог — в тенге. Сравнение «расходы ≥ порог» "
                "требует пересчёта ×100; ошибка в одном месте даёт срабатывание в 100 раз раньше/позже.",
                "ISO 4217 (KZT, minor unit = 2); IEEE 830 §4.3.4 consistent",
                "fintech",
                "Major",
                "Хранить и сравнивать все суммы в одной единице (тиын, int64) и указать код валюты ISO 4217.",
                hint=(b.line_start, b.line_end),
            )

    for b in ctx.doc.blocks:
        m = ONCE_PHRASE.search(b.text)
        if m and b.kind != "code":
            ctx.emit(
                "FIN-RACE-ONCE",
                context_quote(b.text, m.start(), m.end()),
                f"Ограничение «{m.group(0)}» не подкреплено механизмом: при параллельной обработке двух событий "
                "по одной карте оба могут увидеть «порог ещё не достигнут» и отправить два уведомления (race condition).",
                "ISO/IEC/IEEE 29148 §5.2.5 complete, feasible; ISO/IEC 25010 (надёжность, целостность данных)",
                "fintech",
                "Major",
                "Зафиксировать атомарную проверку-и-установку признака отправки (уникальный ключ карта+месяц, "
                "SELECT … FOR UPDATE / compare-and-set) и последовательную обработку событий по карте.",
                hint=(b.line_start, b.line_end),
            )
            break

    for b in ctx.doc.blocks:
        for m in PAN_CANDIDATE.finditer(b.text):
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn(digits):
                ctx.emit(
                    "FIN-PAN-EXPOSED",
                    m.group(0),
                    "В документе присутствует номер, проходящий проверку Луна (похож на PAN карты).",
                    "PCI DSS v4.0.1 Req. 3.4 (PAN unreadable)",
                    "security",
                    "Blocker",
                    "Удалить/замаскировать номер (первые 6 и последние 4 цифры максимум).",
                    hint=(b.line_start, b.line_end),
                )
        for m in IBAN_KZ.finditer(b.text):
            if not _iban_ok(m.group(0)):
                ctx.emit(
                    "FIN-IBAN-INVALID",
                    m.group(0),
                    "IBAN Казахстана не проходит контроль ISO 13616 mod-97.",
                    "ISO 13616-1 (IBAN, контроль mod-97)",
                    "correctness",
                    "Major",
                    "Исправить пример IBAN (KZ + 2 контрольные цифры + 3 кода банка + 13 символов счёта).",
                    hint=(b.line_start, b.line_end),
                )


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0 and len(set(digits)) > 2


def _iban_ok(iban: str) -> bool:
    s = iban[4:] + iban[:4]
    num = "".join(str(int(ch, 36)) for ch in s)
    return int(num) % 97 == 1


# ---------------------------------------------------------------------------


def analyze(doc: ParsedDocument, reqs: list[Requirement], grounder: Grounder, profile: str = "kz-banking") -> list[AuditDefect]:
    """Запускает все детерминированные правила. Возвращает кандидатов (уже заземлённых)."""
    ctx = _Ctx(doc=doc, reqs=reqs, profile=profile, grounder=grounder)
    ctx.req_norm = {r.req_id: normalize(r.text) for r in reqs}
    for r in reqs:
        _lexical(ctx, r)
        _atomicity(ctx, r)
    _placeholders(ctx)
    _tables(ctx)
    _structure(ctx)
    _acceptance_coverage(ctx)
    _api(ctx)
    if profile == "kz-banking":
        _fintech(ctx)
    return ctx.out
