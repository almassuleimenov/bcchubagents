"""Оркестратор 5-ступенчатого гибридного конвейера.

1. Preflight Gatekeeper → 2. Парсинг с координатами + нарезка (код + RequirementExtractorAgent)
→ 3. Детерминированный анализ (0 токенов) → 4. LLM-агенты (QualityFilter батчами ‖ CrossCheckFinTech)
с верификатором цитат и карантином → 5. Скоринг, вопросы (QuestionSynthesizer), отчёт.

Любой сбой LLM-слоя не роняет прогон: конвейер деградирует до детерминированного результата
и честно пишет об этом в паспорте прогона.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from rapidfuzz import fuzz

from . import __version__
from .analyzers import offline_set
from .analyzers.deterministic import analyze
from .analyzers.fallback_questions import fallback_questions
from .grounding import Grounder, normalize
from .models import (
    SEVERITY_ORDER,
    AuditDefect,
    AuditReport,
    CitationStats,
    ConsistencyResult,
    CrossCheckResult,
    ExtractionResult,
    LLMDefect,
    ParsedDocument,
    QualityBatchResult,
    QualityMetrics,
    Question,
    QuestionSet,
    RejectedCandidate,
    Requirement,
    RunPassport,
)
from .parsers import parse_document
from .dedup import merge_duplicates
from .graph import build_graph, summary_for_llm
from .grounding import is_noop_fix, scrub_prompt_leaks, scrub_req_refs
from .postfilter import reconcile_matrix_status, split_false_positives
from .preflight import run_preflight
from .progress import Tracker
from .scoring import compute_metrics
from .segmentation import SKIP_SECTIONS, segment
from .versioning import agent_version

Progress = Callable[[str], None]
BATCH_SIZE = 20


@dataclass
class AuditOptions:
    model: str | None = None
    profile: str = "kz-banking"
    offline: bool = False
    use_cache: bool = True
    cache_path: str = ".cache.db"
    progress: Progress | None = None
    tracker: Tracker | None = None  # живой индикатор прогресса (CLI)
    notes: list[str] = field(default_factory=list)
    model_override: Any = None  # готовая pydantic-ai Model (тесты: FunctionModel)
    rpm: int | None = None  # лимит запросов в минуту (по умолчанию SRS_RPM или 8 — Free Tier Gemini)
    timings: dict[str, float] = field(default_factory=dict)  # агент → секунды (критический путь в паспорте)


def _say(opts: AuditOptions, msg: str) -> None:
    if opts.tracker:
        opts.tracker.set_stage(msg)
    elif opts.progress:
        opts.progress(msg)


async def _tracked(opts: AuditOptions, key: str, label: str, runtime, agent, prompt: str, deps, out_type, cache):
    """Вызов агента с отметками в индикаторе прогресса: старт, причины повторов, итог."""
    from .llm.agents import run_agent
    from .llm.client import describe_error

    tr = opts.tracker
    if tr:
        tr.start(key, label)
    t0 = time.perf_counter()
    try:
        out = await run_agent(
            runtime, agent, prompt, deps, out_type, cache, on_retry=(lambda t: tr.note(key, t)) if tr else None
        )
    except Exception as exc:
        opts.timings[key] = time.perf_counter() - t0
        if tr:
            tr.finish(key, "failed", describe_error(exc))
        raise
    opts.timings[key] = time.perf_counter() - t0
    if tr:
        bits = []
        if deps.stats.retries_issued:
            bits.append(f"самокоррекций цитат: {deps.stats.retries_issued}")
        if deps.stats.quarantined:
            bits.append(f"в карантин: {deps.stats.quarantined}")
        tr.finish(key, "cached" if deps.from_cache else "done", ", ".join(bits))
    return out


def _now() -> str:
    """Локальное время с явным UTC-сдвигом (имена поясов Windows локализованы и неинформативны)."""
    now = datetime.now(timezone.utc).astimezone()
    off = now.strftime("%z")
    return now.strftime("%Y-%m-%d %H:%M:%S") + f" (UTC{off[:3]}:{off[3:]})"


# ---------------------------------------------------------------------------
# Требования: уточнение координат и слияние с LLM-экстракцией
# ---------------------------------------------------------------------------


def _refine_lines(reqs: list[Requirement], g: Grounder) -> None:
    for r in reqs:
        res = g.ground(r.text, (r.line_start, r.line_end))
        if res.found:
            r.line_start, r.line_end = res.line_start, res.line_end
            r.page = res.page or r.page


def _renumber(reqs: list[Requirement]) -> list[Requirement]:
    reqs.sort(key=lambda r: (r.line_start, r.line_end))
    for i, r in enumerate(reqs, start=1):
        r.req_id = f"REQ-{i:03d}"
    return reqs


def _block_section(doc: ParsedDocument, line: int) -> str:
    for b in doc.blocks:
        if b.line_start <= line <= b.line_end:
            return b.section_path
    return ""


def merge_extraction(doc: ParsedDocument, reqs: list[Requirement], ext: ExtractionResult, g: Grounder) -> list[Requirement]:
    norm = {id(r): normalize(r.text) for r in reqs}
    for lr in ext.requirements:
        res = g.ground(lr.exact_quote)
        if not res.found:
            continue
        qn = normalize(lr.exact_quote)
        match = None
        for r in reqs:
            if r.line_start <= res.line_end and res.line_start <= r.line_end:
                rn = norm[id(r)]
                if qn in rn or rn in qn or fuzz.partial_ratio(qn, rn) >= 85:
                    match = r
                    break
        if match is not None:
            if not match.summary:
                match.summary = lr.summary
            if match.kind == "other" and lr.kind != "other":
                match.kind = lr.kind
            if not match.source_id and lr.source_id and lr.source_id in res.matched_text:
                match.source_id = lr.source_id
            continue
        section = _block_section(doc, res.line_start)
        if SKIP_SECTIONS.search(section):
            continue
        new = Requirement(
            req_id="",
            source_id=lr.source_id if lr.source_id and lr.source_id in doc.text else "",
            section_path=section or lr.section,
            kind=lr.kind,
            text=res.matched_text.strip(),
            summary=lr.summary,
            line_start=res.line_start,
            line_end=res.line_end,
            page=res.page,
            origin="llm",
        )
        reqs.append(new)
        norm[id(new)] = normalize(new.text)
    return _renumber(reqs)


# ---------------------------------------------------------------------------
# Дефекты: конвертация, дедупликация, нумерация
# ---------------------------------------------------------------------------


def to_audit(d: LLMDefect, source: str, g: Grounder, reqs_by_id: dict[str, Requirement]) -> AuditDefect | None:
    req = reqs_by_id.get(d.req_id)
    res = g.ground(d.exact_quote, (req.line_start, req.line_end) if req else None)
    if not res.found:
        return None
    req_id = d.req_id if req else "SET"
    return AuditDefect(
        defect_id="",
        req_id=req_id,
        exact_quote=d.exact_quote,
        reasoning=d.reasoning,
        standard_ref=d.standard_ref,
        category=d.category,
        severity=d.severity,
        source=source,  # type: ignore[arg-type]
        recommendation=d.recommendation,
        scope="set" if req_id == "SET" else "requirement",
        verified_location=res.location,
        grounding_score=res.score,
        grounding_status="exact" if res.status == "exact" else "fuzzy",
        line_start=res.line_start,
        line_end=res.line_end,
        page=res.page,
        confirmed_by_llm=True,
    )


def _why(exc: BaseException) -> str:
    """Человекочитаемая причина сбоя LLM для примечаний отчёта (429/503/таймаут по каждой модели цепочки)."""
    from .llm.client import describe_error

    return describe_error(exc)


def _scrub_texts(report: AuditReport, known: set[str]) -> None:
    """Последняя линия обороны: в тексте отчёта нет несуществующих REQ-xxx и ссылок на служебный промпт."""

    def fix(obj, *names: str) -> None:
        for n in names:
            v = getattr(obj, n, "")
            if v:
                setattr(obj, n, scrub_prompt_leaks(scrub_req_refs(v, known)))

    for d in report.defects:
        fix(d, "reasoning", "recommendation", "standard_ref")
    for c in report.set_checks:
        fix(c, "comment")
    for row in report.api_matrix:
        fix(row, "comment")
    for f in report.fintech_domains:
        fix(f, "finding")
    for t in report.acceptance_traces:
        fix(t, "applicable_rules", "derivation", "expected_by_rules", "expected_by_criterion")
    for q in report.questions:
        fix(q, "question", "recommended_option", "decision_impact", "after_ears_proposal")


def number_defects(defects: list[AuditDefect]) -> list[AuditDefect]:
    defects.sort(key=lambda d: (SEVERITY_ORDER[d.severity], d.scope != "requirement", d.line_start))
    for i, d in enumerate(defects, start=1):
        d.defect_id = f"D-{i:03d}"
    return defects


# ---------------------------------------------------------------------------
# LLM-этап
# ---------------------------------------------------------------------------

_STAT_FIELDS = ("checked", "exact", "fuzzy", "corrected_via_retry", "quarantined", "retries_issued")


def _absorb(state: dict, deps) -> None:
    """Переносит статистику цитат и карантин одного вызова агента в общий итог прогона."""
    stats: CitationStats = state["stats"]
    for f in _STAT_FIELDS:
        setattr(stats, f, getattr(stats, f) + getattr(deps.stats, f))
    state["quarantine"].extend(deps.quarantine)


async def _llm_stage(
    doc: ParsedDocument,
    reqs: list[Requirement],
    g: Grounder,
    opts: AuditOptions,
    model_name: str,
    report_notes: list[str],
    model=None,
):
    from .llm.agents import AgentDeps, build_agents
    from .llm.client import LLMRuntime, SQLiteCache, build_model
    from .llm.formatting import numbered_document, requirement_lines

    cache = SQLiteCache(opts.cache_path, enabled=opts.use_cache)
    runtime = LLMRuntime(model if model is not None else build_model(model_name), model_name, cache=cache, rpm=opts.rpm)
    suite = build_agents(runtime.model)
    numbered = numbered_document(doc)
    state: dict = {"runtime": runtime, "cache": cache, "suite": suite, "stats": CitationStats(), "quarantine": []}

    # --- 1. Экстрактор
    _say(opts, "LLM 1/4: RequirementExtractorAgent — нарезка и реестр API")
    deps = AgentDeps(grounder=g, agent_name="RequirementExtractorAgent")
    prompt = (
        f"ДОКУМЕНТ «{doc.title}» (пронумерованные строки):\n{numbered}\n\n"
        f"ЧЕРНОВАЯ НАРЕЗКА (детерминированная, {len(reqs)} шт.):\n{requirement_lines(reqs)}"
    )
    try:
        ext = await _tracked(opts, "extract", "RequirementExtractorAgent · нарезка и реестр API", runtime, suite.extractor, prompt, deps, ExtractionResult, cache)
        _absorb(state, deps)
        reqs = merge_extraction(doc, reqs, ext, g)  # type: ignore[arg-type]
        state["api_endpoints"] = ext.api_endpoints  # type: ignore[attr-defined]
        state["domain_facts"] = ext.domain_facts  # type: ignore[attr-defined]
    except Exception as exc:  # деградация: остаётся детерминированная нарезка
        report_notes.append(f"RequirementExtractorAgent недоступен ({_why(exc)}); "
                            "использована детерминированная нарезка.")
        state["api_endpoints"], state["domain_facts"] = [], []
    g.valid_req_ids = {r.req_id for r in reqs}
    state["reqs"] = reqs
    return state


async def _analysis_stage(
    doc: ParsedDocument,
    reqs: list[Requirement],
    cands: list[AuditDefect],
    g: Grounder,
    state: dict,
    opts: AuditOptions,
    report_notes: list[str],
):
    from .llm.agents import AgentDeps
    from .llm.formatting import candidate_lines, numbered_document, requirement_lines

    runtime, cache, suite = state["runtime"], state["cache"], state["suite"]
    numbered = numbered_document(doc)
    facts = "\n".join(f"- {f.term}: {f.value}" for f in state.get("domain_facts", [])) or "(нет)"
    reqs_by_id = {r.req_id: r for r in reqs}

    # --- 2. Quality батчами ‖ 3. CrossCheck
    batches = [reqs[i : i + BATCH_SIZE] for i in range(0, len(reqs), BATCH_SIZE)]
    _say(opts, f"LLM 2/4 + 3/4: QualityFilterAgent ×{len(batches)} ‖ CrossCheckFinTechAgent ×2 (набор + логика)")

    async def quality(batch: list[Requirement]):
        ids = {r.req_id for r in batch}
        mine = [c for c in cands if c.req_id in ids]
        deps = AgentDeps(grounder=g, agent_name="QualityFilterAgent", valid_candidate_ids={c.defect_id for c in mine})
        prompt = (
            f"ДОКУМЕНТ (для контекста):\n{numbered}\n\nФАКТЫ ПРЕДМЕТНОЙ ОБЛАСТИ:\n{facts}\n\n"
            f"ПАКЕТ ТРЕБОВАНИЙ ДЛЯ ПРОВЕРКИ ({len(batch)} шт.):\n{requirement_lines(batch)}\n\n"
            f"КАНДИДАТЫ ДЕТЕРМИНИРОВАННОГО СЛОЯ ДЛЯ ЭТОГО ПАКЕТА:\n{candidate_lines(mine)}"
        )
        n = batches.index(batch) + 1
        out = await _tracked(opts, f"quality{n}", f"QualityFilterAgent · пакет {n}/{len(batches)} ({len(batch)} треб.)", runtime, suite.quality, prompt, deps, QualityBatchResult, cache)
        _absorb(state, deps)
        return out

    async def cross():
        set_cands = [c for c in cands if c.req_id == "SET"]
        deps = AgentDeps(grounder=g, agent_name="CrossCheckFinTechAgent", valid_candidate_ids={c.defect_id for c in set_cands})
        api = "\n".join(
            f"- {e.method} {e.path}: {e.purpose}; параметры: "
            + ", ".join(f"{p.name}({p.location}, {p.data_type})" for p in e.params)
            + f"; коды: {', '.join(e.status_codes)}"
            for e in state.get("api_endpoints", [])
        ) or "(реестр не построен)"
        prompt = (
            f"ДОКУМЕНТ «{doc.title}»:\n{numbered}\n\nТРЕБОВАНИЯ:\n{requirement_lines(reqs)}\n\n"
            f"РЕЕСТР API:\n{api}\n\nФАКТЫ:\n{facts}\n\n"
            f"ГРАФ ТРАССИРОВКИ (построен кодом; EXTRACTED — связь записана в документе, INFERRED — выведена):\n"
            f"{summary_for_llm(state['graph']) if state.get('graph') else '(нет)'}\n\n"
            f"УЖЕ НАЙДЕНО ДЕТЕРМИНИРОВАННЫМ СЛОЕМ (не дублировать; дать вердикт по кандидатам уровня SET):\n"
            f"{candidate_lines(cands)}\n\nПрофиль: {opts.profile}"
        )
        out = await _tracked(opts, "cross", "CrossCheckFinTechAgent · набор, матрица А↔B, 7 доменов", runtime, suite.crosscheck, prompt, deps, CrossCheckResult, cache)
        _absorb(state, deps)
        return out

    async def consistency():
        deps = AgentDeps(grounder=g, agent_name="CrossCheckFinTechAgent.consistency")
        prompt = (
            f"ДОКУМЕНТ «{doc.title}»:\n{numbered}\n\nТРЕБОВАНИЯ:\n{requirement_lines(reqs)}\n\nФАКТЫ:\n{facts}"
        )
        out = await _tracked(opts, "consistency", "CrossCheckFinTechAgent · трассировка приёмки и противоречия",
                             runtime, suite.consistency, prompt, deps, ConsistencyResult, cache)
        _absorb(state, deps)
        return out

    results = await asyncio.gather(*(quality(b) for b in batches), cross(), consistency(), return_exceptions=True)
    q_results, x_result, c_result = results[:-2], results[-2], results[-1]

    verdicts: dict[str, tuple[bool, str]] = {}
    llm_defects: list[AuditDefect] = []
    for res in q_results:
        if isinstance(res, BaseException):
            report_notes.append(f"QualityFilterAgent: пакет не обработан ({_why(res)}).")
            continue
        for v in res.candidate_verdicts:
            verdicts[v.cand_id] = (v.keep, v.reason)
        for d in res.defects:
            a = to_audit(d, "llm_quality", g, reqs_by_id)
            if a:
                llm_defects.append(a)
    cross_out: CrossCheckResult | None = None
    if isinstance(x_result, BaseException):
        report_notes.append(f"CrossCheckFinTechAgent недоступен ({_why(x_result)}); "
                            "проверка набора выполнена правилами.")
    else:
        cross_out = x_result
        for v in x_result.candidate_verdicts:
            verdicts[v.cand_id] = (v.keep, v.reason)
        for d in x_result.defects:
            a = to_audit(d, "llm_crosscheck", g, reqs_by_id)
            if a:
                llm_defects.append(a)
    traces = []
    if isinstance(c_result, BaseException):
        report_notes.append(f"Проход трассировки критериев приёмки не выполнен ({_why(c_result)}).")
    else:
        traces = list(c_result.traces)
        # Трасса «противоречие / правил недостаточно» сама по себе — доказательство: превращаем её в дефект кодом,
        # не полагаясь на то, что модель продублирует вывод в defects (дедупликация уберёт повтор).
        for t in traces:
            if t.verdict not in ("contradiction", "not_covered"):
                continue
            res = g.ground(t.criterion_quote)
            owner = next((r for r in reqs if r.line_start <= res.line_start <= r.line_end), None) if res.found else None
            contra = t.verdict == "contradiction"
            d = LLMDefect(
                req_id=owner.req_id if owner else "SET",
                exact_quote=t.criterion_quote,
                reasoning=(f"Критерий приёмки ожидает: {t.expected_by_criterion}. По правилам документа "
                           f"({t.applicable_rules}) получается: {t.expected_by_rules}. Ход вывода: {t.derivation}"),
                standard_ref="ISO/IEC/IEEE 29148 §5.2.6 consistent; IEEE 830 §4.3.4 consistent, §4.3.6 verifiable",
                category="consistency" if contra else "completeness",
                # прямое противоречие правила и критерия приёмки блокирует приёмку: тест не пройдёт ни при какой реализации
                severity="Blocker" if contra else "Major",
                recommendation=("Согласовать критерий приёмки и правило: явно определить, какая дата/метка времени и "
                                "какое правило применяются, и исправить ожидаемый результат или требование."
                                if contra else "Дополнить требования так, чтобы результат критерия однозначно вычислялся."),
            )
            a = to_audit(d, "llm_crosscheck", g, reqs_by_id)
            if a:
                llm_defects.append(a)
        for d in c_result.defects:
            a = to_audit(d, "llm_crosscheck", g, reqs_by_id)
            if a:
                llm_defects.append(a)
    state["traces"] = traces
    return verdicts, llm_defects, cross_out


async def _questions_stage(defects: list[AuditDefect], g: Grounder, state: dict, opts: AuditOptions, notes: list[str]) -> list[Question]:
    from .llm.agents import AgentDeps
    from .llm.formatting import defect_lines

    runtime, cache, suite = state["runtime"], state["cache"], state["suite"]
    focus = [d for d in defects if d.severity in ("Blocker", "Critical", "Major")] or defects
    listing = defect_lines(focus[:80])
    _say(opts, "LLM 4/4: QuestionSynthesizerAgent ×2 — бизнес-роли ‖ технические роли")

    async def group(key: str, roles: tuple[str, ...], label: str):
        """Две независимые половины бэклога параллельно: время генерации пропорционально объёму ответа."""
        deps = AgentDeps(grounder=g, agent_name="QuestionSynthesizerAgent", valid_defect_ids={d.defect_id for d in defects})
        prompt = (
            f"ПОДТВЕРЖДЁННЫЕ ДЕФЕКТЫ:\n{listing}\n\n"
            f"В ЭТОМ ВЫЗОВЕ формируй вопросы ТОЛЬКО для ролей: {', '.join(roles)} ({label}); 4–9 вопросов. "
            "Дефекты вне компетенции этих ролей пропусти — их закроет параллельный вызов для других ролей."
        )
        out = await _tracked(opts, key, f"QuestionSynthesizerAgent · {label}: {', '.join(roles)}", runtime,
                             suite.questions, prompt, deps, QuestionSet, cache)
        _absorb(state, deps)
        return [q for q in out.questions if q.target_role in roles]  # type: ignore[attr-defined]

    results = await asyncio.gather(
        group("questions_biz", ("PO", "BA", "QA"), "бизнес и приёмка"),
        group("questions_tech", ("SA", "InfoSec"), "архитектура, API и ИБ"),
        return_exceptions=True,
    )
    failed = [r for r in results if isinstance(r, BaseException)]
    if len(failed) == len(results):
        notes.append(f"QuestionSynthesizerAgent недоступен ({_why(failed[0])}); вопросы сформированы по шаблонам.")
        return []
    if failed:
        notes.append(f"QuestionSynthesizerAgent: одна из двух групп ролей недоступна ({_why(failed[0])}); "
                     "бэклог вопросов неполный.")
    qs = []
    for part in results:
        if isinstance(part, BaseException):
            continue
        for cq in part:
            res = g.ground(cq.before_text)
            q = Question(**cq.model_dump(), before_grounded=res.found, before_location=res.location if res.found else "")
            q.recommended_index = cq.recommended_position()
            qs.append(q)
    return qs


def finalize_questions(questions: list[Question], defects: list[AuditDefect]) -> list[Question]:
    by_id = {d.defect_id: d for d in defects}
    for q in questions:
        sev = [SEVERITY_ORDER[by_id[i].severity] for i in q.related_defect_ids if i in by_id]
        q.priority = min(sev) if sev else 3
        q.recommended_index = q.recommended_position()  # «Вариант N» в тексте > индекс модели > текст
    # нумерация совпадает с порядком вывода: сначала P0 (Blocker), затем P1…P3; внутри — по ролям
    roles = ["PO", "BA", "SA", "QA", "InfoSec"]
    questions.sort(key=lambda q: (q.priority, roles.index(q.target_role)))
    for i, q in enumerate(questions, start=1):
        q.question_id = f"Q-{i:02d}"
    return questions


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------


MANUAL_CHECKLIST = [
    "Сверить с владельцем продукта бизнес-цель и критерии успеха фичи (метрики использования).",
    "Проверить соответствие тарифам и публичной оферте банка (тексты уведомлений, согласие клиента на push).",
    "Подтвердить у ИБ классификацию cardId и требования к журналированию (PCI DSS, Закон РК «О персональных данных»).",
    "Сверить с процессингом фактические типы операций (покупка, возврат, отмена, quasi-cash, комиссии) и их коды.",
    "Проверить схемы/диаграммы, которые агент анализирует только по тексту подписей.",
    "Убедиться, что требования согласованы со смежными командами (процессинг, сервис push, мобильное приложение).",
]


async def run_audit(input_path: str | Path, opts: AuditOptions) -> AuditReport:
    t0 = time.perf_counter()
    started = _now()
    notes: list[str] = list(opts.notes)

    from .llm.client import has_credentials, load_env, resolve_model_name

    load_env()
    model_name = None if opts.offline else resolve_model_name(opts.model)
    if model_name and not has_credentials(model_name):
        notes.append(f"Для модели {model_name} не найден API-ключ — прогон выполнен в офлайн-режиме (только правила).")
        model_name = None
    if not opts.offline and model_name is None and not notes:
        notes.append("LLM-ключ не задан — прогон выполнен в офлайн-режиме (детерминированные правила).")
    mode = "llm" if model_name else "offline"

    _say(opts, "Шаг 1/5: Preflight — проверка файла")
    pre = run_preflight(input_path, vision_available=bool(model_name and model_name.startswith("google:")))
    passport = RunPassport(
        document_title=Path(input_path).name,
        document_path=Path(input_path).as_posix(),
        document_sha256=pre.health.sha256,
        started_at=started,
        model=model_name or "— (офлайн: детерминированные правила)",
        mode=mode,  # type: ignore[arg-type]
        profile=opts.profile,
        agent_version=agent_version(),
        tool_version=__version__,
        notes=notes,
    )
    report = AuditReport(passport=passport, health=pre.health)
    # pydantic копирует списки при валидации: дальше пишем примечания прямо в паспорт отчёта
    passport = report.passport
    notes = passport.notes
    if not pre.ok:
        report.metrics = QualityMetrics(verdict="UNPROCESSABLE", verdict_reasons=[i.message for i in pre.health.issues if i.level == "fatal"])
        passport.finished_at = _now()
        passport.duration_s = round(time.perf_counter() - t0, 2)
        return report

    try:
        return await _run_main(pre, report, opts, model_name, notes, t0)
    except Exception as exc:  # последний рубеж — никогда не отдаём traceback
        pre.health.add("INTERNAL_ERROR", "fatal", f"Внутренняя ошибка анализа: {type(exc).__name__}: {exc}",
                       "Повторите прогон с --offline-lint и сообщите разработчикам.")
        report.metrics = QualityMetrics(verdict="UNPROCESSABLE", verdict_reasons=[f"{type(exc).__name__}: {exc}"])
        passport.finished_at = _now()
        passport.duration_s = round(time.perf_counter() - t0, 2)
        return report


async def _run_main(pre, report: AuditReport, opts: AuditOptions, model_name: str | None, notes: list[str], t0: float) -> AuditReport:
    passport = report.passport
    ocr: dict[int, str] = {}
    state: dict = {}
    if model_name and pre.health.scanned_pages:
        ocr = await _ocr_pages(pre, model_name, opts, notes)

    _say(opts, "Шаг 2/5: Парсинг и нарезка требований")
    doc = parse_document(pre, ocr_pages=ocr)
    passport.document_title = doc.title
    g = Grounder(doc.text, doc.line_pages())
    reqs = segment(doc)
    _refine_lines(reqs, g)
    reqs = _renumber(reqs)
    g.valid_req_ids = {r.req_id for r in reqs}

    if model_name:
        try:
            state = await _llm_stage(doc, reqs, g, opts, model_name, notes, model=opts.model_override)
            reqs = state["reqs"]
        except Exception as exc:
            notes.append(f"LLM-слой отключён: {type(exc).__name__}: {str(exc)[:200]}")
            model_name = None
            passport.mode = "offline"

    _say(opts, "Шаг 3/5: Детерминированный анализ (NLP, IEEE 830, финтех-правила, граф трассировки)")
    trace_graph = build_graph(doc, reqs)
    if state:
        state["graph"] = trace_graph
    cands = analyze(doc, reqs, g, opts.profile)
    for i, c in enumerate(cands, start=1):
        c.defect_id = f"C-{i:03d}"

    rejected: list[RejectedCandidate] = []
    defects = cands
    cross_out = None
    if model_name and state:
        _say(opts, "Шаг 4/5: LLM-аудит требований и набора")
        verdicts, llm_defects, cross_out = await _analysis_stage(doc, reqs, cands, g, state, opts, notes)
        kept = []
        for c in cands:
            keep, why = verdicts.get(c.defect_id, (True, ""))
            if keep:
                c.confirmed_by_llm = c.defect_id in verdicts
                kept.append(c)
            else:
                rejected.append(RejectedCandidate(rule_id=c.rule_id, req_id=c.req_id, quote=c.exact_quote, reason=why))
        # пост-фильтр ложных срабатываний LLM (before == after, верная арифметика, curl -G)
        llm_defects, false_pos = split_false_positives(llm_defects, {r.req_id: r.section_path for r in reqs})
        for d, why in false_pos:
            rejected.append(RejectedCandidate(rule_id=d.rule_id or d.source, req_id=d.req_id, quote=d.exact_quote,
                                              reason=f"пост-фильтр: {why}"))
        defects = kept + llm_defects
    # дедупликация находок всех слоёв — ДО скоринга
    n_before = len(defects)
    defects = number_defects(merge_duplicates(defects))
    if n_before != len(defects):
        notes.append(f"Дедупликация: {n_before} находок всех слоёв → {len(defects)} уникальных дефектов "
                     "(дубли объединены, серьёзность — максимальная в группе).")

    _say(opts, "Шаг 5/5: Скоринг, вопросы, отчёт")
    questions: list[Question] = []
    if model_name and state:
        questions = await _questions_stage(defects, g, state, opts, notes)
    if opts.timings:
        t = opts.timings
        par = max((v for k, v in t.items() if k.startswith(("quality", "cross", "consistency"))), default=0.0)
        qmax = max((v for k, v in t.items() if k.startswith("questions")), default=0.0)
        notes.append(
            "Время LLM-этапов: " + ", ".join(f"{k} {v:.0f} с" for k, v in t.items())
            + f". Критический путь: нарезка {t.get('extract', 0):.0f} с → параллельный аудит {par:.0f} с"
            f" → вопросы {qmax:.0f} с."
        )
    if not questions:
        questions = fallback_questions(defects)
    questions = [
        q for q in questions
        if normalize(q.before_text) != normalize(q.after_ears_proposal) and not is_noop_fix(q.before_text, q.after_ears_proposal)
    ]
    questions = finalize_questions(questions, defects)

    fallback_checks = offline_set.set_checks(defects)
    fallback_domains = offline_set.fintech_domains(defects)
    if cross_out is not None:
        have = {c.prop for c in cross_out.set_checks}
        set_checks = list(cross_out.set_checks) + [c for c in fallback_checks if c.prop not in have]
        order = list(offline_set.PROPERTY_CATEGORIES)
        set_checks.sort(key=lambda c: order.index(c.prop))
        dom_have = {d.domain for d in cross_out.fintech_domains}
        domains = list(cross_out.fintech_domains) + [d for d in fallback_domains if d.domain not in dom_have]
        dorder = list(offline_set.DOMAIN_OK_TEXT)
        domains.sort(key=lambda d: dorder.index(d.domain))
        matrix = list(cross_out.api_matrix) or offline_set.api_matrix(defects)
        fixed = 0
        for row in matrix:
            new = reconcile_matrix_status(row.status, row.comment, row.api_quote)
            if new != row.status:
                row.status, fixed = new, fixed + 1  # type: ignore[assignment]
        if fixed:
            notes.append(f"Матрица A ↔ B: строк со статусом мягче собственного комментария — {fixed} "
                         "(в комментарии описано расхождение или пробел); статус ужесточён кодом.")
    else:
        set_checks, domains, matrix = fallback_checks, fallback_domains, offline_set.api_matrix(defects)

    report.source_text = doc.text
    report.requirements = reqs
    report.defects = defects
    report.set_checks = set_checks
    report.fintech_domains = domains if opts.profile == "kz-banking" else []
    report.api_matrix = matrix
    report.api_endpoints = state.get("api_endpoints", []) if state else []
    report.acceptance_traces = state.get("traces", []) if state else []
    report.trace_graph = trace_graph
    report.questions = questions
    report.rejected_candidates = rejected
    _scrub_texts(report, g.valid_req_ids)
    report.metrics = compute_metrics(reqs, defects)
    if state:
        report.citation = state["stats"]
        report.quarantine = state["quarantine"]
        rt = state["runtime"]
        passport.llm_requests = rt.usage.requests
        passport.input_tokens = rt.usage.input_tokens
        passport.output_tokens = rt.usage.output_tokens
        passport.cache_hits = state["cache"].hits
        if rt.served_models:
            passport.model = f"{passport.model} (ответила: {', '.join(sorted(rt.served_models))})"
        if rt.usage.requests == 0 and state["cache"].hits == 0:
            # ни один вызов LLM не завершился успешно — честно помечаем прогон как офлайн
            passport.mode = "offline"
            passport.model = f"{passport.model} — недоступна, использованы только правила"
    report.priority_fixes = _priority_fixes(defects)
    report.manual_checklist = list(MANUAL_CHECKLIST)
    if pre.health.diagram_pages:
        report.manual_checklist.insert(0, f"Вручную проверить диаграммы на страницах {', '.join(map(str, pre.health.diagram_pages))}.")
    if report.quarantine:
        report.manual_checklist.insert(0, f"Просмотреть {len(report.quarantine)} гипотез(ы) в карантине — модель не смогла подтвердить их цитатой.")
    passport.finished_at = _now()
    passport.duration_s = round(time.perf_counter() - t0, 2)
    return report


def _priority_fixes(defects: list[AuditDefect]) -> list[str]:
    out = []
    for d in defects:
        if d.severity in ("Blocker", "Critical") or (len(out) < 5 and d.severity == "Major"):
            out.append(f"**{d.defect_id}** ({d.severity}, {d.verified_location}): {d.recommendation}")
        if len(out) >= 10:
            break
    return out


async def _ocr_pages(pre, model_name: str, opts: AuditOptions, notes: list[str]) -> dict[int, str]:
    import pymupdf

    from .llm.agents import vision_ocr
    from .llm.client import LLMRuntime, SQLiteCache, build_model

    cache = SQLiteCache(opts.cache_path, enabled=opts.use_cache)
    runtime = LLMRuntime(build_model(model_name), model_name, cache=cache)
    out: dict[int, str] = {}
    with pymupdf.open(stream=pre.data, filetype="pdf") as doc:
        for pno in pre.health.scanned_pages:
            _say(opts, f"Vision OCR: страница {pno}")
            png = doc[pno - 1].get_pixmap(dpi=150).tobytes("png")
            try:
                out[pno] = await vision_ocr(runtime, png, pno, cache)
            except Exception as exc:
                notes.append(f"Vision OCR страницы {pno} не выполнен: {type(exc).__name__}.")
    if out:
        notes.append(f"Страницы {', '.join(map(str, out))} распознаны Vision-моделью: точность цитат ниже, проверьте вручную.")
    return out
