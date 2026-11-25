"""Четыре узкоспециализированных pydantic-ai агента с контуром самокоррекции цитат.

Каждый агент имеет ``@agent.output_validator``, который прогоняет все цитаты ответа через
детерминированный ``Grounder``. Не найденные цитаты → ``ModelRetry`` с точным списком
(до 2 попыток). После исчерпания попыток неподтверждённые гипотезы вырезаются из ответа и
уходят в карантин — в отчёт они не попадают.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.settings import ModelSettings

from ..grounding import (
    Grounder,
    GroundingResult,
    prompt_leaks,
    scrub_prompt_leaks,
    scrub_req_refs,
    unknown_req_refs,
)
from ..models import (
    CitationStats,
    ConsistencyResult,
    CrossCheckResult,
    ExtractionResult,
    QualityBatchResult,
    QuarantinedHypothesis,
    QuestionSet,
)
from . import prompts
from .client import REQUEST_TIMEOUT_S, LLMRuntime, SQLiteCache

MAX_GROUNDING_RETRIES = 2


@dataclass
class AgentDeps:
    grounder: Grounder
    agent_name: str
    valid_defect_ids: set[str] = field(default_factory=set)
    valid_candidate_ids: set[str] = field(default_factory=set)
    stats: CitationStats = field(default_factory=CitationStats)
    quarantine: list[QuarantinedHypothesis] = field(default_factory=list)
    grounding_retries: int = 0
    prev_failures: int = 0
    from_cache: bool = False


@dataclass
class _Check:
    """Одна цитата в ответе агента: где лежит, что утверждает, как заменить на дословный текст."""

    label: str
    quote: str
    req_id: str
    claim: str
    setter: Callable[[str], None]
    remover: Callable[[], None]
    required: bool = True  # пустая цитата недопустима
    # свободный текст элемента (reasoning, recommendation, …): (имя поля, значение, setter)
    texts: list[tuple[str, str, Callable[[str], None]]] = field(default_factory=list)


def _texts(obj: Any, *names: str) -> list[tuple[str, str, Callable[[str], None]]]:
    return [(n, getattr(obj, n) or "", lambda v, n=n: setattr(obj, n, v)) for n in names]


def _text_issues(checks: list[_Check], known: set[str]) -> list[tuple[_Check, str, str, Callable[[str], None], str]]:
    """Несуществующие REQ-xxx и ссылки на служебные части промпта в тексте ответа."""
    out = []
    for ch in checks:
        for name, value, setter in ch.texts:
            problems = []
            bad = unknown_req_refs(value, known)
            if bad:
                problems.append(
                    f"несуществующие ID {', '.join(bad)} (если это номер строки — пиши «стр. N»)"
                )
            leaks = prompt_leaks(value)
            if leaks:
                problems.append(
                    f"ссылка на служебную инструкцию «{leaks[0]}» — замени на реальный стандарт"
                )
            if problems:
                out.append((ch, name, value, setter, "; ".join(problems)))
    return out


def _verify(ctx: RunContext[AgentDeps], checks: list[_Check]) -> None:
    """Общий контур верификации: ModelRetry при провалах, карантин после исчерпания попыток."""
    deps = ctx.deps
    g = deps.grounder
    failures: list[tuple[_Check, str]] = []
    passed: list[tuple[_Check, GroundingResult]] = []
    for ch in checks:
        if not ch.quote.strip():
            if ch.required:
                failures.append((ch, "пустая цитата"))
            continue
        if ch.req_id and not g.req_exists(ch.req_id):
            failures.append((ch, f"неизвестный req_id «{ch.req_id}»"))
            continue
        res = g.ground(ch.quote)
        if res.found:
            passed.append((ch, res))
        else:
            failures.append((ch, "цитата не найдена в документе"))

    text_issues = _text_issues([ch for ch in checks if all(ch is not f for f, _ in failures)], g.valid_req_ids)

    if (failures or text_issues) and deps.grounding_retries < MAX_GROUNDING_RETRIES:
        deps.grounding_retries += 1
        deps.stats.retries_issued += 1
        deps.prev_failures = len(failures)
        lines = [f"{i}) {ch.label}: «{ch.quote[:160]}» — {why}" for i, (ch, why) in enumerate(failures[:25], start=1)]
        lines += [
            f"{len(lines) + i}) {ch.label}.{name}: {why}"
            for i, (ch, name, _v, _s, why) in enumerate(text_issues[:25], start=1)
        ]
        raise ModelRetry(
            "Автоматический верификатор отклонил элементы ответа:\n"
            + "\n".join(lines)
            + "\nИсправь: замени каждую цитату ДОСЛОВНЫМ фрагментом из пронумерованного текста документа "
            "(без номера строки), используй только существующие req_id (REQ-xxx из списка требований) и ссылки "
            "на реальные стандарты, либо удали элемент. Остальные элементы верни без изменений."
        )

    # ретраи исчерпаны: текст чиним детерминированно (REQ-167 → «стр. 167», утечки промпта — убираем)
    for _ch, _name, value, setter, _why in text_issues:
        setter(scrub_prompt_leaks(scrub_req_refs(value, g.valid_req_ids)))

    # финальная попытка: принимаем подтверждённое, остальное — в карантин
    for ch, res in passed:
        deps.stats.checked += 1
        if res.status == "exact":
            deps.stats.exact += 1
        else:
            deps.stats.fuzzy += 1
            ch.setter(res.matched_text)
    if deps.grounding_retries:
        deps.stats.corrected_via_retry += max(0, deps.prev_failures - len(failures))
    for ch, why in failures:
        deps.stats.checked += 1
        deps.stats.quarantined += 1
        deps.quarantine.append(
            QuarantinedHypothesis(agent=deps.agent_name, req_id=ch.req_id, quote=ch.quote, claim=ch.claim, reason=why)
        )
    # удаляем с конца, чтобы индексы не съезжали
    for ch, _ in reversed(failures):
        ch.remover()


def _list_remover(lst: list, item: Any) -> Callable[[], None]:
    def rm() -> None:
        for i, x in enumerate(lst):
            if x is item:
                del lst[i]
                return

    return rm


def _defect_checks(defects: list, prefix: str) -> list[_Check]:
    checks = []
    for i, d in enumerate(defects):
        checks.append(
            _Check(
                label=f"{prefix}[{i}] {d.req_id} ({d.category}/{d.severity})",
                quote=d.exact_quote,
                req_id=d.req_id,
                claim=d.reasoning,
                setter=lambda v, d=d: setattr(d, "exact_quote", v),
                remover=_list_remover(defects, d),
                texts=_texts(d, "reasoning", "recommendation", "standard_ref"),
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Фабрика агентов
# ---------------------------------------------------------------------------

# thinking="low": на этой задаче качество не падает, а ответ приходит в разы быстрее;
# timeout — на один HTTP-запрос, дальше срабатывают FallbackModel и ретраи.
SETTINGS = ModelSettings(temperature=0.1, max_tokens=16000, thinking="low", timeout=REQUEST_TIMEOUT_S)


@dataclass
class AgentSuite:
    extractor: Agent
    quality: Agent
    crosscheck: Agent
    consistency: Agent
    questions: Agent


def build_agents(model: Any) -> AgentSuite:
    extractor = Agent(
        model,
        output_type=ExtractionResult,
        deps_type=AgentDeps,
        instructions=prompts.EXTRACTOR,
        retries=4,
        model_settings=SETTINGS,
        name="RequirementExtractorAgent",
    )

    @extractor.output_validator
    def _v_extract(ctx: RunContext[AgentDeps], out: ExtractionResult) -> ExtractionResult:
        checks = [
            _Check(
                f"requirements[{i}] {r.source_id or ''}",
                r.exact_quote,
                "",
                r.summary,
                lambda v, r=r: setattr(r, "exact_quote", v),
                _list_remover(out.requirements, r),
            )
            for i, r in enumerate(out.requirements)
        ]
        checks += [
            _Check(
                f"api_endpoints[{i}] {e.method} {e.path}",
                e.exact_quote,
                "",
                e.purpose,
                lambda v, e=e: setattr(e, "exact_quote", v),
                _list_remover(out.api_endpoints, e),
            )
            for i, e in enumerate(out.api_endpoints)
        ]
        checks += [
            _Check(
                f"domain_facts[{i}] {f.term}",
                f.exact_quote,
                "",
                f.value,
                lambda v, f=f: setattr(f, "exact_quote", v),
                _list_remover(out.domain_facts, f),
            )
            for i, f in enumerate(out.domain_facts)
        ]
        _verify(ctx, checks)
        return out

    quality = Agent(
        model,
        output_type=QualityBatchResult,
        deps_type=AgentDeps,
        instructions=prompts.QUALITY,
        retries=4,
        model_settings=SETTINGS,
        name="QualityFilterAgent",
    )

    @quality.output_validator
    def _v_quality(ctx: RunContext[AgentDeps], out: QualityBatchResult) -> QualityBatchResult:
        out.candidate_verdicts = [v for v in out.candidate_verdicts if v.cand_id in ctx.deps.valid_candidate_ids]
        _verify(ctx, _defect_checks(out.defects, "defects"))
        return out

    crosscheck = Agent(
        model,
        output_type=CrossCheckResult,
        deps_type=AgentDeps,
        instructions=prompts.CROSSCHECK,
        retries=4,
        model_settings=SETTINGS,
        name="CrossCheckFinTechAgent",
    )

    @crosscheck.output_validator
    def _v_cross(ctx: RunContext[AgentDeps], out: CrossCheckResult) -> CrossCheckResult:
        out.candidate_verdicts = [v for v in out.candidate_verdicts if v.cand_id in ctx.deps.valid_candidate_ids]
        checks = _defect_checks(out.defects, "defects")
        for i, row in enumerate(out.api_matrix):
            for attr in ("business_quote", "api_quote"):
                q = getattr(row, attr)
                if q.strip():
                    checks.append(
                        _Check(
                            f"api_matrix[{i}].{attr} ({row.aspect})",
                            q,
                            "",
                            row.comment,
                            lambda v, row=row, attr=attr: setattr(row, attr, v),
                            _list_remover(out.api_matrix, row),
                            required=False,
                        )
                    )
        _verify(ctx, checks)
        return out

    # Второй проход CrossCheck: трассировка критериев приёмки и логические противоречия
    consistency = Agent(
        model,
        output_type=ConsistencyResult,
        deps_type=AgentDeps,
        instructions=prompts.CONSISTENCY,
        retries=4,
        model_settings=ModelSettings(temperature=0.1, max_tokens=16000, thinking="medium", timeout=REQUEST_TIMEOUT_S),
        name="CrossCheckFinTechAgent.consistency",
    )

    @consistency.output_validator
    def _v_consistency(ctx: RunContext[AgentDeps], out: ConsistencyResult) -> ConsistencyResult:
        checks = _defect_checks(out.defects, "defects")
        checks += [
            _Check(
                f"traces[{i}] ({t.verdict})",
                t.criterion_quote,
                "",
                t.derivation,
                lambda v, t=t: setattr(t, "criterion_quote", v),
                _list_remover(out.traces, t),
                texts=_texts(t, "applicable_rules", "derivation", "expected_by_rules", "expected_by_criterion"),
            )
            for i, t in enumerate(out.traces)
        ]
        _verify(ctx, checks)
        return out

    questions = Agent(
        model,
        output_type=QuestionSet,
        deps_type=AgentDeps,
        instructions=prompts.QUESTIONS,
        retries=4,
        model_settings=SETTINGS,
        name="QuestionSynthesizerAgent",
    )

    @questions.output_validator
    def _v_questions(ctx: RunContext[AgentDeps], out: QuestionSet) -> QuestionSet:
        valid = ctx.deps.valid_defect_ids
        for q in out.questions:
            q.related_defect_ids = [d for d in q.related_defect_ids if d in valid]
        checks = [
            _Check(
                f"questions[{i}] ({q.target_role})",
                q.before_text,
                "",
                q.question,
                lambda v, q=q: setattr(q, "before_text", v),
                _list_remover(out.questions, q),
                texts=_texts(q, "question", "recommended_option", "decision_impact", "after_ears_proposal"),
            )
            for i, q in enumerate(out.questions)
        ]
        _verify(ctx, checks)
        return out

    return AgentSuite(extractor, quality, crosscheck, consistency, questions)


# ---------------------------------------------------------------------------
# Запуск с кэшем
# ---------------------------------------------------------------------------


async def run_agent(
    runtime: LLMRuntime,
    agent: Agent,
    prompt: str,
    deps: AgentDeps,
    output_type: type[BaseModel],
    cache: SQLiteCache,
    on_retry: Callable[[str], None] | None = None,
) -> BaseModel:
    """Запускает агента (или берёт ответ из кэша) и возвращает провалидированный вывод."""
    key = cache.make_key(
        agent.name, prompts.PROMPT_VERSION, runtime.model_name, prompt, output_type.model_json_schema()
    )
    hit = cache.get(key)
    if hit is not None:
        deps.from_cache = True
        deps.stats = CitationStats(**hit["stats"])
        deps.quarantine = [QuarantinedHypothesis(**q) for q in hit["quarantine"]]
        return output_type.model_validate(hit["output"])

    async def _go():
        # каждая попытка tenacity — с чистым счётчиком самокоррекции
        deps.grounding_retries = 0
        deps.prev_failures = 0
        deps.stats = CitationStats()
        deps.quarantine = []
        return await agent.run(prompt, deps=deps, model=runtime.model)

    result = await runtime.call(_go, on_retry=on_retry)
    runtime.usage.add(agent.name or "agent", result.usage)
    out = result.output
    cache.set(
        key,
        agent.name or "agent",
        {
            "output": out.model_dump(mode="json"),
            "stats": deps.stats.model_dump(),
            "quarantine": [q.model_dump() for q in deps.quarantine],
        },
    )
    return out


async def vision_ocr(runtime: LLMRuntime, png: bytes, page_no: int, cache: SQLiteCache) -> str:
    """Распознаёт страницу-скан через мультимодальную модель (Gemini Vision)."""
    import hashlib

    from pydantic_ai import BinaryContent

    key = cache.make_key("vision", prompts.PROMPT_VERSION, runtime.model_name, hashlib.sha256(png).hexdigest())
    hit = cache.get(key)
    if hit is not None:
        return hit["text"]
    agent = Agent(runtime.model, output_type=str, instructions=prompts.VISION, name="VisionOCR")

    async def _go():
        return await agent.run([f"Страница {page_no}.", BinaryContent(data=png, media_type="image/png")])

    result = await runtime.call(_go)
    runtime.usage.add("VisionOCR", result.usage)
    cache.set(key, "VisionOCR", {"text": result.output})
    return result.output
