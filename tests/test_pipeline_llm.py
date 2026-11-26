"""Полный прогон конвейера с мок-моделью (pydantic-ai FunctionModel) — без сети и ключей.

Мок намеренно галлюцинирует, чтобы проверить контур самокоррекции:
* QualityFilter: первая попытка с выдуманной цитатой → ModelRetry → исправленная цитата;
* CrossCheck: одна гипотеза с выдуманной цитатой упорствует → после 2 ретраев уходит в карантин;
* QualityFilter отклоняет кандидата LEX-PASSIVE → он попадает в «отсеянные ложные срабатывания».
"""

from __future__ import annotations

import json
import re

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from typer.testing import CliRunner

from srs_audit.cli import app
from srs_audit.pipeline import AuditOptions, run_audit
from srs_audit.reporting.md_renderer import render_json, render_markdown

HALLUCINATION = "Система должна поддерживать оплату криптовалютой через блокчейн-шлюз"
CALLS: dict[str, int] = {}


def _prompt(messages: list[ModelMessage]) -> str:
    for m in messages:
        for p in getattr(m, "parts", []):
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                return p.content
    return ""


def _is_retry(messages: list[ModelMessage]) -> int:
    """Число предыдущих ответов модели в истории = номер попытки (обратная связь ModelRetry
    в pydantic-ai 2.x приходит как ToolReturnPart/RetryPromptPart после нашего ответа)."""
    return sum(1 for m in messages if isinstance(m, ModelResponse))


def _reply(info: AgentInfo, payload: dict) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, payload)])


def fake_llm(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    props = set(info.output_tools[0].parameters_json_schema.get("properties", {}))
    prompt = _prompt(messages)
    retries = _is_retry(messages)

    if "requirements" in props:  # RequirementExtractorAgent
        CALLS["extract"] = CALLS.get("extract", 0) + 1
        return _reply(
            info,
            {
                "requirements": [
                    {"source_id": "Ф-1", "section": "Функциональные требования", "kind": "functional",
                     "exact_quote": "Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.",
                     "summary": "Клиент задаёт целый порог 1 000–1 000 000 ₸"},
                    {"source_id": "", "section": "Нефункциональные требования", "kind": "nonfunctional",
                     "exact_quote": "Сервис передаёт push на отправку в течение минуты после получения события.",
                     "summary": "Push передаётся на отправку ≤ 1 минуты"},
                ],
                "api_endpoints": [
                    {"method": "GET", "path": "/alerts/saveLimit", "purpose": "Сохранение порога",
                     "params": [{"name": "cardId", "location": "query", "data_type": "string", "constraints": "1–64"}],
                     "status_codes": ["200", "400", "401", "403", "500"],
                     "exact_quote": "Dev: /alerts/saveLimit"}
                ],
                "domain_facts": [
                    {"term": "Порог", "value": "целый, 1 000–1 000 000 ₸",
                     "exact_quote": "целый порог от 1 000 до 1 000 000 ₸ включительно"}
                ],
            },
        )

    if "traces" in props:  # CrossCheckFinTechAgent.consistency — трассировка приёмки
        CALLS["consistency"] = CALLS.get("consistency", 0) + 1
        return _reply(
            info,
            {
                "traces": [
                    {"criterion_quote": "Покупка 31 января в 23:58 на 4 000 ₸",
                     "applicable_rules": "Ф-2: месяц по дате совершения покупки",
                     "derivation": "Совершена 31 января → январский счётчик",
                     "expected_by_rules": "Январский счётчик", "expected_by_criterion": "Февральский счётчик",
                     "verdict": "contradiction"}
                ],
                "defects": [
                    {"exact_quote": "На карту приходится максимум один такой push за календарный месяц.",
                     "req_id": "SET", "category": "consistency", "severity": "Critical",
                     "reasoning": "Ф-3 ограничивает push одним в месяц, а Ф-4 требует push после смены порога «независимо от истории».",
                     "standard_ref": "ISO 29148 §5.2.6", "recommendation": "Определить приоритет Ф-3 и Ф-4."}
                ],
            },
        )

    if "candidate_verdicts" in props and "set_checks" not in props:  # QualityFilterAgent
        CALLS["quality"] = CALLS.get("quality", 0) + 1
        passive = re.findall(r"(C-\d{3}) \[LEX-PASSIVE", prompt)
        verdicts = [{"cand_id": c, "keep": False, "reason": "Описание события, а не требование к системе"} for c in passive[:1]]
        f4 = re.search(r"(REQ-\d{3}) \[Ф-4\]", prompt)  # Ф-4 в ЭТОМ пакете (документ целиком есть в каждом)
        if f4 is None:
            return _reply(info, {"candidate_verdicts": verdicts, "defects": []})
        quote = (
            "Если она ниже нового порога, следующее достижение вызывает push независимо от истории предыдущей настройки."
            if retries
            else "Если сумма уже превышает порог, push отправляется немедленно"  # галлюцинация на 1-й попытке
        )
        return _reply(
            info,
            {
                "candidate_verdicts": verdicts,
                "defects": [
                    {"req_id": f4.group(1), "exact_quote": quote,
                     "reasoning": "Не определено поведение, если накопленная сумма уже не ниже нового порога.",
                     "standard_ref": "ISO/IEC/IEEE 29148 §5.2.5 complete", "category": "completeness",
                     "severity": "Critical", "recommendation": "Добавить правило для суммы ≥ нового порога."}
                ],
            },
        )

    if "set_checks" in props:  # CrossCheckFinTechAgent
        CALLS["cross"] = CALLS.get("cross", 0) + 1
        return _reply(
            info,
            {
                "candidate_verdicts": [],
                "defects": [
                    {"req_id": "REQ-009", "exact_quote": "Месяц определяется по дате совершения покупки в Алматы.",
                     "reasoning": "Ф-2 относит покупку к месяцу по дате совершения (31 января), а критерий приёмки №2 "
                                  "ожидает рост февральского счётчика — противоречие.",
                     "standard_ref": "ISO/IEC/IEEE 29148 §5.2.6 consistent; IEEE 830 §4.3.4", "category": "consistency",
                     "severity": "Critical", "recommendation": "Определить, какая дата операции задаёт месяц."},
                    {"req_id": "SET", "exact_quote": HALLUCINATION, "reasoning": "Выдуманный дефект",
                     "standard_ref": "—", "category": "completeness", "severity": "Major", "recommendation": "—"},
                ],
                "set_checks": [
                    {"prop": "29148_consistent", "verdict": "fail", "comment": "Ф-2 противоречит критерию приёмки №2."}
                ],
                "api_matrix": [
                    {"aspect": "Чтение настроек", "business_quote": "Приложение читает и сохраняет настройки",
                     "api_quote": "", "status": "missing_in_api", "comment": "Нет метода чтения."}
                ],
                "fintech_domains": [
                    {"domain": "time_calendar", "status": "gap", "finding": "Не определена метка времени и UTC+5."}
                ],
            },
        )

    if "questions" in props:  # QuestionSynthesizerAgent
        CALLS["questions"] = CALLS.get("questions", 0) + 1
        ids = re.findall(r"(D-\d{3}) \[Critical", prompt)[:2]
        return _reply(
            info,
            {
                "questions": [
                    {"target_role": "PO", "question": "Какая дата операции определяет месяц расходов?",
                     "related_defect_ids": ids + ["D-999"],
                     "options": ["Дата авторизации", "Дата проведения"], "recommended_index": 0,
                     "recommended_option": "Дата авторизации",
                     "decision_impact": "Меняет тестовые данные на границе месяца.",
                     "before_text": "Месяц определяется по дате совершения покупки в Алматы.",
                     "after_ears_proposal": "Сервис расходов должен относить покупку к месяцу по дате авторизации "
                                            "в часовом поясе Asia/Almaty (UTC+5)."}
                ]
            },
        )
    raise AssertionError(f"unexpected agent schema: {props}")


@pytest.fixture()
def mock_opts(tmp_path):
    CALLS.clear()
    return AuditOptions(model="function:mock", model_override=FunctionModel(fake_llm), cache_path=str(tmp_path / "c.db"),
                        rpm=10_000)


async def test_full_pipeline_with_mock_llm(sample_md, mock_opts):
    report = await run_audit(sample_md, mock_opts)
    assert report.passport.mode == "llm", report.passport.notes
    assert set(CALLS) == {"extract", "quality", "cross", "consistency", "questions"}
    assert report.acceptance_traces and report.acceptance_traces[0].verdict == "contradiction"
    assert any("максимум один такой push" in d.exact_quote for d in report.defects)

    # противоречие Ф-2 ↔ критерий приёмки: найдено с обеих сторон (Ф-2, стр. 38 и трасса критерия, стр. 64),
    # дедупликация слила их в ОДИН дефект уровня Blocker (прямое противоречие правила и критерия приёмки)
    contra = [d for d in report.defects if d.source == "llm_crosscheck" and d.category == "consistency"]
    blocker = [d for d in contra if d.severity == "Blocker"]
    assert len(blocker) == 1 and blocker[0].merged_count == 2 and blocker[0].grounding_status == "exact"
    assert blocker[0].line_start in (38, 64)
    assert report.metrics.by_severity["Blocker"] >= 1

    # самокоррекция: галлюцинация QualityFilter исправлена после ModelRetry
    fixed = [d for d in report.defects if d.source == "llm_quality"]
    assert fixed and "независимо от истории" in fixed[0].exact_quote
    assert report.citation.corrected_via_retry >= 1
    assert report.citation.retries_issued >= 3

    # упорная галлюцинация CrossCheck — в карантине и НЕ в отчёте
    assert any(HALLUCINATION in q.quote for q in report.quarantine)
    assert all(HALLUCINATION not in d.exact_quote for d in report.defects)

    # HLC-фильтр отклонил кандидата
    assert any(c.rule_id == "LEX-PASSIVE" for c in report.rejected_candidates)

    # сводка и гейт
    assert report.metrics.verdict == "NO-GO"
    assert report.metrics.n_requirements >= 40
    assert any(r.summary == "Клиент задаёт целый порог 1 000–1 000 000 ₸" for r in report.requirements)
    assert len(report.set_checks) == 13 and len(report.fintech_domains) == 7

    # вопросы: неизвестный ID отфильтрован, «Было» заземлено
    q = report.questions[0]
    assert "D-999" not in q.related_defect_ids and q.before_grounded

    md = render_markdown(report)
    for h in ("## 1. Паспорт прогона", "## 2. Нарезка требований", "## 3. Проверка каждого требования",
              "## 4. Проверка всего набора требований", "## 5. Сводка", "## 6. Итог и план действий"):
        assert h in md
    assert "Карантин неподтверждённых гипотез" in md and HALLUCINATION[:30] in md
    assert json.loads(render_json(report))["metrics"]["verdict"] == "NO-GO"


async def test_cache_makes_second_run_free(sample_md, mock_opts):
    await run_audit(sample_md, mock_opts)
    first = dict(CALLS)
    CALLS.clear()
    report = await run_audit(sample_md, mock_opts)
    assert CALLS == {}, "второй прогон должен идти целиком из кэша"
    assert report.passport.cache_hits >= 4 and first
    assert any(q.agent == "CrossCheckFinTechAgent" for q in report.quarantine)


async def test_llm_failure_degrades_to_rules(sample_md, tmp_path):
    def broken(messages, info):
        raise RuntimeError("provider down")

    opts = AuditOptions(model="function:broken", model_override=FunctionModel(broken), use_cache=False, rpm=10_000)
    report = await run_audit(sample_md, opts)
    assert report.metrics.verdict == "NO-GO"
    assert report.defects and all(d.source == "deterministic" for d in report.defects)
    assert any("недоступен" in n for n in report.passport.notes)


runner = CliRunner()


def test_cli_offline_exit_code_and_outputs(sample_md, tmp_path):
    out = tmp_path / "r" / "report.md"
    res = runner.invoke(app, ["check", str(sample_md), str(out), "--offline-lint", "--html-highlight", "-q"])
    assert res.exit_code == 1, res.output  # NO-GO
    text = out.read_text(encoding="utf-8")
    assert "## 6. Итог и план действий" in text and "NO-GO" in text
    assert out.with_suffix(".html").exists()


def test_cli_broken_file_exit_2(broken_files, tmp_path):
    out = tmp_path / "diag.md"
    res = runner.invoke(app, ["check", str(broken_files["png_as.md"]), str(out), "-q"])
    assert res.exit_code == 2
    text = out.read_text(encoding="utf-8")
    assert "UNPROCESSABLE" in text and "IMAGE_NOT_DOCUMENT" in text
    assert "## 6. Итог и план действий" in text


def test_cli_json_format(sample_md, tmp_path):
    out = tmp_path / "report.json"
    res = runner.invoke(app, ["check", str(sample_md), str(out), "--offline-lint", "--format", "json", "-q"])
    assert res.exit_code == 1
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["metrics"]["n_requirements"] > 0 and data["defects"]


def test_cli_pdf_export(sample_md, tmp_path):
    pytest.importorskip("typst")
    out = tmp_path / "report.md"
    res = runner.invoke(app, ["check", str(sample_md), str(out), "--offline-lint", "--pdf", "-q"])
    assert res.exit_code == 1
    pdf = out.with_suffix(".pdf")
    assert pdf.exists() and pdf.read_bytes()[:4] == b"%PDF"


async def test_html_dashboard_self_contained(sample_md, mock_opts):
    from srs_audit.reporting.html_renderer import render_html

    report = await run_audit(sample_md, mock_opts)
    page = render_html(report)
    # автономность: ни одного внешнего ресурса
    assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)="https?://', page)
    # 4 вкладки, корректурные пометки на полях, карточки на каждый дефект, граф и вопросы
    assert page.count('role="tab"') == 4 and page.count('role="tabpanel"') == 4
    assert all(f'id="card-{d.defect_id}"' in page for d in report.defects)
    assert "<mark " in page and 'class="pin ' in page
    assert 'id="graph-data"' in page and "Разрывы трассировки" in page
    assert "Копировать для Jira" in page and "Стало (EARS)" in page
    assert "Grounded" in page and report.passport.agent_version in page
