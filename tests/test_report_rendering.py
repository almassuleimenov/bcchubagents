"""Регрессии рендеринга: статусы матрицы A ↔ B, приоритет P0 в бэклоге, рекомендуемый вариант."""

from __future__ import annotations

import re

import pytest

from srs_audit.models import ApiMatrixRow, AuditReport, Question, RunPassport, DocumentHealthReport
from srs_audit.pipeline import finalize_questions
from srs_audit.postfilter import reconcile_matrix_status
from srs_audit.reporting.html_renderer import render_html
from srs_audit.reporting.md_renderer import render_markdown

from test_dedup_and_filters import mk

# ---------------------------------------------------------------------------
# Матрица A ↔ B: статус следует из комментария
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("comment", "expected"),
    [
        ("Изменение настроек реализовано методом GET, что нарушает RFC 9110 (GET должен быть безопасным).", "mismatch"),
        ("Параметры передаются в query, а amount и enabled в запросе integer, но в ответе — string.", "mismatch"),
        ("Бизнес-ошибка возвращается с HTTP 200 и result=ERROR, что дублирует и противоречит коду 400.", "mismatch"),
        ("Коды 401, 403, 500 описаны, но 403 подменяется на 404 в тексте.", "mismatch"),
        ("Сумма за месяц отображается на UI, но в ответах метода параметр суммы расходов отсутствует.", "missing_in_api"),
        ("Для API сохранения настроек не определен Idempotency-Key или защита от гонки.", "missing_in_api"),
        ("Поведение при сбое отправки push может привести к потере уведомления.", "risk"),
        ("Границы 1 000 и 1 000 000 тенге зафиксированы и в части А, и в части В.", "consistent"),
        ("Авторизация через Bearer token совпадает с логикой проверки сессии и карты.", "consistent"),
        ("Противоречий нет: метод и типы совпадают.", "consistent"),
        ("Нет расхождений между частями.", "consistent"),
    ],
)
def test_consistent_status_reconciled_with_comment(comment, expected):
    assert reconcile_matrix_status("consistent", comment, "api") == expected


def test_reconcile_never_softens_model_verdict():
    assert reconcile_matrix_status("mismatch", "Всё совпадает.") == "mismatch"
    assert reconcile_matrix_status("missing_in_business", "Метод отсутствует в части А", "GET /x") == "missing_in_business"


def _report(**kw) -> AuditReport:
    return AuditReport(
        passport=RunPassport(document_title="Тест", document_path="t.md", started_at="2026-11-27 10:00:00 (UTC+05:00)",
                             model="offline", mode="offline", profile="kz-banking", agent_version="v1.0.1",
                             tool_version="1"),
        health=DocumentHealthReport(path="t.md", file_name="t.md"),
        **kw,
    )


def test_every_matrix_status_has_label_and_tone():
    rows = [ApiMatrixRow(aspect=f"a{i}", business_quote="", api_quote="", comment="c", status=s)
            for i, s in enumerate(["consistent", "mismatch", "missing_in_api", "missing_in_business", "risk", "ambiguous"])]
    page = render_html(_report(api_matrix=rows))
    assert page.count('class="pill p-ok"') == 1  # «согласовано» — только у consistent
    assert page.count('class="pill p-bad"') == 2 and page.count('class="pill p-warn"') == 3
    assert "риск / пробел" in page and "нет в API" in page
    md = render_markdown(_report(api_matrix=rows))
    assert md.count("✅ согласовано") == 1 and "⚠️ риск / пробел" in md


# ---------------------------------------------------------------------------
# Бэклог вопросов: P0 первым, рекомендуемый вариант по индексу
# ---------------------------------------------------------------------------


def _q(role, rel, options=("Вариант А", "Вариант Б"), rec="Вариант 1: А, потому что проще", idx=1):
    return Question(target_role=role, question=f"Вопрос {role}", related_defect_ids=list(rel), options=list(options),
                    recommended_index=idx, recommended_option=rec, decision_impact="влияние",
                    before_text="было", after_ears_proposal="Когда X, система должна Y.")


def test_blocker_question_priority_zero_goes_first_everywhere():
    defects = [mk(defect_id="D-001", severity="Minor"), mk(defect_id="D-002", severity="Blocker"),
               mk(defect_id="D-003", severity="Major")]
    qs = finalize_questions([_q("PO", ["D-001"]), _q("QA", ["D-003"]), _q("SA", ["D-002"])], defects)
    assert [q.priority for q in qs] == [0, 2, 3]
    assert qs[0].target_role == "SA" and qs[0].question_id == "Q-01"

    page = render_html(_report(questions=qs))
    order = re.findall(r'<article class="q" data-role="\w+" id="q-(Q-\d+)"', page)
    assert order == ["Q-01", "Q-02", "Q-03"]  # P0 — самый верх списка
    assert 'class="prio p0"' in page and ">P0<" in page and "—</span>" not in page.split('id="q-Q-01"')[1][:200]

    md = render_markdown(_report(questions=qs))
    backlog = md[md.index("Бэклог уточняющих вопросов"):]
    assert backlog.index("SA ·") < backlog.index("PO ·")  # роль с P0-вопросом идёт первой
    assert "**`P0`** <sub>(Blocker)</sub>" in backlog


def test_recommended_option_follows_explicit_number_then_index():
    defects = [mk(defect_id="D-001", severity="Major")]
    # текст говорит «Вариант 1», индекс ошибочно 1 → подсвечиваем первый (его читает человек)
    q = finalize_questions([_q("PO", ["D-001"], rec="Вариант 1: хранить в тиынах", idx=1)], defects)[0]
    assert q.recommended_position() == 0
    page = render_html(_report(questions=[q]))
    opts = page[page.index('<ol class="opts">'):page.index("</ol>")]
    assert opts.index('class="opt rec"') < opts.index('class="opt"')

    # номера в тексте нет — решает recommended_index
    q2 = _q("PO", ["D-001"], options=("Хранить в тиынах", "Хранить в тенге"), rec="Так проще считать", idx=1)
    assert q2.recommended_position() == 1


def test_risk_or_ambiguous_escalated_when_comment_describes_mismatch():
    c = "Типы указаны, однако amount и enabled передаются как integer в запросе, но возвращаются как string."
    assert reconcile_matrix_status("risk", c, "amount | integer") == "mismatch"
    assert reconcile_matrix_status("ambiguous", "Метод чтения отсутствует в API.", "") == "missing_in_api"
    assert reconcile_matrix_status("ambiguous", "Стороны сложно сопоставить.", "") == "ambiguous"
