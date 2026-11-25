"""Дедупликация дефектов, фильтр REQ-xxx / утечек промпта, пост-фильтр ложных срабатываний."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

from srs_audit.dedup import merge_duplicates
from srs_audit.grounding import (
    Grounder,
    is_noop_fix,
    prompt_leaks,
    scrub_prompt_leaks,
    scrub_req_refs,
    unknown_req_refs,
)
from srs_audit.llm.agents import MAX_GROUNDING_RETRIES, AgentDeps, _defect_checks, _verify
from srs_audit.models import AuditDefect, LLMDefect
from srs_audit.postfilter import false_positive_reason, split_false_positives


def mk(**kw) -> AuditDefect:
    base = dict(
        defect_id="",
        req_id="SET",
        exact_quote="цитата",
        reasoning="обоснование",
        standard_ref="ISO/IEC/IEEE 29148 §5.2.5",
        category="completeness",
        severity="Major",
        source="llm_quality",
        recommendation="правка",
        line_start=10,
        line_end=10,
        grounding_status="exact",
    )
    base.update(kw)
    return AuditDefect(**base)


# ---------------------------------------------------------------------------
# Дедупликация
# ---------------------------------------------------------------------------


def test_same_issue_from_three_layers_merged_with_max_severity_and_refs():
    det = mk(source="deterministic", rule_id="API-200-WITH-ERROR", category="consistency", severity="Critical",
             line_start=171, line_end=171, exact_quote='{"result":"ERROR","message":"Проверьте сумму"}',
             reasoning="Ошибка возвращается с HTTP-статусом успеха (2xx) и result=ERROR, при этом объявлен 400.",
             standard_ref="IEEE 830 §4.3.4 consistent; RFC 9110 §15")
    cross = mk(source="llm_crosscheck", category="api_contract", severity="Blocker", line_start=171, line_end=171,
               exact_quote='{"result":"ERROR","message":"Проверьте сумму"}',
               reasoning="Документ возвращает ошибку валидации со статусом 200 OK и полем result: ERROR, "
                         "хотя для неверных параметров объявлен 400. Клиент не отличит успех от отказа.",
               standard_ref="ISO/IEC/IEEE 29148 §5.2.6 consistent; RFC 9110 §15")
    far = mk(source="llm_crosscheck", category="api_contract", req_id="REQ-048", line_start=163, line_end=163,
             exact_quote="400. Ответ в случае, если параметры запроса заполнены неверно.",
             reasoning='Для неверных параметров объявлен 400, но пример для 200 содержит {"result":"ERROR"} — '
                       "ошибка может вернуться со статусом 200.",
             standard_ref="RFC 9110 §15.5.1")
    out = merge_duplicates([det, cross, far])
    assert len(out) == 1
    d = out[0]
    assert d.severity == "Blocker"  # приоритет Blocker > Critical > Major
    assert d.merged_count == 3
    assert {"API-200-WITH-ERROR", "LLM CrossCheck"} <= set(d.merged_rules)
    for ref in ("IEEE 830 §4.3.4 consistent", "RFC 9110 §15", "ISO/IEC/IEEE 29148 §5.2.6 consistent", "RFC 9110 §15.5.1"):
        assert ref in d.standard_ref
    assert d.standard_ref.count("RFC 9110 §15;") <= 1


def test_templated_defects_about_different_api_fields_not_merged():
    text = "Тип данных 'integer' для поля '{f}' противоречит описанию в разделе 'Ответ', где '{f}' указан как 'string'."
    a = mk(req_id="REQ-037", line_start=143, line_end=143, exact_quote="amount | integer | 32 bit",
           reasoning=text.format(f="amount"), category="consistency")
    b = mk(req_id="REQ-038", line_start=144, line_end=144, exact_quote="enabled | integer | 1 знак",
           reasoning=text.format(f="enabled"), category="consistency")
    assert len(merge_duplicates([a, b])) == 2


def test_same_field_type_mismatch_cited_far_apart_merged():
    req_side = mk(req_id="REQ-037", line_start=143, line_end=143, exact_quote="amount | integer | 32 bit",
                  reasoning="Тип 'integer' поля 'amount' в запросе противоречит ответу, где 'amount' — 'string'.",
                  severity="Critical", category="consistency")
    resp_side = mk(source="deterministic", rule_id="API-TYPE-MISMATCH", req_id="REQ-044", line_start=156,
                   line_end=156, exact_quote="amount | string | При успехе",
                   reasoning="Поле «amount» в запросе имеет тип integer (стр. 143), а в ответе — string.",
                   category="api_contract")
    out = merge_duplicates([resp_side, req_side])
    assert len(out) == 1 and out[0].severity == "Critical" and out[0].merged_count == 2


def test_deterministic_findings_never_merge_with_each_other():
    p1 = mk(source="deterministic", rule_id="LEX-PASSIVE", line_start=38, line_end=38, req_id="REQ-009",
            reasoning="Пассивная конструкция «определяется» без субъекта.")
    p2 = mk(source="deterministic", rule_id="LEX-PASSIVE", line_start=40, line_end=40, req_id="REQ-011",
            reasoning="Пассивная конструкция «сохраняется» без субъекта.")
    tz = mk(source="deterministic", rule_id="FIN-TZ-IMPLICIT", line_start=38, line_end=38, req_id="REQ-009",
            reasoning="Время задано выражением «в Алматы» без явного часового пояса (UTC+5).")
    assert len(merge_duplicates([p1, p2, tz])) == 3


def test_different_topics_on_same_line_not_merged():
    tz = mk(line_start=38, line_end=38, req_id="REQ-009", reasoning="Не указан часовой пояс (UTC+5) для границы месяца.")
    race = mk(line_start=38, line_end=38, req_id="REQ-009",
              reasoning="Возможна гонка: параллельная обработка событий по одной карте отправит два push.")
    assert len(merge_duplicates([tz, race])) == 2


def test_text_similarity_alone_needs_same_place():
    t = "Требование неполное, так как не указано, какие именно проверки должны быть выполнены для {x}."
    a = mk(req_id="REQ-030", line_start=119, line_end=119, reasoning=t.format(x="суммы"))
    b = mk(req_id="REQ-031", line_start=120, line_end=120, reasoning=t.format(x="результата сохранения"))
    assert len(merge_duplicates([a, b])) == 2  # соседние строки, разные требования — разные дефекты
    c = mk(req_id="REQ-030", line_start=119, line_end=119, reasoning=t.format(x="суммы и переключателя"))
    assert len(merge_duplicates([a, c])) == 1  # то же требование и тот же смысл — дубль


# ---------------------------------------------------------------------------
# REQ-xxx и утечки промпта
# ---------------------------------------------------------------------------

KNOWN = {"REQ-001", "REQ-009", "REQ-048"}


def test_unknown_req_refs_detected_and_scrubbed():
    text = "REQ-048 требует 400, но пример в REQ-167 возвращает 200; см. также REQ-009."
    assert unknown_req_refs(text, KNOWN) == ["REQ-167"]
    fixed = scrub_req_refs(text, KNOWN)
    assert "REQ-167" not in fixed and "стр. 167" in fixed
    assert "REQ-048" in fixed and "REQ-009" in fixed
    assert unknown_req_refs("REQ-999", set()) == []  # без реестра ID проверка выключена


def test_prompt_leak_detected_and_scrubbed():
    text = "Финтех-профиль (пункт 1 'Деньги') указывает на необходимость единых единиц."
    assert prompt_leaks(text)
    assert not prompt_leaks(scrub_prompt_leaks(text))
    assert not prompt_leaks("ISO 4217: сумма в минорных единицах (тиын).")


def _ctx(text: str) -> SimpleNamespace:
    g = Grounder(text)
    g.valid_req_ids = set(KNOWN)
    return SimpleNamespace(deps=AgentDeps(grounder=g, agent_name="test"))


def _defect(reasoning: str, **kw) -> LLMDefect:
    base = dict(req_id="REQ-048", exact_quote="Если параметры заполнены неверно, вернуть 400.", reasoning=reasoning,
                standard_ref="RFC 9110 §15", category="api_contract", severity="Critical", recommendation="Вернуть 400.")
    base.update(kw)
    return LLMDefect(**base)


DOC = "0001 Заголовок\nЕсли параметры заполнены неверно, вернуть 400.\nПример ответа 200."


def test_verifier_retries_on_unknown_req_id_in_reasoning_then_scrubs():
    ctx = _ctx(DOC)
    defects = [_defect("Пример в REQ-167 противоречит REQ-048.")]
    for _ in range(MAX_GROUNDING_RETRIES):
        with pytest.raises(ModelRetry, match="REQ-167"):
            _verify(ctx, _defect_checks(defects, "defects"))
    _verify(ctx, _defect_checks(defects, "defects"))  # ретраи исчерпаны → детерминированная зачистка
    assert len(defects) == 1  # дефект с верной цитатой не теряется
    assert "REQ-167" not in defects[0].reasoning and "стр. 167" in defects[0].reasoning
    assert "REQ-048" in defects[0].reasoning


def test_verifier_retries_on_prompt_leak_in_standard_ref():
    ctx = _ctx(DOC)
    defects = [_defect("Ошибка контракта.", standard_ref="Финтех-профиль РК (API: коды ошибок)")]
    with pytest.raises(ModelRetry, match="служебную инструкцию"):
        _verify(ctx, _defect_checks(defects, "defects"))


def test_verifier_accepts_clean_answer():
    ctx = _ctx(DOC)
    defects = [_defect("REQ-048 объявляет 400, а пример (стр. 3) отдаёт 200.")]
    _verify(ctx, _defect_checks(defects, "defects"))
    assert ctx.deps.grounding_retries == 0 and len(defects) == 1


# ---------------------------------------------------------------------------
# Пост-фильтр ложных срабатываний
# ---------------------------------------------------------------------------


def test_noop_fix_detected():
    q = "Расходы 48 500 ₸, порог 50 000 ₸ | Провести покупку на 1 500 ₸ | Итог 50 000 ₸ и один push"
    assert is_noop_fix(q, "Исправить ожидаемый результат на 'Итог 50 000 ₸' или уточнить правила округления.")
    assert is_noop_fix("REST type | GET", "Заменить «GET» на «GET».")
    assert not is_noop_fix("REST type | GET", "Заменить «GET» на «PUT».")
    assert not is_noop_fix("в Алматы", "Писать «по времени Asia/Almaty (UTC+5)».")


def test_false_arithmetic_claim_dropped_but_real_error_kept():
    ok = mk(exact_quote="Расходы 48 500 ₸, порог 50 000 ₸ | Провести покупку на 1 500 ₸ | Итог 50 000 ₸",
            reasoning="Ожидаемый результат не соответствует математическому результату — возможна опечатка.",
            recommendation="Уточнить правила округления.")
    assert "арифметическ" in (false_positive_reason(ok) or "")
    bad = mk(exact_quote="Расходы 48 500 ₸ | Провести покупку на 1 000 ₸ | Итог 50 000 ₸",
             reasoning="Арифметическая ошибка: 48 500 + 1 000 = 49 500, а не 50 000.",
             recommendation="Исправить итог на 49 500 ₸.")
    assert false_positive_reason(bad) is None


def test_valid_curl_g_not_reported_but_get_mutation_kept():
    fp = mk(exact_quote='curl -G "$BASE/alerts/saveLimit" --data-urlencode "amount=50000"',
            reasoning="Использование флага `--data` с GET-запросом (`-G`) в CURL-примере противоречит практике HTTP.")
    assert "curl -G" in (false_positive_reason(fp) or "")
    real = mk(exact_quote='curl -G "$BASE/alerts/saveLimit" --data-urlencode "amount=50000"',
              reasoning="Пример curl -G отправляет сохранение настроек методом GET — изменение состояния через "
                        "небезопасный метод.")
    assert false_positive_reason(real) is None


def test_postfilter_never_touches_deterministic_findings():
    det = mk(source="deterministic", rule_id="X", recommendation="Заменить «a b» на «a b».", exact_quote="a b")
    kept, dropped = split_false_positives([det])
    assert kept == [det] and not dropped
