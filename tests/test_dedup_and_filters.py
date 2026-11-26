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


def test_deterministic_findings_of_unrelated_rules_never_merge():
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


# ---------------------------------------------------------------------------
# Остаточные дубли «на той же строке» и описательные разделы
# ---------------------------------------------------------------------------


def test_same_spot_duplicates_merged_across_categories_and_severities():
    # Prod URL: правило (completeness/Major) + LLM (другая формулировка, другая категория)
    rule = mk(source="deterministic", rule_id="API-ENV-MISSING", req_id="REQ-026", line_start=108, line_end=108,
              severity="Major", category="completeness", exact_quote="Request URL | Dev: /alerts/saveLimit . Prod: —",
              reasoning="Не указан адрес для среды «Prod»: контракт нельзя развернуть/протестировать в целевом контуре.",
              standard_ref="IEEE 830 §4.3.3 complete")
    llm = mk(source="llm_quality", req_id="REQ-026", line_start=109, line_end=109, severity="Critical",
             category="verifiability", exact_quote="Request URL | Dev: /alerts/saveLimit . Prod: —",
             reasoning="Для продуктивной среды (Prod) не указан URL конечной точки — тестировщики не смогут проверить релиз.",
             standard_ref="ISO/IEC/IEEE 29148 §5.2.5 verifiable")
    # 201: «упоминается, однако не описан» (без слова «но»)
    s201a = mk(source="deterministic", rule_id="API-STATUS-UNDESCRIBED", req_id="REQ-041", line_start=151, line_end=151,
               reasoning="Код ответа 201 упоминается, но не описан в разделе статусов (нет условия и тела ответа).")
    s201b = mk(source="llm_quality", req_id="REQ-041", line_start=151, line_end=151, category="api_contract",
               severity="Minor",
               reasoning="Упоминается код 201 Created как вариант успешного ответа, однако описание этого кода отсутствует.")
    out = merge_duplicates([rule, llm, s201a, s201b])
    assert len(out) == 2
    prod = next(d for d in out if d.line_start in (108, 109))
    assert prod.severity == "Critical" and prod.merged_count == 2
    assert "IEEE 830 §4.3.3 complete" in prod.standard_ref and "29148 §5.2.5 verifiable" in prod.standard_ref
    code = next(d for d in out if d.line_start == 151)
    assert code.severity == "Major" and code.merged_count == 2


def test_same_spot_rule_does_not_merge_different_issue_of_same_requirement():
    types = mk(source="deterministic", rule_id="API-TYPE-MISMATCH", req_id="REQ-044", line_start=156, line_end=156,
               exact_quote="amount | string | При успехе",
               reasoning="Поле «amount» в запросе имеет тип integer (стр. 143), а в ответе — string.",
               recommendation="Привести тип amount в ответе к integer, как в запросе.")
    presence = mk(source="llm_quality", req_id="REQ-044", line_start=156, line_end=156,
                  exact_quote="amount | string | При успехе",
                  reasoning="Описание поля 'amount' указывает, что оно присутствует 'При успехе', но не уточняет его "
                            "поведение при неуспешном ответе.",
                  recommendation="Указать, возвращается ли amount при ошибке и с каким значением.")
    assert len(merge_duplicates([types, presence])) == 2


DESIGN_FP = dict(
    source="llm_quality", category="feasibility",
    reasoning="Это описание архитектуры и последовательности вызовов, а не требование: лишнее проектное решение.",
    recommendation="Переместить описание в раздел архитектуры / технического проектирования.",
)


@pytest.mark.parametrize("section", [
    "Документ › Часть A › Архитектура решения",
    "Документ › Часть A › Дизайн процесса UI",
    "Документ › Требования к изменениям › BPMN схема процесса",
    "Документ › Часть A › Sequence Diagram",
])
def test_design_claim_dropped_in_descriptive_sections(section):
    d = mk(req_id="REQ-005", **DESIGN_FP)
    kept, dropped = split_false_positives([d], {"REQ-005": section})
    assert not kept and "не лишнее проектное решение" in dropped[0][1]


def test_design_claim_kept_in_functional_section_and_real_defects_kept_in_design_sections():
    d = mk(req_id="REQ-009", **DESIGN_FP)
    assert false_positive_reason(d, "Документ › Требования к изменениям › Функциональные требования") is None
    ambiguity = mk(req_id="REQ-005", reasoning="Не указано, какой сервис хранит признак отправки push.",
                   recommendation="Указать владельца признака отправки.")
    assert false_positive_reason(ambiguity, "Документ › Часть A › Архитектура решения") is None


def test_two_different_rules_on_same_http_code_merge_but_same_rule_does_not():
    undescribed = mk(source="deterministic", rule_id="API-STATUS-UNDESCRIBED", req_id="REQ-041", line_start=151,
                     line_end=151, category="api_contract",
                     reasoning="Код ответа 201 упоминается, но не описан в разделе статусов.")
    contradiction = mk(source="deterministic", rule_id="API-STATUS-CONTRADICTION", req_id="REQ-041", line_start=162,
                       line_end=162, category="completeness",
                       reasoning="В тексте упоминаются коды 201 и 404, но в разделе статусов они не описаны.")
    assert len(merge_duplicates([undescribed, contradiction])) == 1
    tz38 = mk(source="deterministic", rule_id="FIN-TZ-IMPLICIT", req_id="REQ-009", line_start=38, line_end=38,
              reasoning="Время задано выражением «в Алматы» без явного часового пояса (UTC+5).")
    tz64 = tz38.model_copy(update={"req_id": "REQ-020", "line_start": 64, "line_end": 64})
    assert len(merge_duplicates([tz38, tz64])) == 2  # одно правило в двух местах — два места правки


def test_form_defect_does_not_swallow_content_defect_on_same_line():
    atomic = mk(source="deterministic", rule_id="STRUCT-NON-ATOMIC", req_id="REQ-010", line_start=39, line_end=39,
                category="singularity", severity="Minor",
                reasoning="Требование Ф-3 объединяет 2 самостоятельных утверждения: их нельзя независимо проверить.")
    race_llm = mk(source="llm_quality", req_id="REQ-010", line_start=39, line_end=39, category="consistency",
                  reasoning="Ограничение «один push» не защищено от гонки при одновременном поступлении двух событий.")
    race_rule = mk(source="deterministic", rule_id="FIN-RACE", req_id="REQ-010", line_start=39, line_end=39,
                   category="fintech", reasoning="Ограничение «максимум один» не подкреплено механизмом: при "
                   "параллельной обработке двух событий возможна гонка.")
    out = merge_duplicates([atomic, race_llm, race_rule])
    assert len(out) == 2
    assert next(d for d in out if d.category != "singularity").merged_count == 2


def test_nfr_measurement_conditions_from_rule_and_llm_merged():
    rule = mk(source="deterministic", rule_id="NFR-NO-CONDITIONS", req_id="REQ-015", line_start=46, line_end=46,
              category="verifiability",
              reasoning="Показатель времени задан без условий измерения: нет перцентиля (p95/p99), нагрузки и окна.")
    llm = mk(source="llm_quality", req_id="REQ-015", line_start=46, line_end=46, category="verifiability",
             reasoning="В REQ-015 не указаны условия измерения времени доставки (перцентили p95/p99 и нагрузка).")
    assert len(merge_duplicates([rule, llm])) == 1


def test_rule_vs_acceptance_contradiction_cited_from_both_sides_merged():
    rule_side = mk(source="llm_crosscheck", req_id="REQ-009", line_start=38, line_end=38, category="consistency",
                   severity="Blocker", exact_quote="Ф-2. Месяц определяется по дате совершения покупки в Алматы.",
                   reasoning="Пункт REQ-009 устанавливает, что месяц расходов определяется по дате совершения покупки. "
                             "Однако во втором критерии приёмки (стр. 64) указано, что покупка от 31 января, проведённая "
                             "1 февраля, увеличивает «февральский счётчик», что противоречит REQ-009.")
    criterion_side = mk(source="llm_crosscheck", req_id="REQ-020", line_start=64, line_end=64, category="consistency",
                        severity="Blocker", exact_quote="Покупка 31 января в 23:58 на 4 000 ₸",
                        reasoning="Критерий приёмки ожидает: Февральский счётчик вырос на 4 000 ₸. По правилам документа "
                                  "(REQ-009 (Ф-2)) получается: январский счётчик вырос на 4 000 ₸.")
    out = merge_duplicates([rule_side, criterion_side])
    assert len(out) == 1 and out[0].merged_count == 2


def test_contract_property_merged_wherever_cited_but_not_across_endpoints():
    rule = mk(source="deterministic", rule_id="API-GET-MUTATION", req_id="REQ-024", line_start=104, line_end=104,
              category="api_contract", severity="Blocker", exact_quote="REST type | GET",
              reasoning="Операция изменения состояния (/alerts/saveLimit) объявлена методом GET.")
    llm_set = mk(source="llm_quality", req_id="SET", line_start=104, line_end=104, category="api_contract",
                 severity="Blocker", exact_quote="**REST type** | GET",
                 reasoning="Операция сохранения настроек является изменяющей, однако объявлена с использованием GET.")
    llm_far = mk(source="llm_quality", req_id="REQ-017", line_start=52, line_end=52, category="correctness",
                 severity="Blocker", exact_quote="Приложение читает и сохраняет настройки через дочерний документ",
                 reasoning="Метод сохранения настроек ошибочно объявлен с типом GET, хотя выполняет изменение данных.")
    out = merge_duplicates([rule, llm_set, llm_far])
    assert len(out) == 1 and out[0].merged_count == 3

    other = mk(source="llm_quality", req_id="REQ-090", line_start=300, line_end=300, category="api_contract",
               exact_quote="REST type | GET",
               reasoning="Операция удаления (/cards/block) объявлена методом GET и изменяет состояние.")
    assert len(merge_duplicates([rule, other])) == 2  # другой эндпоинт — другой дефект
