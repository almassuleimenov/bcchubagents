"""razdel + pymorphy3 и детерминированные правила на эталонном документе."""

import pytest

from srs_audit.analyzers.deterministic import analyze
from srs_audit.analyzers.nlp import is_passive, tokens
from srs_audit.grounding import Grounder
from srs_audit.models import ParsedDocument
from srs_audit.parsers import parse_document, parse_markdown
from srs_audit.preflight import run_preflight
from srs_audit.scoring import compute_metrics, quality_score, verdict
from srs_audit.segmentation import segment


def run_rules(md: str, profile="kz-banking"):
    doc: ParsedDocument = parse_markdown(md, "x.md", "x")
    reqs = segment(doc)
    return reqs, analyze(doc, reqs, Grounder(doc.text), profile)


def test_lemmas_catch_inflected_weak_words():
    lem = {t.lemma for t in tokens("Система должна быстрого и удобного отклика")}
    assert {"быстрый", "удобный"} <= lem


@pytest.mark.parametrize(
    "phrase,passive",
    [
        ("Месяц определяется по дате", True),
        ("Документ не заполнен", True),
        ("Сообщение появляется под полем", False),
        ("Сервис передаёт push", False),
    ],
)
def test_passive_detection(phrase, passive):
    assert any(is_passive(t) for t in tokens(phrase)) is passive


def test_weak_loophole_openlist_universal():
    md = (
        "# Док\n\n## Функциональные требования\n\n"
        "* **Ф-1.** Система должна быстро сохранять порог.\n"
        "* **Ф-2.** Система при возможности отправляет push, SMS и т.д.\n"
        "* **Ф-3.** Система всегда показывает баланс.\n"
        "* **Ф-4.** Система показывает максимум один push.\n"
    )
    _, ds = run_rules(md)
    rules = {(d.rule_id, d.req_id) for d in ds}
    assert ("LEX-WEAK", "REQ-001") in rules
    assert ("LEX-LOOPHOLE", "REQ-002") in rules
    assert ("LEX-OPEN-LIST", "REQ-002") in rules
    assert ("LEX-UNIVERSAL", "REQ-003") in rules
    # «максимум один» — точная формулировка, не слабое слово
    assert not any(d.rule_id == "LEX-WEAK" and d.req_id == "REQ-004" for d in ds)


def test_placeholders_and_old_timezone():
    md = "# Док\n\n## Нефункциональные требования\n\nВремя по UTC+6. Формат ответа TBD.\n"
    _, ds = run_rules(md)
    ids = {d.rule_id for d in ds}
    assert "FIN-TZ-OBSOLETE" in ids
    assert "STRUCT-PLACEHOLDER" in ids


def test_money_float_in_api_table():
    md = (
        "# Док\n\n## REST API\n\n### Параметры запроса\n\n#### Query-параметры\n\n"
        "| Parameter | Format |\n| :-- | :-- |\n| `amount` | double |\n| `pan` | string |\n"
    )
    _, ds = run_rules(md)
    ids = {d.rule_id: d for d in ds}
    assert ids["FIN-MONEY-FLOAT"].severity == "Critical"
    assert "API-SENSITIVE-QUERY" in ids


def test_reference_document_key_findings(sample_md):
    pre = run_preflight(sample_md)
    doc = parse_document(pre)
    reqs = segment(doc)
    g = Grounder(doc.text, doc.line_pages())
    ds = analyze(doc, reqs, g)
    found = {d.rule_id for d in ds}
    expected = {
        "API-GET-MUTATION",
        "API-200-WITH-ERROR",
        "API-STATUS-CONTRADICTION",
        "API-STATUS-UNDESCRIBED",  # 201 упомянут, но не описан
        "API-TYPE-MISMATCH",
        "API-SENSITIVE-QUERY",
        "XCHK-READ-ENDPOINT-MISSING",
        "FIN-TZ-IMPLICIT",
        "FIN-MONEY-UNITS",
        "FIN-RACE-ONCE",
        "NFR-UNMEASURABLE",
        "STRUCT-PLACEHOLDER",
        "STRUCT-NO-APPROVAL",
        "SET-BOUNDARY-UNTESTED",
    }
    assert expected <= found, expected - found
    # каждая цитата детерминированного слоя дословно найдена в исходнике
    assert all(d.grounding_status == "exact" for d in ds)
    assert all(d.line_start > 0 for d in ds)
    # Ф-1..Ф-4 распознаны с исходными ID
    assert [r.source_id for r in reqs if r.source_id.startswith("Ф")] == ["Ф-1", "Ф-2", "Ф-3", "Ф-4"]


def test_scoring_and_gate():
    assert quality_score({}, 10)[0] == 100.0
    s, _ = quality_score({"Major": 3, "Minor": 2}, 30)
    assert s > 80
    assert verdict({"Blocker": 1}, 95)[0] == "NO-GO"
    assert verdict({"Critical": 3}, 95)[0] == "NO-GO"
    assert verdict({"Critical": 1}, 95)[0] == "CONDITIONAL GO"
    assert verdict({"Major": 5}, 95)[0] == "CONDITIONAL GO"
    assert verdict({"Major": 4}, 85)[0] == "GO"
    assert verdict({}, 59.9)[0] == "NO-GO"
    m = compute_metrics([], [])
    assert m.verdict == "GO" and m.score == 100.0
