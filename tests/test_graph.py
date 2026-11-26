"""Граф трассировки: узлы, EXTRACTED/INFERRED-рёбра, orphan-узлы и сводка для LLM."""

from __future__ import annotations

from srs_audit.graph import build_graph, summary_for_llm
from srs_audit.parsers import parse_document
from srs_audit.preflight import run_preflight
from srs_audit.segmentation import segment


def _graph(path):
    doc = parse_document(run_preflight(path))
    return build_graph(doc, segment(doc))


def _orphans(g, kind):
    return {o.label for o in g.orphans if o.kind == kind}


def test_reference_document_graph(sample_md):
    g = _graph(sample_md)
    kinds = {n.kind for n in g.nodes}
    assert {"requirement", "acceptance", "ui_field", "endpoint", "param", "status"} <= kinds
    assert {e.origin for e in g.edges} == {"EXTRACTED", "INFERRED"}

    # правила без критерия приёмки
    assert {"Ф-1", "Ф-4"} <= _orphans(g, "requirement_without_acceptance")
    assert "Ф-2" not in _orphans(g, "requirement_without_acceptance")
    # статус упомянут в тексте, но не описан; поле экрана без API; чтение без метода
    assert "HTTP 201" in _orphans(g, "status_undescribed")
    assert "сумма за месяц" in _orphans(g, "ui_field_without_api")
    assert any(o.kind == "operation_without_endpoint" and "чтение" in o.label for o in g.orphans)
    # подпись в кавычках не разорвана, поле связано с параметром
    labels = {n.node_id: n.label for n in g.nodes}
    maps = {(labels[e.src], labels[e.dst]) for e in g.edges if e.relation == "maps_to"}
    assert any(src == "поле «Порог, ₸»" and "amount" in dst for src, dst in maps)
    # несовпадение типов запрос ↔ ответ
    mism = {labels[e.src] for e in g.edges if e.relation == "type_mismatch"}
    assert any("amount" in m for m in mism) and any("enabled" in m for m in mism)

    text = summary_for_llm(g)
    assert "ORPHAN" in text and "EXTRACTED" in text and len(text.splitlines()) <= 45


SYNTHETIC = """# Перевод между своими счетами

## Функциональные требования

* **ПР-1.** Клиент переводит деньги между своими счетами в тенге без комиссии.
* **ПР-2.** Лимит перевода — 5 000 000 ₸ в сутки по времени Asia/Almaty.

## Дизайн экрана

На экране видны список счетов, поле «Сумма, ₸» и кнопка «Перевести».

## Требования к интеграции

* Приложение сохраняет перевод через REST API.

## Критерии приемки

| Дано | Когда | Тогда |
| --- | --- | --- |
| Остаток 10 000 ₸ | Перевести 3 000 ₸ по ПР-1 | Комиссия 0 ₸ |

## REST API

### Общие параметры

| Параметр | Значение |
| --- | --- |
| REST type | POST |
| Request URL | /transfers |

### Request body

| Имя | Тип | Обяз. | Описание |
| --- | --- | --- | --- |
| amount | integer | Да | Сумма перевода в тиынах |

### Статусы

#### 201. Created

```json
{"id":"t-1"}
```

#### 409. Conflict

Повторный перевод с тем же ключом.
"""


def test_generic_document_extracted_reference_and_orphans(tmp_path):
    p = tmp_path / "transfer.md"
    p.write_text(SYNTHETIC, encoding="utf-8")
    g = _graph(p)
    labels = {n.node_id: n.label for n in g.nodes}
    verifies = [(e, labels[e.dst]) for e in g.edges if e.relation == "verifies"]
    # явная ссылка «ПР-1» в критерии → EXTRACTED
    assert any(e.origin == "EXTRACTED" and dst.startswith("ПР-1") for e, dst in verifies)
    assert "ПР-2" in _orphans(g, "requirement_without_acceptance")
    # 409 описан заголовком, но без примера ответа; у 201 пример есть
    assert "HTTP 409" in _orphans(g, "status_without_example")
    assert "HTTP 201" not in _orphans(g, "status_without_example")
    # поле экрана связано с параметром, «список счетов» — нет
    assert "список счетов" in _orphans(g, "ui_field_without_api")
    assert not _orphans(g, "operation_without_endpoint")  # сохранение → POST /transfers
