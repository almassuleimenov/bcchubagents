from srs_audit.grounding import Grounder, ground_quote, normalize, normalize_with_index_map

DOC = (
    "# Заголовок\n"
    "* **Ф-1.** Держатель активной карты KZT задаёт целый порог от 1 000 до 1 000 000 ₸ включительно.\n"
    "* **Ф-2.** Месяц определяется по дате совершения покупки в Алматы.\n"
    "| `cardId` | string | 1–64 | Да |\n"
    "Ответ: «Проверьте сумму» — ошибка.\n"
)


def test_normalize_map_roundtrip():
    norm, idx = normalize_with_index_map("Ёлка  —  «Да»\n**жирный**")
    assert norm == 'елка - "да" жирный'
    assert len(norm) == len(idx)
    assert idx == sorted(idx)


def test_exact_ignores_markdown_yo_and_quotes():
    g = Grounder(DOC)
    r = g.ground("Ф-1. Держатель активной карты KZT задает целый порог")
    assert r.found and r.status == "exact"
    assert r.line_start == r.line_end == 2
    r2 = g.ground('Ответ: "Проверьте сумму" - ошибка')
    assert r2.found and r2.line_start == 5


def test_table_row_without_pipes_and_backticks():
    r = ground_quote("cardId string 1–64 Да", DOC)
    assert r.found and r.line_start == 4


def test_fuzzy_match_recovers_real_span():
    # LLM допустила опечатки («определяеться», «Алмате») — это не галлюцинация, спан восстанавливается
    r = ground_quote("Месяц определяеться по дате совершения покупки в Алмате", DOC)
    assert r.found and r.status == "fuzzy"
    assert r.score >= 86
    assert r.line_start == 3
    assert "Месяц" in r.matched_text


def test_hallucinated_quote_rejected():
    r = ground_quote("Система должна поддерживать оплату в биткоинах через блокчейн", DOC)
    assert not r.found


def test_too_short_rejected():
    assert not ground_quote("да", DOC).found


def test_ellipsis_quote():
    r = ground_quote("Держатель активной карты … включительно", DOC)
    assert r.found and r.line_start == 2


def test_hint_prefers_nearby_occurrence():
    text = "порог карты\nдругое\nпорог карты\n"
    g = Grounder(text)
    assert g.ground("порог карты", hint=(3, 3)).line_start == 3
    assert g.ground("порог карты").line_start == 1


def test_normalize_idempotent():
    s = "Тест — «кавычки» ёж"
    assert normalize(normalize(s)) == normalize(s)
