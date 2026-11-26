"""Детерминированный пост-фильтр ложных срабатываний LLM.

Срабатывает только на проверяемые кодом признаки: правка «Было = Стало», ложная «арифметическая
ошибка» в расчёте, который на самом деле сходится, и претензия к валидному ``curl -G --data``.
Отброшенные находки не исчезают молча: они попадают в отчёт в таблицу отсеянных ложных срабатываний.
"""

from __future__ import annotations

import itertools
import re

from .grounding import is_noop_fix
from .models import AuditDefect

# «48 500», «1 000 000», «1500», «50 000,00» → int (дробная часть отбрасывается)
NUMBER = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d])|(?<![\d.,])\d+(?![\d])")
ARITH_CLAIM = re.compile(
    r"арифмет|математическ|не сход|неверн\w*\s+(?:расч|подсч|сумм|итог)|ошибк\w*\s+(?:в\s+)?(?:расч|подсч|сумм)"
    r"|не соответству\w*\s+(?:сумм|расч|итог)|опечатк",
    re.I,
)
CURL_G = re.compile(r"(?<![\w-])-G\b|--get\b")
CURL_DATA_CLAIM = re.compile(r"--data|(?<![\w-])-d\b|\bdata\b|тел[оау]\b|телом|body", re.I)
# «Лишнее проектное решение / перенесите в раздел архитектуры» — претензия к уместности описания
_DESIGN_WORDS = r"(?:архитектур|дизайн|\bui\b|\bux\b|интерфейс|экран|макет|bpmn|процесс|sequence|последовательност|диаграмм|проектировани)"
DESIGN_CLAIM = re.compile(
    r"лишн\w*\s+проектн\w*\s+решени|проектн\w*\s+решени\w*[^.]{0,80}(?:вместо|а не|не\s+требовани)"
    r"|implementation[- ]free|(?:не является|а не)\s+требовани\w*[^.]{0,60}" + _DESIGN_WORDS
    + r"|" + _DESIGN_WORDS + r"\w*[^.]{0,80}(?:не является|а не)\s+требовани"
    r"|(?:перемест|перенест|перенес|вынест|вынес)\w*[^.]{0,80}(?:раздел|документ|приложени|спецификаци)\w*[^.]{0,40}" + _DESIGN_WORDS,
    re.I,
)
# Разделы, само назначение которых — описывать архитектуру, UI, процесс или взаимодействие
DESCRIPTIVE_SECTION = re.compile(
    r"архитектур|дизайн|\bui\b|\bux\b|интерфейс|экран|макет|bpmn|sequence|последовательност|диаграмм|схем\w*\s+процесс",
    re.I,
)
# если претензия на самом деле про семантику GET (изменение состояния) — это не ложное срабатывание
GET_SEMANTICS = re.compile(r"сохран|измен\w*\s+состояни|мутир|небезопасн|кэшир", re.I)


def _numbers(text: str) -> list[int]:
    out = []
    for m in NUMBER.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if digits and len(digits) <= 12:
            out.append(int(digits))
    return out


def arithmetic_is_consistent(text: str) -> bool:
    """В тексте есть тройка чисел a ± b = c — расчёт сходится."""
    nums = [n for n in dict.fromkeys(_numbers(text)) if n > 0]
    for a, b in itertools.permutations(nums, 2):
        for c in nums:
            if c not in (a, b) and (a + b == c or a - b == c):
                return True
    return False


def false_positive_reason(d: AuditDefect, section_path: str = "") -> str | None:
    """Причина отбросить дефект или ``None``. Применяется к находкам LLM.

    ``section_path`` — раздел требования, к которому относится дефект (путь заголовков).
    """
    if section_path and DESCRIPTIVE_SECTION.search(section_path) and DESIGN_CLAIM.search(f"{d.reasoning} {d.recommendation}"):
        last = section_path.split("›")[-1].strip()
        return f"описание архитектуры/UI/процесса стоит в своём разделе («{last}») — это не лишнее проектное решение"
    if is_noop_fix(d.exact_quote, d.recommendation):
        return "предложенная правка дословно совпадает с исходным текстом (before == after)"
    # считаем только по цитате критерия: в reasoning модель пишет свой (возможно, ошибочный) расчёт
    if ARITH_CLAIM.search(d.reasoning) and arithmetic_is_consistent(d.exact_quote):
        return "заявлена арифметическая ошибка, но расчёт в критерии сходится"
    blob = f"{d.exact_quote} {d.reasoning}"
    if (
        "curl" in blob.lower()
        and CURL_G.search(blob)
        and CURL_DATA_CLAIM.search(d.reasoning)
        and not GET_SEMANTICS.search(d.reasoning)
    ):
        return "`curl -G --data` — валидный синтаксис: флаг -G передаёт --data как query-параметры"
    return None


def split_false_positives(
    defects: list[AuditDefect], sections: dict[str, str] | None = None
) -> tuple[list[AuditDefect], list[tuple[AuditDefect, str]]]:
    """``sections`` — {req_id: section_path} для проверки «уместности» описательных разделов."""
    sections = sections or {}
    kept, dropped = [], []
    for d in defects:
        why = false_positive_reason(d, sections.get(d.req_id, "")) if d.source != "deterministic" else None
        if why:
            dropped.append((d, why))
        else:
            kept.append(d)
    return kept, dropped


# ---------------------------------------------------------------------------
# Матрица A ↔ B: статус должен следовать из комментария
# ---------------------------------------------------------------------------
# Модель нередко ставит «consistent» и тут же пишет в комментарии «GET … нарушает RFC 9110» или
# «integer в запросе, string в ответе». Комментарий — довод, статус — вывод: если довод описывает
# расхождение или пробел, статус «согласовано» ложен. Код только ужесточает статус, никогда не смягчает.

_NO_PROBLEM = re.compile(
    r"(?:нет|без|отсутству\w*|не\s+выявлен\w*|не\s+обнаружен\w*)\s+(?:каких-либо\s+)?"
    r"(?:противореч\w*|расхожд\w*|несоответств\w*|пробел\w*|проблем\w*)"
    r"|(?:противореч|расхожд|несоответств|пробел|проблем)\w*\s+(?:нет|не\s+выявлен\w*|не\s+обнаружен\w*)",
    re.I,
)
_MISMATCH = re.compile(
    r"противореч|не\s+совпада|расход[яи]т|расхожд|несоответств|наруша|вместо|подменя"
    r"|\binteger\b[^.]{0,80}\bstring\b|\bstring\b[^.]{0,80}\binteger\b"
    r"|\b200\b[^.]{0,80}\b(?:400|error)\b|\b403\b[^.]{0,80}\b404\b|\b404\b[^.]{0,80}\b403\b"
    r"|тиын\w*[^.]{0,80}тенге|тенге[^.]{0,80}тиын",
    re.I,
)
_MISSING = re.compile(
    r"отсутству|отсутстви|не\s+(?:описан|указан|определ[её]н|предусмотрен|реализован|возвраща|переда)"
    r"|недоступ|нет\s+(?:метод|пол|эндпоинт|endpoint|параметр|механизм|заголов|способ)",
    re.I,
)
_IN_BUSINESS = re.compile(r"(?:в|из)\s+(?:част\w*\s+а\b|бизнес-?\w*|бт\b|ui\b|интерфейс\w*)", re.I)
_RISK = re.compile(r"риск|неоднозначн|неясн|не\s+уточн|недостаточн|может\s+привести", re.I)


_SEVERITY = {"consistent": 0, "ambiguous": 1, "risk": 2, "missing_in_business": 3, "missing_in_api": 3, "mismatch": 3}


def _status_from_comment(comment: str, api_quote: str) -> str:
    text = _NO_PROBLEM.sub(" ", comment or "")
    if _MISMATCH.search(text):
        return "mismatch"
    if _MISSING.search(text):
        return "missing_in_business" if _IN_BUSINESS.search(text) and api_quote else "missing_in_api"
    if _RISK.search(text):
        return "risk"
    return "consistent"


def reconcile_matrix_status(status: str, comment: str, api_quote: str = "") -> str:
    """Статус строки матрицы, согласованный с её комментарием: только ужесточение, никогда не смягчение.
    «consistent»/«ambiguous»/«risk» поднимаются до уровня, который описан в комментарии
    (integer ↔ string — расхождение, а не риск); между missing и mismatch решает модель."""
    derived = _status_from_comment(comment, api_quote)
    return derived if _SEVERITY[derived] > _SEVERITY.get(status, 0) else status
