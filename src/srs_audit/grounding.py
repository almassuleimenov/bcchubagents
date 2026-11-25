"""Детерминированный верификатор цитат (anti-hallucination).

Любая цитата, которую вернула LLM, проходит через ``Grounder.ground``:

1. нормализация с картой индексов (ё→е, кавычки, тире, markdown, пробелы, регистр);
2. точный поиск в нормализованном тексте документа;
3. нечёткий поиск ``rapidfuzz.fuzz.partial_ratio_alignment`` с порогом 86;
4. обратное отображение в координаты исходника: строки и символы.

Если цитата не найдена — агент получает ``ModelRetry`` с перечнем «не найденных»
цитат (до 2 попыток), после чего неподтверждённые гипотезы уходят в карантин и
в отчёт НЕ попадают. Координаты и оценку заполняет только этот модуль.
"""

from __future__ import annotations

import bisect
import re
import unicodedata
from dataclasses import dataclass

from rapidfuzz import fuzz

from .models import CitationStats

FUZZY_CUTOFF = 86.0
MIN_QUOTE_CHARS = 4
MIN_FUZZY_CHARS = 12

_CHAR_MAP: dict[str, str] = {
    "ё": "е",
    "«": '"',
    "»": '"',
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "'": '"',
    "‘": '"',
    "’": '"',
    "‚": '"',
    "‛": '"',
    "`": "",
    "*": "",
    "\\": "",
    "|": " ",
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "―": "-",
    "−": "-",
    "…": "...",
    " ": " ",
    " ": " ",
    " ": " ",
    "​": "",
    "﻿": "",
}
_ELLIPSIS_SPLIT = re.compile(r"\s*(?:\.\.\.|…|\[\.\.\.\])\s*")


def normalize_with_index_map(text: str) -> tuple[str, list[int]]:
    """Нормализует текст и возвращает карту: индекс нормализованного символа → индекс исходного.

    >>> normalize_with_index_map("Ёлка  —«да»")[0]
    'елка -"да"'
    """
    out: list[str] = []
    idx: list[int] = []
    prev_space = True  # схлопываем ведущие пробелы
    for i, ch in enumerate(text):
        folded = ch.casefold()
        rep = _CHAR_MAP.get(folded, _CHAR_MAP.get(ch, folded))
        if not rep:
            continue
        for c in rep:
            if c.isspace() or unicodedata.category(c) == "Zs":
                if prev_space:
                    continue
                out.append(" ")
                idx.append(i)
                prev_space = True
            else:
                out.append(c)
                idx.append(i)
                prev_space = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(text: str) -> str:
    return normalize_with_index_map(text)[0]


@dataclass
class GroundingResult:
    found: bool
    status: str  # exact | fuzzy | not_found | too_short
    score: float = 0.0
    char_start: int = 0
    char_end: int = 0
    line_start: int = 0
    line_end: int = 0
    page: int | None = None
    matched_text: str = ""

    @property
    def location(self) -> str:
        loc = f"стр. {self.line_start}" if self.line_start == self.line_end else f"стр. {self.line_start}–{self.line_end}"
        return f"с. {self.page}, {loc}" if self.page else loc


class Grounder:
    """Верификатор цитат по каноническому тексту документа."""

    def __init__(self, text: str, line_pages: dict[int, int] | None = None) -> None:
        self.text = text
        self.norm, self.idx = normalize_with_index_map(text)
        self.line_offsets = [0] + [m.end() for m in re.finditer("\n", text)]
        self.line_pages = line_pages or {}
        self.valid_req_ids: set[str] = set()
        self.stats = CitationStats()

    # -- координаты ----------------------------------------------------------
    def line_of(self, char_pos: int) -> int:
        return bisect.bisect_right(self.line_offsets, char_pos)

    def _result(self, n_start: int, n_end: int, status: str, score: float) -> GroundingResult:
        o_start = self.idx[n_start]
        o_end = self.idx[max(n_start, n_end - 1)] + 1
        ls, le = self.line_of(o_start), self.line_of(max(o_start, o_end - 1))
        return GroundingResult(
            found=True,
            status=status,
            score=round(score, 1),
            char_start=o_start,
            char_end=o_end,
            line_start=ls,
            line_end=le,
            page=self.line_pages.get(ls),
            matched_text=self.text[o_start:o_end],
        )

    # -- поиск ---------------------------------------------------------------
    def _ground_piece(self, qn: str, hint: tuple[int, int] | None) -> GroundingResult:
        if len(qn) < MIN_QUOTE_CHARS:
            return GroundingResult(False, "too_short")
        # 1. точное вхождение (из нескольких — ближайшее к подсказке)
        positions: list[int] = []
        start = self.norm.find(qn)
        while start != -1 and len(positions) < 50:
            positions.append(start)
            start = self.norm.find(qn, start + 1)
        if positions:
            best = positions[0]
            if hint and len(positions) > 1:
                hs, he = hint

                def dist(p: int) -> int:
                    line = self.line_of(self.idx[p])
                    return 0 if hs <= line <= he else min(abs(line - hs), abs(line - he))

                best = min(positions, key=dist)
            return self._result(best, best + len(qn), "exact", 100.0)
        # 2. нечёткое выравнивание
        if len(qn) < MIN_FUZZY_CHARS or len(qn) > len(self.norm):
            return GroundingResult(False, "not_found")
        al = fuzz.partial_ratio_alignment(qn, self.norm, score_cutoff=FUZZY_CUTOFF)
        if al is None or al.dest_end <= al.dest_start:
            return GroundingResult(False, "not_found")
        # подрезаем по границам слов, чтобы matched_text был читаемым
        ds, de = al.dest_start, al.dest_end
        while ds > 0 and self.norm[ds - 1].isalnum() and self.norm[ds].isalnum():
            ds -= 1
        while de < len(self.norm) and self.norm[de - 1].isalnum() and self.norm[de].isalnum():
            de += 1
        return self._result(ds, de, "fuzzy", al.score)

    def ground(self, quote: str, hint: tuple[int, int] | None = None) -> GroundingResult:
        """Ищет цитату в документе. Поддерживает эллипсисы «...» внутри цитаты."""
        q = (quote or "").strip().strip('"«»“”').strip()
        pieces = [p for p in _ELLIPSIS_SPLIT.split(q) if p.strip()]
        if not pieces:
            return GroundingResult(False, "too_short")
        if len(pieces) == 1:
            return self._ground_piece(normalize(pieces[0]), hint)
        results = [self._ground_piece(normalize(p), hint) for p in pieces if len(normalize(p)) >= MIN_QUOTE_CHARS]
        if not results or not all(r.found for r in results):
            return GroundingResult(False, "not_found")
        first, last = min(results, key=lambda r: r.char_start), max(results, key=lambda r: r.char_end)
        if last.line_end - first.line_start > 6:  # куски слишком далеко друг от друга — склейка
            return GroundingResult(False, "not_found")
        score = min(r.score for r in results)
        status = "exact" if all(r.status == "exact" for r in results) else "fuzzy"
        res = GroundingResult(
            True,
            status,
            score,
            first.char_start,
            last.char_end,
            first.line_start,
            last.line_end,
            self.line_pages.get(first.line_start),
            quote.strip(),
        )
        return res

    def req_exists(self, req_id: str) -> bool:
        return req_id in self.valid_req_ids or req_id.upper() in {"SET", "НАБОР", ""}

    # -- учёт для Citation Audit Badge -------------------------------------
    def record(self, res: GroundingResult, *, after_retry: bool = False) -> None:
        self.stats.checked += 1
        if not res.found:
            return
        if res.status == "exact":
            self.stats.exact += 1
        else:
            self.stats.fuzzy += 1
        if after_retry:
            self.stats.corrected_via_retry += 1


def ground_quote(quote: str, text: str, hint: tuple[int, int] | None = None) -> GroundingResult:
    """Удобная функция для разовой проверки."""
    return Grounder(text).ground(quote, hint)
