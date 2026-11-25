"""Простой текст → блоки (абзацы, пункты списков, эвристические заголовки, таблицы с «|»)."""

from __future__ import annotations

import re

from ..models import ParsedDocument
from .base import BlockBuilder

_LIST = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)]|[a-zа-я][.)])\s+")
_NUM_HEADING = re.compile(r"^\s*(\d+(?:\.\d+){0,4})\.?\s+(\S.{1,90})$")
_REQ_ID_LINE = re.compile(r"^\s*[A-ZА-ЯЁ]{1,6}[-‑.]?\d{1,4}[.)]")


def _heading_level(line: str) -> int | None:
    s = line.strip()
    if not s or len(s) > 100:
        return None
    m = _NUM_HEADING.match(s)
    if m and not s.endswith((".", ";", ",")) and len(s.split()) <= 12 and not _REQ_ID_LINE.match(s):
        return m.group(1).count(".") + 1
    if s.isupper() and 3 <= len(s) <= 80 and any(ch.isalpha() for ch in s):
        return 1
    return None


def parse_text_lines(
    lines: list[str], source_path: str, title_fallback: str, fmt: str = "txt", provenance: str = "txt"
) -> ParsedDocument:
    bb = BlockBuilder(provenance)
    para: list[str] = []
    para_start = 0
    table_id: int | None = None

    def flush(end_line: int) -> None:
        nonlocal para
        if para:
            bb.add("paragraph", " ".join(s.strip() for s in para), para_start, end_line)
            para = []

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip()
        s = line.strip()
        if not s:
            flush(idx - 1)
            table_id = None
            continue
        if s.count("|") >= 2:
            flush(idx - 1)
            if table_id is None:
                table_id = bb.new_table()
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            bb.add("table_row", " | ".join(cells), idx, idx, table_id=table_id, cells=cells)
            continue
        table_id = None
        level = _heading_level(s)
        if level is not None:
            flush(idx - 1)
            bb.add("heading", s, idx, idx, level=level)
            continue
        if _LIST.match(s) or _REQ_ID_LINE.match(s):
            flush(idx - 1)
            bb.add("list_item", s, idx, idx)
            continue
        if not para:
            para_start = idx
        para.append(s)
    flush(len(lines))

    title = next((b.text for b in bb.blocks if b.kind == "heading"), title_fallback)
    return ParsedDocument(source_path=source_path, title=title, format=fmt, text="\n".join(lines), blocks=bb.blocks)


def parse_txt(text: str, source_path: str, title_fallback: str) -> ParsedDocument:
    return parse_text_lines(text.splitlines(), source_path, title_fallback)
