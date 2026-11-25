"""DOCX → блоки в строгом порядке абзацев и таблиц (python-docx ``iter_inner_content``)."""

from __future__ import annotations

import io
import re

from ..models import ParsedDocument
from .base import BlockBuilder

_HEADING_STYLE = re.compile(r"^(?:heading|заголовок)\s*(\d)", re.I)
_LIST_TEXT = re.compile(r"^\s*(?:[-*•–]|\d{1,2}[.)]|[A-ZА-ЯЁ]{1,6}[-‑]\d{1,4}[.)])\s+")


def parse_docx(data: bytes, source_path: str, title_fallback: str) -> ParsedDocument:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(io.BytesIO(data))
    bb = BlockBuilder("docx")
    canonical: list[str] = []

    def emit_paragraph(p: Paragraph) -> None:
        text = " ".join(p.text.split())
        if not text:
            return
        canonical.append(text)
        ln = len(canonical)
        style = (p.style.name if p.style is not None else "") or ""
        m = _HEADING_STYLE.match(style)
        if m or style.lower() == "title":
            bb.add("heading", text, ln, ln, level=int(m.group(1)) if m else 1)
            return
        has_num = p._p.pPr is not None and p._p.pPr.numPr is not None  # noqa: SLF001
        kind = "list_item" if has_num or "list" in style.lower() or _LIST_TEXT.match(text) else "paragraph"
        bb.add(kind, text, ln, ln)

    def emit_table(t: Table) -> None:
        tid = bb.new_table()
        for r_i, row in enumerate(t.rows):
            cells: list[str] = []
            prev = None
            for cell in row.cells:
                # объединённые ячейки python-docx повторяет — убираем дубли подряд
                if prev is not None and cell._tc is prev:  # noqa: SLF001
                    continue
                prev = cell._tc  # noqa: SLF001
                cells.append(" ".join(cell.text.split()))
            if not any(cells):
                continue
            line = " | ".join(cells)
            canonical.append(line)
            ln = len(canonical)
            bb.add("table_header" if r_i == 0 else "table_row", line, ln, ln, table_id=tid, cells=cells)

    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            emit_paragraph(item)
        elif isinstance(item, Table):
            emit_table(item)

    title = (document.core_properties.title or "").strip()
    if not title:
        title = next((b.text for b in bb.blocks if b.kind == "heading"), title_fallback)
    return ParsedDocument(source_path=source_path, title=title, format="docx", text="\n".join(canonical), blocks=bb.blocks)
