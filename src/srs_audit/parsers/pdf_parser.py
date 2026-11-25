"""PDF → блоки: pymupdf get_text("dict") + хирургический pdfplumber на страницах с таблицами.

Канонический текст строится построчно (по одной визуальной строке PDF на строку),
таблицы — по строке на строку таблицы с разделителем « | ». Номер строки в отчёте —
номер строки канонического текста; страница хранится в каждом блоке.
Страницы-сканы заменяются текстом Vision-распознавания, если он передан (``ocr_pages``).
"""

from __future__ import annotations

import io
import re
import statistics
from dataclasses import dataclass, field

from ..models import ParsedDocument
from .base import BlockBuilder

_LIST = re.compile(r"^\s*(?:[-*•–▪●]|\d{1,2}[.)]|[A-ZА-ЯЁ]{1,6}[-‑]\d{1,4}[.)])\s*")


@dataclass
class _Item:
    y0: float
    y1: float
    text: str
    size: float = 0.0
    bold: bool = False
    cells: list[str] = field(default_factory=list)
    is_table_row: bool = False
    table_key: int = 0


def _in_bbox(bbox: tuple[float, float, float, float], rects: list[tuple[float, float, float, float]]) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return any(r[0] - 1 <= cx <= r[2] + 1 and r[1] - 1 <= cy <= r[3] + 1 for r in rects)


def _plumber_tables(data: bytes, pno: int) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(data)) as pdf:
            page = pdf.pages[pno]
            out = []
            for tab in page.find_tables():
                rows = [[(c or "").replace("\n", " ").strip() for c in row] for row in tab.extract()]
                out.append((tuple(tab.bbox), rows))
            return out
    except Exception:
        return []


def parse_pdf(
    data: bytes, source_path: str, title_fallback: str, ocr_pages: dict[int, str] | None = None
) -> ParsedDocument:
    import pymupdf

    ocr_pages = ocr_pages or {}
    doc = pymupdf.open(stream=data, filetype="pdf")
    if doc.needs_pass:
        doc.authenticate("")

    canonical: list[str] = []
    bb = BlockBuilder("pymupdf")
    sizes: list[float] = []
    pages_items: list[tuple[int, list[_Item], str]] = []

    for pno in range(doc.page_count):
        page = doc[pno]
        page_no = pno + 1
        if page_no in ocr_pages:
            items = [_Item(i, i, ln) for i, ln in enumerate(ocr_pages[page_no].splitlines()) if ln.strip()]
            pages_items.append((page_no, items, "vision"))
            continue

        # 1) таблицы: pymupdf детектирует, pdfplumber извлекает «хирургически» только на этой странице
        table_rects: list[tuple[float, float, float, float]] = []
        table_items: list[_Item] = []
        try:
            found = page.find_tables()
            mu_tables = list(found.tables) if found else []
        except Exception:
            mu_tables = []
        if mu_tables:
            plumbed = _plumber_tables(data, pno)
            source = plumbed or [(tuple(t.bbox), t.extract()) for t in mu_tables]
            for key, (bbox, rows) in enumerate(source, start=1):
                table_rects.append(bbox)  # type: ignore[arg-type]
                step = (bbox[3] - bbox[1]) / max(1, len(rows))
                for r_i, row in enumerate(rows):
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if not any(cells):
                        continue
                    y = bbox[1] + r_i * step
                    table_items.append(
                        _Item(y, y + step, " | ".join(cells), cells=cells, is_table_row=True, table_key=key)
                    )

        # 2) текстовые строки
        items: list[_Item] = []
        d = page.get_text("dict")
        for blk in d.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                text = "".join(s.get("text", "") for s in spans).strip()
                if not text:
                    continue
                bbox = tuple(line.get("bbox", (0, 0, 0, 0)))
                if table_rects and _in_bbox(bbox, table_rects):  # type: ignore[arg-type]
                    continue
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold = any((s.get("flags", 0) & 16) or "Bold" in s.get("font", "") for s in spans)
                sizes.append(size)
                items.append(_Item(bbox[1], bbox[3], text, size=size, bold=bold))
        items.extend(table_items)
        items.sort(key=lambda it: it.y0)
        pages_items.append((page_no, items, "pymupdf"))

    body = statistics.median(sizes) if sizes else 10.0

    for page_no, items, prov in pages_items:
        para: list[tuple[int, _Item]] = []
        table_id: int | None = None
        last_key = -1

        def flush() -> None:
            nonlocal para
            if para:
                bb.add(
                    "list_item" if _LIST.match(para[0][1].text) else "paragraph",
                    " ".join(it.text for _, it in para),
                    para[0][0],
                    para[-1][0],
                    page=page_no,
                    provenance=prov,
                )
                para = []

        for it in items:
            canonical.append(it.text)
            ln = len(canonical)
            if it.is_table_row:
                flush()
                if it.table_key != last_key:
                    table_id = bb.new_table()
                    last_key = it.table_key
                    kind = "table_header"
                else:
                    kind = "table_row"
                bb.add(kind, it.text, ln, ln, page=page_no, table_id=table_id, cells=it.cells, provenance="pdfplumber")
                continue
            last_key = -1
            is_heading = prov != "vision" and (
                (it.size >= body * 1.18 and len(it.text) < 120)
                or (it.bold and len(it.text) < 90 and not it.text.endswith((".", ";", ",")) and not _LIST.match(it.text))
            )
            if is_heading:
                flush()
                level = 1 if it.size >= body * 1.6 else 2 if it.size >= body * 1.3 else 3
                bb.add("heading", it.text, ln, ln, page=page_no, level=level, provenance=prov)
                continue
            if _LIST.match(it.text) and para:
                flush()
            if para:
                prev = para[-1][1]
                gap = it.y0 - prev.y1
                if prov != "vision" and gap > max(4.0, (prev.y1 - prev.y0) * 0.9):
                    flush()
            para.append((ln, it))
            if it.text.endswith((".", ":", ";")) and prov == "vision":
                flush()
        flush()

    pages = doc.page_count
    doc.close()
    title = next((b.text for b in bb.blocks if b.kind == "heading"), title_fallback)
    return ParsedDocument(
        source_path=source_path, title=title, format="pdf", text="\n".join(canonical), blocks=bb.blocks, pages=pages
    )
