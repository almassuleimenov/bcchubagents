"""Маршрутизатор форматов: md / txt / pdf / docx → ParsedDocument с координатами."""

from __future__ import annotations

from pathlib import Path

from ..models import ParsedDocument
from ..preflight import PreflightResult
from .docx_parser import parse_docx
from .md_parser import parse_markdown
from .pdf_parser import parse_pdf
from .txt_parser import parse_txt

__all__ = ["parse_document", "parse_markdown", "parse_pdf", "parse_docx", "parse_txt"]


def parse_document(pre: PreflightResult, ocr_pages: dict[int, str] | None = None) -> ParsedDocument:
    health = pre.health
    path = health.path
    stem = Path(path).stem
    fmt = health.detected_format
    if fmt == "pdf":
        doc = parse_pdf(pre.data, path, stem, ocr_pages=ocr_pages)
    elif fmt == "docx":
        doc = parse_docx(pre.data, path, stem)
    elif fmt == "md":
        doc = parse_markdown(pre.text or "", path, stem)
    else:
        text = pre.text or ""
        # .txt, в котором на самом деле markdown — разбираем AST-парсером
        if any(line.lstrip().startswith(("#", "|", "```", "* ")) for line in text.splitlines()[:200]):
            doc = parse_markdown(text, path, stem)
            doc.format = "txt(md)"
        else:
            doc = parse_txt(text, path, stem)
    if not doc.pages:
        doc.pages = health.pages
    return doc
