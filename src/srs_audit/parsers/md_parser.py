"""Markdown → блоки с точными номерами строк (markdown-it-py, token.map)."""

from __future__ import annotations

from markdown_it import MarkdownIt

from ..models import ParsedDocument
from .base import BlockBuilder, clean_inline

FENCE_LANGS = {"json", "yaml", "yml", "http", "mermaid", "curl", "bash", "sh", "text", "xml", "sql", "plantuml"}


def parse_markdown(text: str, source_path: str, title_fallback: str) -> ParsedDocument:
    md = MarkdownIt("commonmark").enable("table")
    tokens = md.parse(text)
    bb = BlockBuilder("md-ast")
    lines = text.splitlines()

    list_depth = 0
    table_id: int | None = None
    row_cells: list[str] = []
    row_map: list[int] | None = None
    in_thead = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        typ = t.type
        if typ == "heading_open":
            inline = tokens[i + 1]
            level = int(t.tag[1:]) if t.tag[1:].isdigit() else 1
            ls, le = (t.map or [0, 1])
            bb.add("heading", clean_inline(inline.content), ls + 1, le, level=level)
            i += 3
            continue
        if typ in ("bullet_list_open", "ordered_list_open"):
            list_depth += 1
        elif typ in ("bullet_list_close", "ordered_list_close"):
            list_depth -= 1
        elif typ == "paragraph_open":
            inline = tokens[i + 1]
            ls, le = (t.map or [0, 1])
            kind = "list_item" if list_depth > 0 else "paragraph"
            # многострочный абзац: каждая «жёсткая» строка с переносом <br> — отдельный блок не нужен,
            # но абзацы вида «**200.** ...  \n**400.** ...» удобнее разрезать построчно
            raw_lines = [ln for ln in inline.content.split("\n")]
            if kind == "paragraph" and len(raw_lines) > 1 and all(ln.strip() for ln in raw_lines):
                for off, ln in enumerate(raw_lines):
                    bb.add(kind, clean_inline(ln), ls + 1 + off, ls + 1 + off)
            else:
                bb.add(kind, clean_inline(inline.content), ls + 1, le)
            i += 3
            continue
        elif typ == "fence" or typ == "code_block":
            ls, le = (t.map or [0, 1])
            lang = (t.info or "").strip().split(" ")[0].lower() if typ == "fence" else ""
            lang = lang if lang in FENCE_LANGS else (lang or "text")
            # строки содержимого без ограждающих ```
            start = ls + 2 if typ == "fence" else ls + 1
            end = le - 1 if typ == "fence" else le
            bb.add("code", t.content.rstrip("\n"), start, max(start, end), lang=lang)
        elif typ == "table_open":
            table_id = bb.new_table()
        elif typ == "table_close":
            table_id = None
        elif typ == "thead_open":
            in_thead = True
        elif typ == "thead_close":
            in_thead = False
        elif typ == "tr_open":
            row_cells = []
            row_map = t.map
        elif typ in ("th_open", "td_open"):
            inline = tokens[i + 1]
            row_cells.append(clean_inline(inline.content))
            i += 3
            continue
        elif typ == "tr_close":
            ls, le = (row_map or [0, 1])
            kind = "table_header" if in_thead else "table_row"
            text_row = " | ".join(c for c in row_cells)
            bb.add(kind, text_row, ls + 1, le, table_id=table_id, cells=list(row_cells))
        elif typ == "html_block":
            ls, le = (t.map or [0, 1])
            bb.add("other", t.content, ls + 1, le)
        i += 1

    title = next((b.text for b in bb.blocks if b.kind == "heading"), title_fallback)
    return ParsedDocument(
        source_path=source_path,
        title=title,
        format="md",
        text="\n".join(lines),
        blocks=bb.blocks,
    )
