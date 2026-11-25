"""Детерминированная нарезка документа на атомарные требования.

Используется как (а) офлайн-режим, (б) подсказка и страховка для LLM-экстрактора:
если LLM недоступна или вернула мусор, нарезка всё равно воспроизводима.
ID в отчёте (REQ-001 …) назначаются только кодом, в порядке следования по документу.
"""

from __future__ import annotations

import re

from razdel import sentenize

from .models import Block, ParsedDocument, Requirement

SOURCE_ID = re.compile(r"^\s*(?P<id>(?:[A-ZА-ЯЁ]{1,6}[-‑]\d{1,4}(?:\.\d+)*)|(?:[A-Z]{2,6}\d{1,4}))[.):]?\s+")
ORDERED_LINE = re.compile(r"^\s*(\d{1,3})[.)]\s")
STATUS_LINE = re.compile(r"^\s*(\d{3})\.\s")

SKIP_SECTIONS = re.compile(
    r"общие сведения|ревью требований|блок согласующ|согласован|история изменений|содержание|"
    r"лист регистрации|approval|change log",
    re.I,
)
KEY_VALUE_API_KEYS = re.compile(r"^(rest type|http method|method|метод|authorization|авторизац|request url|endpoint|curl|description)", re.I)
CHECKLIST_ROW = re.compile(r"Да\s*/\s*Нет", re.I)

KIND_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"нефункциональн|non-?functional|nfr|производительн|надёжн|надежн", re.I), "nonfunctional"),
    (re.compile(r"критери\w* при[её]мк|acceptance", re.I), "acceptance"),
    (re.compile(r"rest|api|параметр|ответ|статус|логика работы метода|особые правила|request|response", re.I), "api"),
    (re.compile(r"функциональн|functional", re.I), "functional"),
    (re.compile(r"данн|data", re.I), "data"),
    (re.compile(r"интеграц|integration", re.I), "integration"),
    (re.compile(r"\bui\b|ux|дизайн|экран|интерфейс", re.I), "ui"),
    (re.compile(r"bpmn|sequence|процесс|архитектур|схем", re.I), "process"),
    (re.compile(r"цел[ьи]|бизнес-требован|business", re.I), "business"),
    (re.compile(r"ограничен|constraint", re.I), "constraint"),
]


def classify(section_path: str) -> str:
    # ищем по самому глубокому разделу, затем по родителям
    parts = [p.strip() for p in section_path.split("›")][::-1]
    for part in parts:
        for rx, kind in KIND_RULES:
            if rx.search(part):
                return kind
    return "other"


def _table_header_map(blocks: list[Block]) -> dict[int, list[str]]:
    return {b.table_id: b.cells for b in blocks if b.kind == "table_header" and b.table_id is not None}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    buf = ""
    for s in sentenize(text):
        piece = s.text.strip()
        # «Ф-1.» или «1.» razdel отрезает как отдельное «предложение» — приклеиваем
        if buf:
            piece = f"{buf} {piece}"
            buf = ""
        if re.fullmatch(r"[A-ZА-ЯЁ]{0,6}[-‑]?\d{1,4}\.?", piece):
            buf = piece
            continue
        out.append(piece)
    if buf:
        out.append(buf)
    return out


def segment(doc: ParsedDocument) -> list[Requirement]:
    lines = doc.lines
    headers = _table_header_map(doc.blocks)
    reqs: list[Requirement] = []

    def add(block: Block, text: str, kind: str, source_id: str = "") -> None:
        reqs.append(
            Requirement(
                req_id="",
                source_id=source_id,
                section_path=block.section_path,
                kind=kind,  # type: ignore[arg-type]
                text=text.strip(),
                line_start=block.line_start,
                line_end=block.line_end,
                page=block.page,
                origin="deterministic",
            )
        )

    for b in doc.blocks:
        if b.kind in ("heading", "code", "table_header", "other"):
            continue
        if SKIP_SECTIONS.search(b.section_path):
            continue
        kind = classify(b.section_path)
        if kind == "other" and b.section_path.count("›") == 0:
            continue  # преамбула под заголовком документа — это не требования
        if b.kind == "table_row":
            header = " ".join(headers.get(b.table_id or -1, [])).lower()
            if CHECKLIST_ROW.search(b.text):
                continue
            if "ожидаем" in header or "expected" in header or kind == "acceptance":
                add(b, b.text, "acceptance")
            elif "parameter" in header or "параметр" in header:
                add(b, b.text, "api")
            elif len(b.cells) == 2 and kind == "api" and KEY_VALUE_API_KEYS.match(b.cells[0] or ""):
                add(b, b.text, "api")
            elif len(b.cells) == 2:
                continue  # метаданные «поле — значение»
            elif sum(1 for c in b.cells if c) >= 2:
                add(b, b.text, kind)
            continue

        text = b.text
        m = SOURCE_ID.match(text)
        source_id = m.group("id") if m else ""
        if not source_id and b.kind == "list_item" and 0 < b.line_start <= len(lines):
            om = ORDERED_LINE.match(lines[b.line_start - 1])
            if om:
                source_id = f"п. {om.group(1)}"
        if not source_id:
            sm = STATUS_LINE.match(text)
            if sm:
                source_id = f"HTTP {sm.group(1)}"
        if source_id or b.kind == "list_item":
            if len(text.split()) >= 3:
                add(b, text, kind, source_id)
            continue
        for sent in _sentences(text):
            if len(sent.split()) >= 3:
                add(b, sent, kind)

    for i, r in enumerate(reqs, start=1):
        r.req_id = f"REQ-{i:03d}"
    return reqs
