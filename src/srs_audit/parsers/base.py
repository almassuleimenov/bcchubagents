"""Общие утилиты парсеров."""

from __future__ import annotations

import re

from ..models import Block

_MD_INLINE = re.compile(r"(\*\*|__|`)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def clean_inline(text: str) -> str:
    """Убирает markdown-разметку выделения, сохраняя текст дословно."""
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_INLINE.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


class SectionTracker:
    """Ведёт «хлебные крошки» заголовков для section_path."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []

    def push(self, level: int, title: str) -> str:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, title))
        return self.path

    @property
    def path(self) -> str:
        return " › ".join(t for _, t in self.stack)


class BlockBuilder:
    def __init__(self, provenance: str) -> None:
        self.blocks: list[Block] = []
        self.sections = SectionTracker()
        self.provenance = provenance
        self._table_seq = 0

    def new_table(self) -> int:
        self._table_seq += 1
        return self._table_seq

    def add(self, kind: str, text: str, line_start: int, line_end: int, **kw) -> Block | None:
        text = text.strip()
        if not text:
            return None
        if kind == "heading":
            level = kw.pop("level", 1)
            self.sections.push(level, text)
        block = Block(
            block_id=len(self.blocks) + 1,
            kind=kind,  # type: ignore[arg-type]
            text=text,
            line_start=line_start,
            line_end=max(line_start, line_end),
            section_path=self.sections.path,
            provenance=kw.pop("provenance", self.provenance),
            **kw,
        )
        self.blocks.append(block)
        return block
