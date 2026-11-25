"""Компактное представление документа и артефактов для промптов."""

from __future__ import annotations

from ..models import AuditDefect, ParsedDocument, Requirement

MAX_DOC_CHARS = 120_000


def numbered_document(doc: ParsedDocument) -> str:
    out = []
    total = 0
    for i, line in enumerate(doc.lines, start=1):
        s = f"{i:04d}│ {line}"
        total += len(s)
        if total > MAX_DOC_CHARS:
            out.append("…(документ усечён для контекста модели)…")
            break
        out.append(s)
    return "\n".join(out)


def requirement_lines(reqs: list[Requirement]) -> str:
    return "\n".join(
        f"{r.req_id} [{r.source_id or '—'}] ({r.kind}, стр. {r.line_start}"
        f"{'–' + str(r.line_end) if r.line_end != r.line_start else ''}): {r.text}"
        for r in reqs
    )


def candidate_lines(cands: list[AuditDefect]) -> str:
    if not cands:
        return "(кандидатов нет)"
    return "\n".join(
        f"{c.defect_id} [{c.rule_id}, {c.severity}] {c.req_id}: «{c.exact_quote[:160]}» — {c.reasoning}" for c in cands
    )


def defect_lines(defects: list[AuditDefect]) -> str:
    return "\n".join(
        f"{d.defect_id} [{d.severity}/{d.category}] {d.req_id} ({d.verified_location}): «{d.exact_quote[:180]}» — "
        f"{d.reasoning}"
        for d in defects
    )
