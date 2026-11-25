"""Markdown-отчёт на русском (Jinja2) — ровно 6 обязательных разделов README хакатона."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from ..models import (
    CATEGORY_RU,
    FINTECH_DOMAIN_RU,
    SET_PROPERTY_RU,
    SEVERITY_ORDER,
    AuditDefect,
    AuditReport,
)

TEMPLATES = Path(__file__).parent / "templates"

SEV_ICON = {"Blocker": "⛔", "Critical": "🔴", "Major": "🟠", "Minor": "🟡"}
VERDICT_ICON = {"GO": "🟢", "CONDITIONAL GO": "🟡", "NO-GO": "🔴", "UNPROCESSABLE": "⚫"}
VERDICT_TEXT = {
    "GO": "Документ готов: можно начинать разработку и тестирование.",
    "CONDITIONAL GO": "Можно начинать разработку только частей, не затронутых критичными дефектами; "
    "до старта тестирования дефекты Critical/Major должны быть закрыты.",
    "NO-GO": "Начинать разработку и тестирование по этому документу НЕЛЬЗЯ: сначала устранить критичные дефекты "
    "и ответить на уточняющие вопросы.",
    "UNPROCESSABLE": "Документ не может быть обработан — проверка требований не выполнялась.",
}
SET_VERDICT = {"pass": "✅ соответствует", "partial": "⚠️ частично", "fail": "❌ не соответствует"}
DOMAIN_STATUS = {"ok": "✅ OK", "risk": "⚠️ риск", "gap": "❌ пробел", "n_a": "— н/п"}
TRACE_STATUS = {
    "consistent": "✓ согласуется",
    "contradiction": "✗ противоречие",
    "not_covered": "! правил недостаточно",
    "ambiguous": "? неоднозначно",
}
MATRIX_STATUS = {
    "consistent": "✅ согласовано",
    "mismatch": "❌ расхождение",
    "missing_in_api": "❌ нет в API",
    "missing_in_business": "⚠️ нет в части А",
    "ambiguous": "⚠️ неоднозначно",
}
ROLE_RU = {
    "PO": "Product Owner — бизнес-правила",
    "BA": "Бизнес-аналитик — формулировки и сценарии",
    "SA": "Системный аналитик / архитектор — API, данные, интеграции",
    "QA": "QA — критерии приёмки и тест-дизайн",
    "InfoSec": "Информационная безопасность — PCI DSS, ПДн",
}
KIND_RU = {
    "business": "бизнес-цель",
    "functional": "функц.",
    "nonfunctional": "НФТ",
    "data": "данные",
    "integration": "интеграция",
    "ui": "UI",
    "process": "процесс",
    "acceptance": "приёмка",
    "api": "API",
    "constraint": "ограничение",
    "other": "прочее",
}
SOURCE_RU = {
    "deterministic": "правило",
    "llm_quality": "LLM · QualityFilter",
    "llm_crosscheck": "LLM · CrossCheck",
}
CRITICAL_GROUPS = OrderedDict(
    [
        ("verifiability", "Непроверяемые формулировки"),
        ("consistency", "Противоречия"),
        ("completeness", "Неполные формулировки и пропуски"),
    ]
)


def cell(value: object, limit: int = 0) -> str:
    s = " ".join(str(value or "").split())
    if limit and len(s) > limit:
        s = s[: limit - 1].rsplit(" ", 1)[0] + "…"
    return s.replace("|", "\\|")


def quote(value: object, limit: int = 220) -> str:
    return "«" + cell(value, limit) + "»"


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
        autoescape=False,
    )
    env.filters["cell"] = cell
    env.filters["quote"] = quote
    return env


def _req_rows(report: AuditReport) -> list[dict]:
    by_req: dict[str, list[AuditDefect]] = {}
    for d in report.defects:
        by_req.setdefault(d.req_id, []).append(d)
    rows = []
    for r in report.requirements:
        ds = sorted(by_req.get(r.req_id, []), key=lambda d: SEVERITY_ORDER[d.severity])
        worst = ds[0].severity if ds else None
        status = "✅ OK" if not ds else ("❌ " if worst in ("Blocker", "Critical") else "⚠️ ") + worst
        props = sorted({CATEGORY_RU[d.category] for d in ds})
        rows.append({"req": r, "defects": ds, "status": status, "props": props})
    return rows


def build_context(report: AuditReport) -> dict:
    m = report.metrics
    set_defects = [d for d in report.defects if d.scope == "set" or d.req_id == "SET"]
    critical_groups = OrderedDict()
    for cat, title in CRITICAL_GROUPS.items():
        items = [d for d in report.defects if d.category == cat and d.severity in ("Blocker", "Critical", "Major")]
        if items:
            critical_groups[title] = items
    other_critical = [
        d for d in report.defects if d.severity in ("Blocker", "Critical") and d.category not in CRITICAL_GROUPS
    ]
    roles = OrderedDict()
    for role in ("PO", "BA", "SA", "QA", "InfoSec"):
        qs = [q for q in report.questions if q.target_role == role]
        if qs:
            roles[role] = qs
    det = [d for d in report.defects if d.source == "deterministic"]
    llm = [d for d in report.defects if d.source != "deterministic"]
    by_cat = [(CATEGORY_RU.get(k, k), v) for k, v in m.by_category.items()]
    ok_reqs = [row["req"].req_id for row in _req_rows(report) if not row["defects"]]
    return dict(
        r=report,
        p=report.passport,
        h=report.health,
        m=m,
        c=report.citation,
        req_rows=_req_rows(report),
        ok_reqs=ok_reqs,
        set_defects=set_defects,
        critical_groups=critical_groups,
        other_critical=other_critical,
        roles=roles,
        det_count=len(det),
        llm_count=len(llm),
        llm_exact=sum(1 for d in llm if d.grounding_status == "exact"),
        by_cat=by_cat,
        SEV_ICON=SEV_ICON,
        VERDICT_ICON=VERDICT_ICON,
        VERDICT_TEXT=VERDICT_TEXT,
        SET_VERDICT=SET_VERDICT,
        SET_PROPERTY_RU=SET_PROPERTY_RU,
        DOMAIN_STATUS=DOMAIN_STATUS,
        FINTECH_DOMAIN_RU=FINTECH_DOMAIN_RU,
        MATRIX_STATUS=MATRIX_STATUS,
        TRACE_STATUS=TRACE_STATUS,
        ROLE_RU=ROLE_RU,
        KIND_RU=KIND_RU,
        SOURCE_RU=SOURCE_RU,
        CATEGORY_RU=CATEGORY_RU,
    )


def render_markdown(report: AuditReport) -> str:
    env = _env()
    name = "unprocessable.md.j2" if report.metrics.verdict == "UNPROCESSABLE" else "report.md.j2"
    return env.get_template(name).render(**build_context(report))


def render_json(report: AuditReport) -> str:
    return report.model_dump_json(indent=2)
