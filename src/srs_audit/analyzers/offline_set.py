"""Детерминированные выводы уровня набора: используются в офлайн-режиме и как страховка,
если LLM вернула неполный ответ (например, оценила не все 13 свойств или 7 доменов)."""

from __future__ import annotations

from ..models import (
    ApiMatrixRow,
    AuditDefect,
    FintechDomainCheck,
    SetPropertyCheck,
)

# rule_id → домены финтех-профиля
RULE_DOMAIN = {
    "FIN-MONEY-FLOAT": "money_currency",
    "FIN-MONEY-UNITS": "money_currency",
    "FIN-TZ-OBSOLETE": "time_calendar",
    "FIN-TZ-IMPLICIT": "time_calendar",
    "FIN-IDEMPOTENCY": "idempotency",
    "FIN-RACE-ONCE": "concurrency",
    "API-SENSITIVE-QUERY": "security_pci_pii",
    "FIN-PAN-EXPOSED": "security_pci_pii",
    "API-GET-MUTATION": "api_errors",
    "API-200-WITH-ERROR": "api_errors",
    "API-STATUS-CONTRADICTION": "api_errors",
    "API-STATUS-UNDESCRIBED": "api_errors",
    "API-MISSING-CODES": "api_errors",
    "API-TYPE-MISMATCH": "api_errors",
    "STRUCT-NO-APPROVAL": "audit_compliance",
    "STRUCT-NO-VERSION": "audit_compliance",
    "STRUCT-CHECKLIST-UNFILLED": "audit_compliance",
}

CATEGORY_DOMAIN = {"security": "security_pci_pii", "api_contract": "api_errors"}

DOMAIN_OK_TEXT = {
    "money_currency": "Денежные поля целочисленные, единицы согласованы — нарушений правилами не выявлено.",
    "time_calendar": "Явных проблем с часовым поясом не выявлено.",
    "idempotency": "Механизмы идемпотентности описаны или не требуются.",
    "concurrency": "Ограничений, требующих атомарности, правилами не выявлено.",
    "security_pci_pii": "PAN/ПДн в URL и примерах не обнаружены.",
    "api_errors": "Контракт API не содержит выявленных правилами противоречий.",
    "audit_compliance": "Версионность и согласование оформлены.",
}

# свойство набора → категории дефектов, которые его нарушают
PROPERTY_CATEGORIES: dict[str, set[str]] = {
    "29148_complete": {"completeness", "placeholder"},
    "29148_consistent": {"consistency"},
    "29148_feasible": {"feasibility"},
    "29148_comprehensible": {"ambiguity", "singularity"},
    "29148_able_to_be_validated": {"verifiability", "conformity"},
    "830_correct": {"correctness", "fintech", "security"},
    "830_unambiguous": {"ambiguity"},
    "830_complete": {"completeness", "placeholder", "api_contract"},
    "830_consistent": {"consistency"},
    "830_ranked": set(),
    "830_verifiable": {"verifiability"},
    "830_modifiable": {"singularity", "conformity"},
    "830_traceable": {"traceability"},
}
PROPERTY_RULES = {"830_ranked": {"STRUCT-NOT-RANKED"}, "830_modifiable": {"STRUCT-NO-VERSION", "STRUCT-NON-ATOMIC"}}


def set_checks(defects: list[AuditDefect]) -> list[SetPropertyCheck]:
    out = []
    for prop, cats in PROPERTY_CATEGORIES.items():
        rules = PROPERTY_RULES.get(prop, set())
        hits = [d for d in defects if d.category in cats or d.rule_id in rules]
        severe = [d for d in hits if d.severity in ("Blocker", "Critical")]
        if severe:
            verdict = "fail"
        elif len(hits) >= 2 or any(d.severity == "Major" for d in hits):
            verdict = "partial" if not (len(hits) >= 4) else "fail"
        elif hits:
            verdict = "partial"
        else:
            verdict = "pass"
        if hits:
            refs = ", ".join(f"{d.defect_id} ({d.verified_location})" for d in hits[:5])
            more = f" и ещё {len(hits) - 5}" if len(hits) > 5 else ""
            comment = f"Нарушений: {len(hits)}, из них Blocker/Critical: {len(severe)}. См. {refs}{more}."
        else:
            comment = "Нарушений этого свойства не выявлено."
        out.append(SetPropertyCheck(prop=prop, verdict=verdict, comment=comment))  # type: ignore[arg-type]
    return out


def fintech_domains(defects: list[AuditDefect]) -> list[FintechDomainCheck]:
    by_domain: dict[str, list[AuditDefect]] = {}
    for d in defects:
        dom = RULE_DOMAIN.get(d.rule_id) or CATEGORY_DOMAIN.get(d.category)
        if dom:
            by_domain.setdefault(dom, []).append(d)
    out = []
    for dom in DOMAIN_OK_TEXT:
        hits = by_domain.get(dom, [])
        if not hits:
            out.append(FintechDomainCheck(domain=dom, status="ok", finding=DOMAIN_OK_TEXT[dom]))  # type: ignore[arg-type]
            continue
        status = "gap" if any(d.severity in ("Blocker", "Critical") for d in hits) else "risk"
        finding = "; ".join(f"{d.defect_id}: {d.reasoning.split('.')[0]}" for d in hits[:3])
        out.append(FintechDomainCheck(domain=dom, status=status, finding=finding))  # type: ignore[arg-type]
    return out


MATRIX_RULES = {
    "XCHK-READ-ENDPOINT-MISSING": ("Чтение настроек и суммы за месяц", "missing_in_api"),
    "API-GET-MUTATION": ("HTTP-метод операции сохранения", "mismatch"),
    "API-TYPE-MISMATCH": ("Типы полей запроса и ответа", "mismatch"),
    "API-200-WITH-ERROR": ("Ошибка ввода суммы: UI ↔ код ответа", "mismatch"),
    "API-STATUS-CONTRADICTION": ("Доступ к чужой карте", "mismatch"),
    "FIN-MONEY-UNITS": ("Единицы сумм (тиын / тенге)", "mismatch"),
    "API-MISSING-CODES": ("Коды ошибок 409/422/429", "missing_in_api"),
    "FIN-IDEMPOTENCY": ("Идемпотентность / повтор запроса", "missing_in_api"),
}


def api_matrix(defects: list[AuditDefect]) -> list[ApiMatrixRow]:
    rows = []
    seen = set()
    for d in defects:
        if d.rule_id in MATRIX_RULES and d.rule_id not in seen:
            seen.add(d.rule_id)
            aspect, status = MATRIX_RULES[d.rule_id]
            api_side = d.exact_quote if d.rule_id != "XCHK-READ-ENDPOINT-MISSING" else ""
            biz_side = d.exact_quote if d.rule_id == "XCHK-READ-ENDPOINT-MISSING" else ""
            rows.append(
                ApiMatrixRow(
                    aspect=aspect,
                    business_quote=biz_side,
                    api_quote=api_side,
                    status=status,  # type: ignore[arg-type]
                    comment=f"{d.defect_id}: {d.reasoning}",
                )
            )
    return rows
