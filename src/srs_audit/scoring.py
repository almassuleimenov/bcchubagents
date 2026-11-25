"""Индекс качества ТЗ (0–100) и жёсткий Go/No-Go гейт.

Индекс — нормированная экспоненциальная функция взвешенной нагрузки дефектов на одно
требование:

    load  = Σ w(severity) · n(severity) / N_requirements
    score = 100 · exp(−load / τ),   w = {Blocker: 10, Critical: 5, Major: 2, Minor: 0.5},  τ = 1.5

Свойства: 0 дефектов → 100; монотонно убывает; не уходит в минус; нормирован на
размер документа (длинный документ не штрафуется за длину). Гейт применяется ПОВЕРХ
индекса: один Blocker даёт NO-GO при любом score.
"""

from __future__ import annotations

import math
from collections import Counter

from .models import SEVERITY_ORDER, AuditDefect, QualityMetrics, Requirement

WEIGHTS = {"Blocker": 10.0, "Critical": 5.0, "Major": 2.0, "Minor": 0.5}
TAU = 1.5


def quality_score(by_severity: dict[str, int], n_requirements: int) -> tuple[float, float]:
    load = sum(WEIGHTS[s] * by_severity.get(s, 0) for s in WEIGHTS) / max(1, n_requirements)
    return round(100.0 * math.exp(-load / TAU), 1), round(load, 3)


def verdict(by_severity: dict[str, int], score: float) -> tuple[str, list[str]]:
    b, c, m = by_severity.get("Blocker", 0), by_severity.get("Critical", 0), by_severity.get("Major", 0)
    reasons: list[str] = []
    if b >= 1:
        reasons.append(f"Blocker-дефектов: {b} (порог NO-GO: ≥ 1)")
    if c >= 3:
        reasons.append(f"Critical-дефектов: {c} (порог NO-GO: ≥ 3)")
    if score < 60:
        reasons.append(f"Индекс качества {score} < 60")
    if reasons:
        return "NO-GO", reasons
    if 1 <= c <= 2:
        reasons.append(f"Critical-дефектов: {c} (1–2 → условный допуск)")
    if m >= 5:
        reasons.append(f"Major-дефектов: {m} (≥ 5 → условный допуск)")
    if 60 <= score < 80:
        reasons.append(f"Индекс качества {score} в диапазоне 60–79.9")
    if reasons:
        return "CONDITIONAL GO", reasons
    return "GO", [f"Blocker = 0, Critical = 0, Major = {m} < 5, индекс {score} ≥ 80"]


def compute_metrics(requirements: list[Requirement], defects: list[AuditDefect]) -> QualityMetrics:
    by_sev = Counter(d.severity for d in defects)
    by_sev_full = {s: by_sev.get(s, 0) for s in SEVERITY_ORDER}
    by_cat = dict(Counter(d.category for d in defects).most_common())
    n = len(requirements)
    score, load = quality_score(by_sev_full, n)
    v, reasons = verdict(by_sev_full, score)
    affected = {d.req_id for d in defects if d.req_id not in ("SET", "")}
    return QualityMetrics(
        n_requirements=n,
        n_defects=len(defects),
        by_severity=by_sev_full,
        by_category=by_cat,
        requirements_with_defects=len(affected),
        defect_density=round(len(defects) / max(1, n), 2),
        weighted_load=load,
        score=score,
        verdict=v,  # type: ignore[arg-type]
        verdict_reasons=reasons,
    )
