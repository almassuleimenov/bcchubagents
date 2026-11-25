"""Типизированные схемы данных конвейера.

Правило проекта: схемы, которые уходят в LLM как ``output_type``, — плоские
(без ``dict[str, Any]`` и сложных ``Union``), иначе Gemini ``response_schema`` /
function-declarations отвергает их. Поля, которые заполняет ТОЛЬКО код
(координаты, оценка заземления, статус), живут в отдельных «внутренних»
моделях и никогда не входят в LLM-схему — модель физически не может их подделать.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Словари
# ---------------------------------------------------------------------------

Severity = Literal["Blocker", "Critical", "Major", "Minor"]
SEVERITY_ORDER: dict[str, int] = {"Blocker": 0, "Critical": 1, "Major": 2, "Minor": 3}

Category = Literal[
    "ambiguity",  # неоднозначность (29148 §5.2.5 unambiguous, §5.2.7)
    "verifiability",  # непроверяемость (§5.2.5 verifiable)
    "completeness",  # неполнота (§5.2.5/§5.2.6 complete)
    "consistency",  # противоречие (§5.2.6 consistent, IEEE 830 §4.3.5)
    "singularity",  # неатомарность (§5.2.5 singular)
    "feasibility",  # нереализуемость
    "necessity",  # лишнее / не обосновано
    "implementation_free",  # лишнее проектное решение
    "correctness",  # фактическая/доменная ошибка
    "conformity",  # оформление, шаблон, версионность
    "traceability",  # трассируемость, ID, ссылки
    "security",  # ИБ / PCI DSS / ПДн
    "api_contract",  # контракт REST API
    "fintech",  # банковская предметная область РК
    "placeholder",  # TBD / заглушки / незаполненные поля
]

CATEGORY_RU: dict[str, str] = {
    "ambiguity": "Неоднозначность",
    "verifiability": "Непроверяемость",
    "completeness": "Неполнота",
    "consistency": "Противоречие",
    "singularity": "Неатомарность",
    "feasibility": "Реализуемость",
    "necessity": "Необходимость",
    "implementation_free": "Лишнее проектное решение",
    "correctness": "Корректность",
    "conformity": "Оформление / шаблон",
    "traceability": "Трассируемость",
    "security": "Информационная безопасность",
    "api_contract": "Контракт API",
    "fintech": "Финтех-домен",
    "placeholder": "Заглушка / не заполнено",
}

CRITICAL_CATEGORIES = {"verifiability", "consistency", "completeness"}

DefectSource = Literal["deterministic", "llm_quality", "llm_crosscheck"]
Scope = Literal["requirement", "set"]
Role = Literal["PO", "BA", "SA", "QA", "InfoSec"]
ReqKind = Literal[
    "business",
    "functional",
    "nonfunctional",
    "data",
    "integration",
    "ui",
    "process",
    "acceptance",
    "api",
    "constraint",
    "other",
]
GroundingStatus = Literal["exact", "fuzzy", "corrected", "quarantined", "unverified"]
Verdict = Literal["GO", "CONDITIONAL GO", "NO-GO", "UNPROCESSABLE"]
HealthStatus = Literal["OK", "DEGRADED", "UNPROCESSABLE"]


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


class HealthIssue(BaseModel):
    code: str
    level: Literal["info", "warning", "fatal"]
    message: str
    recommendation: str = ""


class DocumentHealthReport(BaseModel):
    path: str
    file_name: str
    size_bytes: int = 0
    sha256: str = ""
    declared_extension: str = ""
    detected_format: Literal["md", "txt", "pdf", "docx", "unknown"] = "unknown"
    encoding: str = ""
    status: HealthStatus = "OK"
    pages: int = 0
    text_chars: int = 0
    scanned_pages: list[int] = Field(default_factory=list)
    diagram_pages: list[int] = Field(default_factory=list)
    issues: list[HealthIssue] = Field(default_factory=list)

    def add(self, code: str, level: str, message: str, recommendation: str = "") -> None:
        self.issues.append(HealthIssue(code=code, level=level, message=message, recommendation=recommendation))  # type: ignore[arg-type]
        if level == "fatal":
            self.status = "UNPROCESSABLE"
        elif level == "warning" and self.status == "OK":
            self.status = "DEGRADED"


# ---------------------------------------------------------------------------
# Разобранный документ
# ---------------------------------------------------------------------------

BlockKind = Literal["heading", "paragraph", "list_item", "table_row", "table_header", "code", "other"]


class Block(BaseModel):
    """Минимальная единица исходника с координатами."""

    block_id: int
    kind: BlockKind
    text: str
    line_start: int
    line_end: int
    page: int | None = None
    section_path: str = ""
    lang: str = ""  # язык code fence (json/yaml/http/mermaid/curl/text)
    table_id: int | None = None
    cells: list[str] = Field(default_factory=list)
    provenance: str = ""  # md-ast / pymupdf / pdfplumber / docx / vision / txt


class ParsedDocument(BaseModel):
    source_path: str
    title: str
    format: str
    text: str  # «канонический» текст, по которому идёт заземление цитат
    blocks: list[Block]
    pages: int = 0

    @property
    def lines(self) -> list[str]:
        return self.text.splitlines()

    def line_pages(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for b in self.blocks:
            if b.page:
                for ln in range(b.line_start, b.line_end + 1):
                    out.setdefault(ln, b.page)
        return out


# ---------------------------------------------------------------------------
# Требования
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    """Атомарное требование. ID в отчёте назначает только код (стабильно, по порядку)."""

    req_id: str
    source_id: str = ""  # ID в документе (Ф-1) или пусто
    section_path: str = ""
    kind: ReqKind = "other"
    text: str  # дословный фрагмент исходника
    summary: str = ""
    line_start: int = 0
    line_end: int = 0
    page: int | None = None
    origin: Literal["deterministic", "llm"] = "deterministic"

    @property
    def location(self) -> str:
        loc = f"стр. {self.line_start}" if self.line_start == self.line_end else f"стр. {self.line_start}–{self.line_end}"
        if self.page:
            loc = f"с. {self.page}, {loc}"
        return loc


# --- LLM-схемы извлечения -------------------------------------------------


class LLMRequirement(BaseModel):
    source_id: str = Field("", description="ID требования в документе (например «Ф-1»), иначе пустая строка")
    section: str = Field(description="Раздел документа, где стоит требование")
    kind: ReqKind
    exact_quote: str = Field(description="ДОСЛОВНАЯ цитата требования из документа, без перефразирования")
    summary: str = Field(description="Краткая формулировка на русском, до 20 слов")


class APIParamFact(BaseModel):
    name: str
    location: Literal["path", "query", "header", "body", "response"]
    data_type: str
    constraints: str = ""


class APIEndpointFact(BaseModel):
    method: str = Field(description="HTTP-метод как указан в документе")
    path: str
    purpose: str
    params: list[APIParamFact] = Field(default_factory=list)
    status_codes: list[str] = Field(default_factory=list)
    exact_quote: str = Field(description="Дословная цитата из документа, где описан метод")


class DomainFact(BaseModel):
    term: str = Field(description="Понятие предметной области: порог, месяц, валюта, часовой пояс ...")
    value: str = Field(description="Что о нём сказано в документе")
    exact_quote: str


class ExtractionResult(BaseModel):
    requirements: list[LLMRequirement]
    api_endpoints: list[APIEndpointFact] = Field(default_factory=list)
    domain_facts: list[DomainFact] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Дефекты
# ---------------------------------------------------------------------------


class LLMDefect(BaseModel):
    """Evidence-first: сначала доказательство (цитата), потом рассуждение, потом вывод."""

    req_id: str = Field(description="ID требования из отчёта (REQ-xxx) или «SET» для дефекта набора")
    exact_quote: str = Field(description="ДОСЛОВНАЯ цитата из документа, на которой основан дефект (5–200 символов)")
    reasoning: str = Field(description="Почему это дефект: конкретно, со ссылкой на текст, на русском")
    standard_ref: str = Field(description="Пункт стандарта: ISO 29148 §5.2.5/§5.2.6/§5.2.7, IEEE 830 §4.3.x, GtWR Rx")
    category: Category
    severity: Severity
    recommendation: str = Field(description="Что именно исправить, на русском")


class AuditDefect(BaseModel):
    defect_id: str
    req_id: str
    exact_quote: str
    reasoning: str
    standard_ref: str
    category: Category
    severity: Severity
    source: DefectSource
    recommendation: str
    scope: Scope = "requirement"
    rule_id: str = ""
    # --- заполняет ТОЛЬКО код (grounding.py) ---
    verified_location: str = ""
    grounding_score: float = 0.0
    grounding_status: GroundingStatus = "unverified"
    line_start: int = 0
    line_end: int = 0
    page: int | None = None
    confirmed_by_llm: bool = False

    @property
    def category_ru(self) -> str:
        return CATEGORY_RU.get(self.category, self.category)


class CandidateVerdict(BaseModel):
    cand_id: str
    keep: bool
    reason: str = Field(description="Кратко: почему кандидат подтверждён или отклонён как ложное срабатывание")


class QualityBatchResult(BaseModel):
    candidate_verdicts: list[CandidateVerdict] = Field(default_factory=list)
    defects: list[LLMDefect] = Field(default_factory=list)


SetProperty = Literal[
    "29148_complete",
    "29148_consistent",
    "29148_feasible",
    "29148_comprehensible",
    "29148_able_to_be_validated",
    "830_correct",
    "830_unambiguous",
    "830_complete",
    "830_consistent",
    "830_ranked",
    "830_verifiable",
    "830_modifiable",
    "830_traceable",
]

SET_PROPERTY_RU: dict[str, str] = {
    "29148_complete": "ISO 29148 §5.2.6 — Полнота набора",
    "29148_consistent": "ISO 29148 §5.2.6 — Непротиворечивость",
    "29148_feasible": "ISO 29148 §5.2.6 — Реализуемость",
    "29148_comprehensible": "ISO 29148 §5.2.6 — Понятность",
    "29148_able_to_be_validated": "ISO 29148 §5.2.6 — Валидируемость",
    "830_correct": "IEEE 830 §4.3.1 — Корректность",
    "830_unambiguous": "IEEE 830 §4.3.2 — Однозначность",
    "830_complete": "IEEE 830 §4.3.3 — Полнота",
    "830_consistent": "IEEE 830 §4.3.4 — Согласованность",
    "830_ranked": "IEEE 830 §4.3.5 — Ранжирование по важности/стабильности",
    "830_verifiable": "IEEE 830 §4.3.6 — Проверяемость",
    "830_modifiable": "IEEE 830 §4.3.7 — Модифицируемость",
    "830_traceable": "IEEE 830 §4.3.8 — Трассируемость",
}


class SetPropertyCheck(BaseModel):
    prop: SetProperty
    verdict: Literal["pass", "partial", "fail"]
    comment: str


MatrixStatus = Literal["consistent", "mismatch", "missing_in_api", "missing_in_business", "ambiguous"]


class ApiMatrixRow(BaseModel):
    aspect: str = Field(description="Что сверяем: метод, тип суммы, коды ошибок, чтение настроек ...")
    business_quote: str = Field(description="Дословная цитата из части А (БТ/UI/BPMN/критерии) или пусто")
    api_quote: str = Field(description="Дословная цитата из части B (REST API) или пусто")
    status: MatrixStatus
    comment: str


FintechDomain = Literal[
    "money_currency",
    "time_calendar",
    "idempotency",
    "concurrency",
    "security_pci_pii",
    "api_errors",
    "audit_compliance",
]

FINTECH_DOMAIN_RU: dict[str, str] = {
    "money_currency": "1. Деньги и валюта (ISO 4217, тиыны, запрет float)",
    "time_calendar": "2. Время и календарь (UTC+5 с 01.03.2024, границы месяца)",
    "idempotency": "3. Идемпотентность и повторная доставка",
    "concurrency": "4. Конкурентность и race conditions",
    "security_pci_pii": "5. ИБ: PCI DSS 4.0.1, ПДн, ИИН/БИН, PAN",
    "api_errors": "6. Контракт API и обработка ошибок",
    "audit_compliance": "7. Аудит, версионность, откат, эксплуатация",
}


class FintechDomainCheck(BaseModel):
    domain: FintechDomain
    status: Literal["ok", "risk", "gap", "n_a"]
    finding: str


class CrossCheckResult(BaseModel):
    candidate_verdicts: list[CandidateVerdict] = Field(default_factory=list)
    defects: list[LLMDefect] = Field(default_factory=list)
    set_checks: list[SetPropertyCheck] = Field(default_factory=list)
    api_matrix: list[ApiMatrixRow] = Field(default_factory=list)
    fintech_domains: list[FintechDomainCheck] = Field(default_factory=list)


TraceVerdict = Literal["consistent", "contradiction", "not_covered", "ambiguous"]


class AcceptanceTrace(BaseModel):
    """Пошаговая трассировка одного критерия приёмки через правила документа."""

    criterion_quote: str = Field(description="ДОСЛОВНАЯ цитата строки критерия приёмки")
    applicable_rules: str = Field(description="ID и суть применимых требований (например «Ф-2: месяц по дате совершения»)")
    derivation: str = Field(description="Пошаговый расчёт ожидаемого результата ПО БУКВАЛЬНОМУ тексту требований")
    expected_by_rules: str = Field(description="Что должно получиться по требованиям")
    expected_by_criterion: str = Field(description="Что ожидает критерий приёмки")
    verdict: TraceVerdict


class ConsistencyResult(BaseModel):
    traces: list[AcceptanceTrace] = Field(default_factory=list)
    defects: list[LLMDefect] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Вопросы
# ---------------------------------------------------------------------------


class ClarifyingQuestion(BaseModel):
    target_role: Role
    question: str = Field(description="Конкретный вопрос-решение, на русском")
    related_defect_ids: list[str] = Field(description="ID дефектов (D-xxx), которые закрывает ответ")
    options: list[str] = Field(description="2–3 варианта решения")
    recommended_option: str = Field(description="Рекомендуемый вариант и почему (кратко)")
    decision_impact: str = Field(description="Что изменится в разработке/тестировании в зависимости от ответа")
    before_text: str = Field(description="«Было»: ДОСЛОВНАЯ цитата проблемного места")
    after_ears_proposal: str = Field(description="«Стало»: правка в синтаксисе EARS (Когда/Если/Пока/Где ... система должна ...)")


class QuestionSet(BaseModel):
    questions: list[ClarifyingQuestion]


class Question(ClarifyingQuestion):
    question_id: str = ""
    priority: int = 0
    before_grounded: bool = False
    before_location: str = ""


# ---------------------------------------------------------------------------
# Метрики, паспорт, итоговый отчёт
# ---------------------------------------------------------------------------


class CitationStats(BaseModel):
    checked: int = 0
    exact: int = 0
    fuzzy: int = 0
    corrected_via_retry: int = 0
    quarantined: int = 0
    retries_issued: int = 0

    @property
    def confirmed(self) -> int:
        return self.exact + self.fuzzy

    @property
    def hallucination_rate_in_report(self) -> float:
        return 0.0  # по построению: в отчёт попадают только заземлённые цитаты


class QuarantinedHypothesis(BaseModel):
    agent: str
    req_id: str = ""
    quote: str
    claim: str
    reason: str


class RejectedCandidate(BaseModel):
    rule_id: str
    req_id: str
    quote: str
    reason: str


class QualityMetrics(BaseModel):
    n_requirements: int = 0
    n_defects: int = 0
    by_severity: dict[str, int] = Field(default_factory=dict)
    by_category: dict[str, int] = Field(default_factory=dict)
    requirements_with_defects: int = 0
    defect_density: float = 0.0
    weighted_load: float = 0.0
    score: float = 100.0
    verdict: Verdict = "GO"
    verdict_reasons: list[str] = Field(default_factory=list)


class RunPassport(BaseModel):
    document_title: str
    document_path: str
    document_sha256: str = ""
    started_at: str
    finished_at: str = ""
    duration_s: float = 0.0
    model: str
    mode: Literal["llm", "offline"]
    profile: str
    agent_version: str
    tool_version: str
    llm_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    notes: list[str] = Field(default_factory=list)


class AuditReport(BaseModel):
    passport: RunPassport
    health: DocumentHealthReport
    requirements: list[Requirement] = Field(default_factory=list)
    defects: list[AuditDefect] = Field(default_factory=list)
    set_checks: list[SetPropertyCheck] = Field(default_factory=list)
    api_matrix: list[ApiMatrixRow] = Field(default_factory=list)
    fintech_domains: list[FintechDomainCheck] = Field(default_factory=list)
    api_endpoints: list[APIEndpointFact] = Field(default_factory=list)
    acceptance_traces: list[AcceptanceTrace] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    citation: CitationStats = Field(default_factory=CitationStats)
    quarantine: list[QuarantinedHypothesis] = Field(default_factory=list)
    rejected_candidates: list[RejectedCandidate] = Field(default_factory=list)
    manual_checklist: list[str] = Field(default_factory=list)
    priority_fixes: list[str] = Field(default_factory=list)
    source_text: str = Field(default="", exclude=True)  # канонический текст для HTML-подсветки
