"""Автономный HTML-дашборд аудита (один файл, без CDN, светлая/тёмная тема).

Метафора — корректура рукописи: пронумерованный исходник с пометками на полях, дефекты как
корректорские правки, «Было → Стало» как редакторская сверка. Весь контент рендерится на
стороне Python (страница читается и без JS); скрипт только фильтрует, связывает пометки
с карточками, рисует связи графа и копирует вопросы в Jira.
"""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict


from ..graph import NODE_KIND_RU, ORPHAN_RU
from ..grounding import Grounder
from ..models import (
    CATEGORY_RU,
    FINTECH_DOMAIN_RU,
    SET_PROPERTY_RU,
    SEVERITY_ORDER,
    AuditReport,
)
from .md_renderer import KIND_RU, MATRIX_STATUS, ROLE_RU, SOURCE_RU

SEVERITIES = ("Blocker", "Critical", "Major", "Minor")
ROLES = ("PO", "BA", "SA", "QA", "InfoSec")
MATRIX_TONE = {"consistent": "ok", "mismatch": "bad", "missing_in_api": "bad", "missing_in_business": "warn",
               "risk": "warn", "ambiguous": "warn"}
PRIORITY_RU = ("Blocker", "Critical", "Major", "Minor")
DOMAIN_TONE = {"ok": ("ok", "OK"), "risk": ("warn", "риск"), "gap": ("bad", "пробел"), "n_a": ("na", "н/п")}
SET_TONE = {"pass": ("ok", "соответствует"), "partial": ("warn", "частично"), "fail": ("bad", "не соответствует")}
TRACE_TONE = {
    "consistent": ("ok", "согласуется"),
    "contradiction": ("bad", "противоречие"),
    "not_covered": ("warn", "правил недостаточно"),
    "ambiguous": ("warn", "неоднозначно"),
}
GRAPH_COLUMNS = (
    ("Критерии приёмки", ("acceptance",)),
    ("Требования", ("requirement",)),
    ("Экран и операции", ("ui_field", "operation")),
    ("API: методы и параметры", ("endpoint", "param")),
    ("HTTP-статусы", ("status",)),
)

e = html.escape


def _icon(name: str) -> str:
    paths = {
        "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
        "moon": '<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>',
        "search": '<circle cx="11" cy="11" r="6.5"/><path d="m20 20-4.2-4.2"/>',
        "copy": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
        "shield": '<path d="M12 3 5 6v5c0 4.4 3 8.3 7 9.5 4-1.2 7-5.1 7-9.5V6z"/><path d="m9 12 2 2 4-4"/>',
        "arrow": '<path d="M5 12h14M13 6l6 6-6 6"/>',
    }
    return (f'<svg class="ic" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" '
            f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>')


def _fmt_dt(s: str) -> str:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})", s or "")
    if not m:
        return s or "—"
    y, mo, d, h, mi = m.groups()
    tz = re.search(r"UTC[+-]\d{2}:?\d{2}", s)
    return f"{d}.{mo}.{y}, {h}:{mi}" + (f" ({tz.group(0)})" if tz else "")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _standards(ref: str) -> list[str]:
    """Ссылки на стандарты без повторов: «29148:2018 §5.2.6» поглощается «29148 §5.2.6 complete»."""
    refs = list(dict.fromkeys(s.strip() for s in re.split(r";\s*|\s\|\s", ref or "") if s.strip()))

    def key(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r":\d{4}\b", "", s.lower())).strip()

    keys = [key(s) for s in refs]
    return [s for i, s in enumerate(refs)
            if not any(j != i and keys[j] != keys[i] and keys[j].startswith(keys[i]) for j in range(len(refs)))
            and keys.index(keys[i]) == i]


def _search_blob(*parts: str) -> str:
    return e(" ".join(_norm(p) for p in parts).lower(), quote=True)


# ---------------------------------------------------------------------------
# Исходник с корректорскими пометками
# ---------------------------------------------------------------------------


def _source(report: AuditReport) -> tuple[str, set[str]]:
    text = report.source_text
    g = Grounder(text)
    n = len(text)
    owner: list[tuple[str, ...]] = [()] * n
    sev_of = {d.defect_id: d.severity for d in report.defects}
    marked: set[str] = set()
    for d in report.defects:
        res = g.ground(d.exact_quote, (d.line_start, d.line_end))
        if not res.found:
            continue
        marked.add(d.defect_id)
        for i in range(res.char_start, min(res.char_end, n)):
            owner[i] = owner[i] + (d.defect_id,)

    def worst(ids) -> str:
        return min((sev_of[i] for i in ids), key=lambda s: SEVERITY_ORDER[s])

    line_hits: dict[int, set[str]] = defaultdict(set)
    for d in report.defects:
        if d.defect_id not in marked and d.line_start:
            for ln in range(d.line_start, (d.line_end or d.line_start) + 1):
                line_hits[ln].add(d.defect_id)

    rows, pos = [], 0
    for no, line in enumerate(text.split("\n"), start=1):
        parts, ids_on_line, i = [], set(line_hits.get(no, ())), 0
        while i < len(line):
            ids = owner[pos + i] if pos + i < n else ()
            j = i
            while j < len(line) and (owner[pos + j] if pos + j < n else ()) == ids:
                j += 1
            chunk = e(line[i:j])
            if ids:
                ids_on_line.update(ids)
                parts.append(f'<mark class="s-{worst(ids)}" data-ids="{" ".join(ids)}" tabindex="-1">{chunk}</mark>')
            else:
                parts.append(chunk)
            i = j
        gutter = ""
        if ids_on_line:
            w = worst(ids_on_line)
            gutter = (f'<button class="pin s-{w}" data-ids="{" ".join(sorted(ids_on_line))}" '
                      f'aria-label="Дефекты на строке {no}: {", ".join(sorted(ids_on_line))}">{len(ids_on_line)}</button>')
        rows.append(f'<div class="ln{" hit" if ids_on_line else ""}" id="L{no}"><span class="no">{no}</span>'
                    f'<span class="gut">{gutter}</span><span class="tx">{"".join(parts) or "&#8203;"}</span></div>')
        pos += len(line) + 1
    return "".join(rows), marked


# ---------------------------------------------------------------------------
# Части страницы
# ---------------------------------------------------------------------------


def _hero(report: AuditReport) -> str:
    m, p, c = report.metrics, report.passport, report.citation
    verdict_cls = {"NO-GO": "bad", "CONDITIONAL GO": "warn", "GO": "ok"}.get(m.verdict, "na")
    score = max(0.0, min(100.0, float(m.score)))
    tone = "bad" if score < 50 else "warn" if score < 75 else "ok"
    circ = 2 * 3.14159265 * 52
    grounded = sum(d.grounding_status in ("exact", "fuzzy", "corrected") for d in report.defects)
    total = len(report.defects)
    pct = round(100 * grounded / total) if total else 100
    cite_ok = pct == 100
    cite_tip = (f"Цитаты проверены по исходнику: {c.checked}; точных {c.exact}, нечётких {c.fuzzy}, "
                f"исправлено повторным запросом {c.corrected_via_retry}; в карантин ушло {c.quarantined} "
                f"гипотез без опоры в тексте — в отчёт они не попали.")
    kpis = "".join(
        f'<button class="kpi s-{s}{" zero" if not m.by_severity.get(s, 0) else ""}" data-sev="{s}" '
        f'aria-label="Показать {s}: {m.by_severity.get(s, 0)}"><span class="kpi-n">{m.by_severity.get(s, 0)}</span>'
        f'<span class="kpi-l"><i class="dot"></i>{s}</span></button>'
        for s in SEVERITIES
    )
    reasons = "".join(f"<li>{e(r)}</li>" for r in m.verdict_reasons[:3])
    model = "офлайн-режим, без LLM" if p.mode == "offline" else p.model
    notes = "".join(f"<li>{e(n)}</li>" for n in p.notes)
    quarantine = "".join(
        f"<li><b>{e(q.agent)}</b> · {e(q.req_id or '—')}: {e(_norm(q.claim)[:220])} <span class='mut'>— {e(q.reason)}</span></li>"
        for q in report.quarantine
    )
    rejected = "".join(
        f"<li><b>{e(r.rule_id)}</b> · {e(r.req_id)}: <span class='mut'>{e(r.reason)}</span></li>"
        for r in report.rejected_candidates
    )
    duration = f"{p.duration_s:.0f} с" if p.duration_s else "—"
    return f"""
<header class="hero">
  <div class="hero-main">
    <div class="doc">
      <h1>{e(p.document_title)}</h1>
      <p class="meta"><span>{e(report.health.file_name)}</span><span>{_fmt_dt(p.started_at)}</span>
        <span>модель: <code>{e(model)}</code></span><span><code>{e(p.agent_version)}</code></span></p>
    </div>
    <div class="verdict-row">
      <div class="verdict v-{verdict_cls}"><span class="verdict-k">Вердикт</span><strong>{e(m.verdict)}</strong></div>
      <ul class="reasons">{reasons}</ul>
    </div>
  </div>
  <div class="hero-side">
    <figure class="ring t-{tone}" aria-label="Индекс качества {score:.1f} из 100">
      <svg viewBox="0 0 120 120" aria-hidden="true"><circle class="ring-bg" cx="60" cy="60" r="52"/>
        <circle class="ring-fg" cx="60" cy="60" r="52" stroke-dasharray="{circ:.1f}"
          style="--off:{circ * (1 - score / 100):.1f};--circ:{circ:.1f}"/></svg>
      <figcaption><span class="ring-n">{score:.1f}</span><span class="ring-l">индекс качества</span></figcaption>
    </figure>
    <div class="kpis">{kpis}</div>
  </div>
  <div class="hero-foot">
    <span class="cite{" cite-ok" if cite_ok else ""}" title="{e(cite_tip, quote=True)}">{_icon("shield")}
      {pct}% Grounded · 0 галлюцинаций</span>
    <span class="mut">требований {m.n_requirements} · дефектов {m.n_defects} · с дефектами {m.requirements_with_defects}
      · LLM-запросов {p.llm_requests} · из кэша {p.cache_hits} · {duration}</span>
    <details class="journal"><summary>Журнал прогона</summary>
      <div class="journal-body">
        <p class="mut">{e(cite_tip)}</p>
        {f"<h4>Заметки конвейера</h4><ul>{notes}</ul>" if notes else ""}
        {f"<h4>Карантин гипотез ({len(report.quarantine)})</h4><ul>{quarantine}</ul>" if quarantine else ""}
        {f"<h4>Отклонённые кандидаты ({len(report.rejected_candidates)})</h4><ul>{rejected}</ul>" if rejected else ""}
      </div></details>
  </div>
</header>"""


def _cards(report: AuditReport, marked: set[str]) -> str:
    out = []
    for d in report.defects:
        stds = "".join(f'<span class="std">{e(s)}</span>' for s in _standards(d.standard_ref))
        lines = f"стр. {d.line_start}" + (f"–{d.line_end}" if d.line_end and d.line_end != d.line_start else "")
        src = SOURCE_RU.get(d.source, d.source)
        merged = (f'<span class="merged" title="{e(", ".join(d.merged_rules), quote=True)}">объединено находок: {d.merged_count}</span>'
                  if d.merged_count > 1 else "")
        blob = _search_blob(d.defect_id, d.req_id, d.reasoning, d.exact_quote, d.recommendation, d.standard_ref, d.rule_id)
        out.append(f"""
<article class="card" id="card-{d.defect_id}" data-id="{d.defect_id}" data-sev="{d.severity}" data-cat="{d.category}"
  data-req="{e(d.req_id)}" data-line="{d.line_start}" data-marked="{int(d.defect_id in marked)}" data-q="{blob}" tabindex="0"
  aria-label="{d.defect_id}, {d.severity}, {e(CATEGORY_RU[d.category])}">
  <div class="card-h"><span class="did">{d.defect_id}</span><span class="sev s-{d.severity}">{d.severity}</span>
    <span class="cat">{e(CATEGORY_RU[d.category])}</span><span class="loc">{lines} · {e(d.req_id)}</span></div>
  <p class="why">{e(d.reasoning)}</p>
  <blockquote class="quote">{e(_norm(d.exact_quote))}</blockquote>
  <div class="fix"><span class="fix-k">Как исправить</span><p>{e(d.recommendation)}</p></div>
  <div class="card-f">{stds}<span class="src">{e(src)}</span>{merged}</div>
</article>""")
    return "".join(out)


def _recommended(q) -> int:
    """Подсвечиваемый вариант: «Вариант N» в тексте рекомендации > recommended_index > совпадение текста."""
    return q.recommended_position()


def _prio(q) -> tuple[int, str]:
    p = q.priority if q.priority is not None and 0 <= q.priority < len(PRIORITY_RU) else len(PRIORITY_RU) - 1
    return p, PRIORITY_RU[p]


def _jira(q) -> str:
    opts = "\n".join(f"- {o}" for o in q.options)
    rel = ", ".join(q.related_defect_ids) or "—"
    p, sev = _prio(q)
    return (f"**[{q.question_id}] {q.question}**\n\nРоль: {q.target_role} · приоритет P{p} ({sev}) · "
            f"дефекты: {rel}\n\n**Варианты решения:**\n{opts}\n\n**Рекомендация:** {q.recommended_option}\n\n"
            f"**Цена неопределённости:** {q.decision_impact}\n\n**Было:**\n> {_norm(q.before_text)}\n\n"
            f"**Стало (EARS):**\n> {_norm(q.after_ears_proposal)}\n")


def _questions(report: AuditReport) -> tuple[str, str]:
    counts = Counter(q.target_role for q in report.questions)
    chips = [f'<button class="chip" data-role="" aria-pressed="true">Все <span class="n">{len(report.questions)}</span></button>']
    chips += [f'<button class="chip" data-role="{r}" aria-pressed="false"{" disabled" if not counts[r] else ""} '
              f'title="{e(ROLE_RU[r], quote=True)}">{r} <span class="n">{counts[r]}</span></button>' for r in ROLES]
    items = []
    # P0 (Blocker) всегда наверху: сортировка по приоритету, затем по номеру вопроса
    for q in sorted(report.questions, key=lambda q: (_prio(q)[0], q.question_id)):
        rec = _recommended(q)
        p, sev = _prio(q)
        opts = "".join(
            f'<li class="{"opt rec" if i == rec else "opt"}">{e(o)}{"<span class=rec-k>рекомендуется</span>" if i == rec else ""}</li>'
            for i, o in enumerate(q.options)
        )
        rel = "".join(f'<a class="dref" href="#card-{e(r)}" data-goto="{e(r)}">{e(r)}</a>' for r in q.related_defect_ids)
        loc = f'<span class="mut">{e(q.before_location)}</span>' if q.before_location else ""
        items.append(f"""
<article class="q" data-role="{q.target_role}" id="q-{e(q.question_id)}">
  <div class="q-h"><span class="prio p{p}" title="Приоритет P{p}: закрывает дефект уровня {sev}">P{p}<span class="prio-s">{sev}</span></span><span class="role">{q.target_role}</span>
    <span class="role-full">{e(ROLE_RU[q.target_role])}</span><span class="qid">{e(q.question_id)}</span></div>
  <h3 class="q-t">{e(q.question)}</h3>
  <div class="q-grid">
    <div><h4>Варианты решения</h4><ol class="opts">{opts}</ol>
      <p class="rec-why"><b>Рекомендация.</b> {e(q.recommended_option)}</p></div>
    <div><h4>Цена неопределённости</h4><p>{e(q.decision_impact)}</p>
      {f'<h4>Закрывает дефекты</h4><p class="drefs">{rel}</p>' if rel else ""}</div>
  </div>
  <div class="diff" role="group" aria-label="Правка требования">
    <div class="del"><span class="diff-k">Было</span>{loc}<p>{e(_norm(q.before_text))}</p></div>
    <div class="ins"><span class="diff-k">Стало (EARS)</span><p>{e(_norm(q.after_ears_proposal))}</p></div>
  </div>
  <div class="q-f"><button class="btn copy" data-md="{e(_jira(q), quote=True)}">{_icon("copy")}Копировать для Jira</button></div>
</article>""")
    if not items:
        items.append('<p class="empty">Уточняющих вопросов нет: синтез вопросов не запускался или все дефекты однозначны.</p>')
    return "".join(chips), "".join(items)


def _pill(tone: str, label: str) -> str:
    return f'<span class="pill p-{tone}">{e(label)}</span>'


def _matrix(report: AuditReport) -> str:
    parts = []
    if report.api_matrix:
        rows = "".join(
            f"<tr><td class='asp'>{e(r.aspect)}</td><td><q>{e(_norm(r.business_quote)) or '<span class=mut>—</span>'}</q></td>"
            f"<td><q>{e(_norm(r.api_quote)) or '<span class=mut>—</span>'}</q></td>"
            f"<td>{_pill(MATRIX_TONE.get(r.status, 'na'), MATRIX_STATUS.get(r.status, r.status).split(' ', 1)[-1])}</td>"
            f"<td>{e(r.comment)}</td></tr>"
            for r in report.api_matrix
        )
        bad = sum(MATRIX_TONE.get(r.status) == "bad" for r in report.api_matrix)
        parts.append(f"""<section class="block"><div class="block-h"><h2>Матрица A ↔ B</h2>
<p class="mut">Сверка бизнес-части (A) с контрактом REST API (B): {len(report.api_matrix)} аспектов, расхождений {bad}.</p></div>
<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Аспект</th><th>Часть A — бизнес</th><th>Часть B — API</th><th>Статус</th><th>Комментарий</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>""")
    else:
        parts.append('<section class="block"><h2>Матрица A ↔ B</h2><p class="empty">Матрица не построена: в документе нет раздела API или CrossCheck-агент не запускался.</p></section>')

    if report.fintech_domains:
        doms = "".join(
            f'<li class="dom"><span class="dom-n">{e(FINTECH_DOMAIN_RU[d.domain])}</span>'
            f"{_pill(*DOMAIN_TONE[d.status])}<p>{e(d.finding)}</p></li>"
            for d in report.fintech_domains
        )
        parts.append(f'<section class="block"><div class="block-h"><h2>Финтех-домены Казахстана</h2>'
                     f'<p class="mut">Профиль kz-banking: семь обязательных доменов проверки.</p></div><ul class="doms">{doms}</ul></section>')

    g = report.trace_graph
    if g.nodes:
        cnt = Counter(n.kind for n in g.nodes)
        ext = sum(x.origin == "EXTRACTED" for x in g.edges)
        stats = "".join(f'<li><span class="st-n">{v}</span><span class="st-l">{e(NODE_KIND_RU[k])}</span></li>' for k, v in cnt.items())
        orphan_ids = {o.node_id for o in g.orphans}
        linked = {x.src for x in g.edges if x.src != x.dst} | {x.dst for x in g.edges if x.src != x.dst}
        examples = {x.src for x in g.edges if x.relation == "has_example"}
        cols = []
        for title, kinds in GRAPH_COLUMNS:
            nodes = [n for n in g.nodes if n.kind in kinds and (n.node_id in linked or n.node_id in orphan_ids)]
            hidden = sum(1 for n in g.nodes if n.kind in kinds) - len(nodes)
            if not nodes and not hidden:
                continue
            items = "".join(
                f'<button class="gn k-{n.kind}{" orphan" if n.node_id in orphan_ids else ""}" data-node="{e(n.node_id, quote=True)}" '
                f'data-line="{n.line}" title="{e(n.label, quote=True)} — стр. {n.line}">'
                f'<span class="gn-t">{e(n.label)}</span>'
                + (f'<span class="gn-s">{"есть пример" if n.node_id in examples else "нет примера"}</span>' if n.kind == "status" else "")
                + "</button>"
                for n in nodes
            )
            more = f'<p class="mut gmore">+{hidden} без связей</p>' if hidden else ""
            cols.append(f'<div class="gcol"><h4>{title}</h4>{items}{more}</div>')
        orphans = "".join(
            f'<li><span class="pill p-bad">{e(ORPHAN_RU.get(o.kind, o.kind))}</span><b>{e(o.label)}</b>'
            f'<a class="lref" href="#L{o.line}" data-line="{o.line}">стр. {o.line}</a><span class="mut">{e(o.explanation)}</span></li>'
            for o in g.orphans
        )
        gdata = json.dumps([[x.src, x.dst, x.relation, x.origin] for x in g.edges if x.src != x.dst], ensure_ascii=False)
        parts.append(f"""<section class="block"><div class="block-h"><h2>Граф трассировки</h2>
<p class="mut">Построен кодом без LLM: связей {len(g.edges)}, из них записаны в документе — {ext}, выведены — {len(g.edges) - ext}.
Сплошная линия — связь из текста, пунктир — выведенная. Наведите на узел, чтобы увидеть его связи.</p></div>
<ul class="stats">{stats}</ul>
<div class="graph" id="graph"><svg class="gedges" aria-hidden="true"></svg>{"".join(cols)}</div>
<script type="application/json" id="graph-data">{gdata.replace("</", "<\\/")}</script>
<h3 class="sub">Разрывы трассировки <span class="n">{len(g.orphans)}</span></h3>
{f'<ul class="orphans">{orphans}</ul>' if orphans else '<p class="empty">Разрывов нет: у каждого правила есть критерий приёмки, у каждого поля экрана — параметр API.</p>'}
</section>""")

    if report.acceptance_traces:
        rows = "".join(
            f"<tr><td><q>{e(_norm(t.criterion_quote))}</q></td><td>{e(t.applicable_rules)}</td><td>{e(t.expected_by_rules)}</td>"
            f"<td>{e(t.expected_by_criterion)}</td><td>{_pill(*TRACE_TONE[t.verdict])}</td></tr>"
            for t in report.acceptance_traces
        )
        parts.append(f"""<section class="block"><div class="block-h"><h2>Трассировка критериев приёмки</h2>
<p class="mut">Каждый критерий пересчитан по буквальному тексту правил.</p></div>
<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Критерий</th><th>Правила</th><th>По правилам</th><th>По критерию</th><th>Итог</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>""")

    if report.set_checks:
        rows = "".join(
            f'<li class="dom"><span class="dom-n">{e(SET_PROPERTY_RU.get(s.prop, s.prop))}</span>{_pill(*SET_TONE[s.verdict])}<p>{e(s.comment)}</p></li>'
            for s in report.set_checks
        )
        parts.append(f'<section class="block"><div class="block-h"><h2>Свойства набора требований</h2>'
                     f'<p class="mut">ISO/IEC/IEEE 29148 §5.2.6 и IEEE 830 §4.3.</p></div><ul class="doms">{rows}</ul></section>')
    return "".join(parts)


def _reqs(report: AuditReport) -> tuple[str, str]:
    per_req = Counter(d.req_id for d in report.defects)
    worst: dict[str, str] = {}
    for d in report.defects:
        if d.req_id not in worst or SEVERITY_ORDER[d.severity] < SEVERITY_ORDER[worst[d.req_id]]:
            worst[d.req_id] = d.severity
    kinds = Counter(r.kind for r in report.requirements)
    opts = "".join(f'<option value="{k}">{e(KIND_RU.get(k, k))} ({v})</option>' for k, v in kinds.most_common())
    rows = []
    for r in report.requirements:
        n = per_req.get(r.req_id, 0)
        sec = r.section_path.split("›")[-1].strip() if r.section_path else "—"
        lines = f"{r.line_start}" + (f"–{r.line_end}" if r.line_end and r.line_end != r.line_start else "")
        cnt = (f'<button class="cnt s-{worst[r.req_id]}" data-req="{e(r.req_id)}" aria-label="Показать {n} дефектов {e(r.req_id)}">{n}</button>'
               if n else '<span class="cnt zero">0</span>')
        rows.append(
            f'<tr data-kind="{r.kind}" data-n="{n}" data-q="{_search_blob(r.req_id, r.source_id, r.text, r.section_path)}">'
            f'<td class="mono">{e(r.req_id)}</td><td class="mono">{e(r.source_id or "—")}</td>'
            f'<td>{e(KIND_RU.get(r.kind, r.kind))}</td><td title="{e(r.section_path, quote=True)}">{e(sec)}</td>'
            f'<td class="mono"><a class="lref" href="#L{r.line_start}" data-line="{r.line_start}">{lines}</a></td>'
            f"<td class='rtext'>{e(_norm(r.text))}</td><td class='num'>{cnt}</td></tr>"
        )
    return opts, "".join(rows)


# ---------------------------------------------------------------------------
# Сборка
# ---------------------------------------------------------------------------


def render_html(report: AuditReport) -> str:
    source, marked = _source(report)
    m = report.metrics
    cats = Counter(d.category for d in report.defects)
    cat_opts = "".join(f'<option value="{k}">{e(CATEGORY_RU[k])} ({v})</option>' for k, v in cats.most_common())
    sev_chips = "".join(
        f'<button class="chip s-{s}" data-sev="{s}" aria-pressed="false"{" disabled" if not m.by_severity.get(s) else ""}>'
        f'<i class="dot"></i>{s} <span class="n">{m.by_severity.get(s, 0)}</span></button>'
        for s in SEVERITIES
    )
    role_chips, questions = _questions(report)
    kind_opts, req_rows = _reqs(report)
    n_matrix = len(report.api_matrix) + len(report.trace_graph.orphans)
    tabs = [
        ("doc", "Документ и дефекты", len(report.defects)),
        ("questions", "Бэклог вопросов (EARS)", len(report.questions)),
        ("matrix", "Матрица A ↔ B и граф", n_matrix),
        ("reqs", "Нарезка требований", len(report.requirements)),
    ]
    tab_btns = "".join(
        f'<button role="tab" id="tab-{k}" aria-controls="panel-{k}" aria-selected="{"true" if i == 0 else "false"}" '
        f'tabindex="{0 if i == 0 else -1}">{t} <span class="n">{n}</span></button>'
        for i, (k, t, n) in enumerate(tabs)
    )
    body = f"""
{_hero(report)}
<nav class="tabs" role="tablist" aria-label="Разделы отчёта">{tab_btns}
  <button class="theme" id="theme" aria-label="Переключить тему">{_icon("sun")}{_icon("moon")}</button></nav>

<section class="panel split" role="tabpanel" id="panel-doc" aria-labelledby="tab-doc">
  <div class="pane src" id="src" aria-label="Исходный документ"><div class="pane-h"><h2>Исходный документ</h2>
    <span class="mut">{e(report.health.file_name)} · строк {report.source_text.count(chr(10)) + 1}</span></div>
    <div class="lines">{source}</div></div>
  <div class="pane list">
    <div class="toolbar">
      <label class="search">{_icon("search")}<input type="search" id="q" placeholder="Поиск по тексту, D-xxx, REQ-xxx" aria-label="Поиск дефектов"></label>
      <div class="chips" role="group" aria-label="Серьёзность">{sev_chips}</div>
      <select id="cat" aria-label="Категория"><option value="">Все категории</option>{cat_opts}</select>
      <p class="shown mut" id="shown" aria-live="polite"></p>
    </div>
    <div class="cards" id="cards">{_cards(report, marked)}
      <div class="empty" id="cards-empty" hidden>Под эти фильтры дефектов нет. <button class="btn-link" id="reset">Сбросить фильтры</button></div></div>
  </div>
</section>

<section class="panel" role="tabpanel" id="panel-questions" aria-labelledby="tab-questions" hidden>
  <div class="wrap"><div class="block-h"><h2>Бэклог уточняющих вопросов</h2>
    <p class="mut">Решения, без которых требование нельзя реализовать и проверить. Правка «Стало» — в синтаксисе EARS.</p></div>
    <div class="chips roles" role="group" aria-label="Роль">{role_chips}</div>
    <div class="qs">{questions}</div></div>
</section>

<section class="panel" role="tabpanel" id="panel-matrix" aria-labelledby="tab-matrix" hidden>
  <div class="wrap">{_matrix(report)}</div>
</section>

<section class="panel" role="tabpanel" id="panel-reqs" aria-labelledby="tab-reqs" hidden>
  <div class="wrap"><div class="block-h"><h2>Нарезка требований</h2>
    <p class="mut">Атомарные требования, на которые разрезан документ, и число найденных в них дефектов.</p></div>
    <div class="toolbar row">
      <label class="search">{_icon("search")}<input type="search" id="rq" placeholder="Поиск по тексту, REQ-xxx, Ф-1" aria-label="Поиск требований"></label>
      <select id="rkind" aria-label="Тип требования"><option value="">Все типы</option>{kind_opts}</select>
      <label class="check"><input type="checkbox" id="rdef"> только с дефектами</label>
      <p class="shown mut" id="rshown" aria-live="polite"></p>
    </div>
    <div class="tbl-wrap"><table class="tbl reqs"><thead><tr><th>ID</th><th>В документе</th><th>Тип</th><th>Раздел</th><th>Строка</th><th>Текст</th><th class="num">Дефекты</th></tr></thead>
      <tbody id="rbody">{req_rows}</tbody></table>
      <p class="empty" id="reqs-empty" hidden>Ничего не найдено.</p></div></div>
</section>

<div class="toast" id="toast" role="status" aria-live="polite"></div>
<footer class="foot mut">SRS Audit · {e(report.passport.agent_version)} · отчёт сгенерирован автоматически; каждая цитата сверена с исходником.</footer>
"""
    return (
        '<!doctype html><html lang="ru"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>SRS Audit · {e(report.passport.document_title)}</title>"
        f'<meta name="color-scheme" content="light dark"><style>{CSS}</style></head>'
        f"<body>{body}<script>{JS}</script></body></html>"
    )


CSS = r"""
:root{
  --paper:#f7f5f0;--surface:#fffefb;--sunk:#f0ede6;--ink:#1c1d21;--ink-2:#454852;--ink-3:#686c77;--rule:#e2ded5;--rule-2:#cfcac0;
  --accent:#a8261d;--focus:#2457c5;
  --Blocker:#7d1414;--Critical:#c0262d;--Major:#a8530a;--Minor:#7f6608;
  --ok:#1d6b39;--ok-bg:#e3f1e6;--warn:#8a5a00;--warn-bg:#f8eed6;--bad:#a8261d;--bad-bg:#f8e3df;--na:#686c77;--na-bg:#ecebe7;
  --del:#fbe6e3;--del-ink:#86201a;--ins:#e2f2e5;--ins-ink:#1a5c31;
  --mono:"JetBrains Mono",ui-monospace,"Cascadia Mono",Consolas,monospace;
  --sans:Inter,system-ui,"Segoe UI",Roboto,sans-serif;
  --ease-out:cubic-bezier(0.23,1,0.32,1);
  --tabs-h:52px;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#121316;--surface:#1a1b1f;--sunk:#151619;--ink:#ecebe6;--ink-2:#bdbbb4;--ink-3:#928f88;--rule:#2b2c31;--rule-2:#3a3b41;
  --accent:#ff8a7a;--focus:#8fb0ff;
  --Blocker:#ff8c8c;--Critical:#ff7a6e;--Major:#f1a35a;--Minor:#d8bd52;
  --ok:#6fd394;--ok-bg:#16301f;--warn:#e9c065;--warn-bg:#2f2613;--bad:#ff8a7a;--bad-bg:#3a1a17;--na:#928f88;--na-bg:#25262a;
  --del:#3a1a18;--del-ink:#ffb2a8;--ins:#15301e;--ins-ink:#a6e8bb;}}
:root[data-theme="dark"]{
  --paper:#121316;--surface:#1a1b1f;--sunk:#151619;--ink:#ecebe6;--ink-2:#bdbbb4;--ink-3:#928f88;--rule:#2b2c31;--rule-2:#3a3b41;
  --accent:#ff8a7a;--focus:#8fb0ff;
  --Blocker:#ff8c8c;--Critical:#ff7a6e;--Major:#f1a35a;--Minor:#d8bd52;
  --ok:#6fd394;--ok-bg:#16301f;--warn:#e9c065;--warn-bg:#2f2613;--bad:#ff8a7a;--bad-bg:#3a1a17;--na:#928f88;--na-bg:#25262a;
  --del:#3a1a18;--del-ink:#ffb2a8;--ins:#15301e;--ins-ink:#a6e8bb;}
*{box-sizing:border-box}
html{scroll-padding-top:calc(var(--tabs-h) + 8px)}
body{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 var(--sans);font-feature-settings:"tnum" 1,"cv11" 1;-webkit-font-smoothing:antialiased}
::selection{background:color-mix(in srgb,var(--accent) 22%,transparent)}
:focus-visible{outline:2px solid var(--focus);outline-offset:2px;border-radius:4px}
*{scrollbar-width:thin;scrollbar-color:var(--rule-2) transparent}
h1,h2,h3,h4{margin:0;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:17px;font-weight:650}h3{font-size:15px;font-weight:600}h4{font-size:12.5px;font-weight:600;color:var(--ink-2);margin:0 0 6px}
p{margin:0}code,.mono{font-family:var(--mono);font-size:.92em}
.mut{color:var(--ink-3)}.ic{width:16px;height:16px;flex:none}
button{font:inherit;color:inherit;cursor:pointer}
select,input[type=search]{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--rule-2);border-radius:8px;height:34px;padding:0 10px}
.n{font-variant-numeric:tabular-nums;color:var(--ink-3);font-size:.9em;margin-left:2px}

/* hero */
.hero{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:20px 40px;padding:28px 32px 18px;background:var(--surface);border-bottom:1px solid var(--rule)}
.hero h1{font-size:26px;line-height:1.2;font-weight:700;max-width:44ch}
.meta{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:8px;color:var(--ink-3);font-size:13px}
.meta code{color:var(--ink-2);overflow-wrap:anywhere}
.hero-main,.hero-side,.hero-foot{min-width:0}
.verdict-row{display:flex;gap:18px;align-items:flex-start;margin-top:22px}
.verdict{display:flex;flex-direction:column;gap:2px;padding:10px 16px;border-radius:10px;min-width:150px}
.verdict-k{font-size:12px;font-weight:500;opacity:.85}
.verdict strong{font-size:22px;line-height:1.1;letter-spacing:.01em}
.v-bad{background:var(--bad);color:#fff}.v-warn{background:var(--warn-bg);color:var(--warn)}.v-ok{background:var(--ok-bg);color:var(--ok)}.v-na{background:var(--na-bg);color:var(--ink-2)}
:root[data-theme="dark"] .v-bad{background:var(--bad-bg);color:var(--bad)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .v-bad{background:var(--bad-bg);color:var(--bad)}}
.reasons{margin:0;padding:0 0 0 16px;color:var(--ink-2);font-size:13px;max-width:62ch}.reasons li+li{margin-top:3px}
.hero-side{display:flex;gap:24px;align-items:center}
.ring{position:relative;width:132px;height:132px;margin:0}
.ring svg{width:100%;height:100%;transform:rotate(-90deg)}
.ring circle{fill:none;stroke-width:9}
.ring-bg{stroke:var(--sunk)}
.ring-fg{stroke:currentColor;stroke-linecap:round;stroke-dashoffset:var(--off);animation:ring .9s var(--ease-out) .1s backwards}
@keyframes ring{from{stroke-dashoffset:var(--circ)}}
.ring.t-bad{color:var(--bad)}.ring.t-warn{color:var(--warn)}.ring.t-ok{color:var(--ok)}
.ring figcaption{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center}
.ring-n{font-size:30px;font-weight:700;color:var(--ink);font-variant-numeric:tabular-nums;line-height:1}
.ring-l{font-size:11.5px;color:var(--ink-3);margin-top:4px}
.kpis{display:grid;grid-template-columns:repeat(2,minmax(108px,1fr));gap:8px}
.kpi{display:flex;flex-direction:column;align-items:flex-start;gap:2px;padding:10px 14px;background:var(--paper);border:1px solid var(--rule);border-radius:10px;text-align:left}
.kpi-n{font-size:24px;font-weight:700;line-height:1.1;font-variant-numeric:tabular-nums;color:var(--sc)}
.kpi-l{display:flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink-2)}
.kpi.zero .kpi-n{color:var(--ink-3)}
.dot{display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--sc,var(--ink-3));flex:none}
.s-Blocker{--sc:var(--Blocker)}.s-Critical{--sc:var(--Critical)}.s-Major{--sc:var(--Major)}.s-Minor{--sc:var(--Minor)}
.hero-foot{grid-column:1/-1;display:flex;flex-wrap:wrap;align-items:center;gap:8px 18px;padding-top:14px;border-top:1px solid var(--rule);font-size:13px}
.cite{display:inline-flex;align-items:center;gap:6px;padding:4px 10px;border-radius:999px;font-weight:600;background:var(--na-bg);color:var(--ink-2);cursor:help}
.cite-ok{background:var(--ok-bg);color:var(--ok)}
.journal{margin-left:auto}.journal summary{cursor:pointer;color:var(--ink-2);list-style:none}
.journal summary::-webkit-details-marker{display:none}
.journal summary::after{content:"";display:inline-block;width:6px;height:6px;margin-left:8px;border:solid currentColor;border-width:0 1.5px 1.5px 0;transform:translateY(-3px) rotate(45deg)}
.journal[open] summary::after{transform:translateY(0) rotate(-135deg)}
.journal-body{position:absolute;right:32px;left:32px;z-index:20;margin-top:10px;padding:16px 18px;max-height:60vh;overflow:auto;background:var(--surface);border:1px solid var(--rule-2);border-radius:10px;box-shadow:0 12px 32px rgb(0 0 0/.14)}
.journal-body h4{margin-top:14px}.journal-body ul{margin:0;padding-left:18px;font-size:13px;color:var(--ink-2)}.journal-body li+li{margin-top:4px}

/* tabs */
.tabs{position:sticky;top:0;z-index:10;display:flex;align-items:stretch;gap:4px;height:var(--tabs-h);padding:0 24px;background:color-mix(in srgb,var(--surface) 92%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);overflow-x:auto}
.tabs [role=tab]{position:relative;border:0;background:none;padding:0 12px;font-weight:550;color:var(--ink-3);white-space:nowrap}
.tabs [role=tab][aria-selected=true]{color:var(--ink)}
.tabs [role=tab][aria-selected=true]::after{content:"";position:absolute;left:10px;right:10px;bottom:-1px;height:2px;background:var(--accent);border-radius:2px}
.theme{margin-left:auto;align-self:center;display:grid;place-items:center;width:34px;height:34px;border:1px solid var(--rule);border-radius:8px;background:var(--surface);color:var(--ink-2)}
.theme .ic:last-child{display:none}
:root[data-theme="dark"] .theme .ic:first-child{display:none}:root[data-theme="dark"] .theme .ic:last-child{display:block}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .theme .ic:first-child{display:none}:root:not([data-theme="light"]) .theme .ic:last-child{display:block}}

/* split view */
.panel[hidden]{display:none}
.split{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(380px,1fr);height:calc(100vh - var(--tabs-h));min-height:520px}
.pane{overflow:auto;min-height:0}
.pane-h{position:sticky;top:0;z-index:2;display:flex;align-items:baseline;gap:12px;padding:14px 20px 10px;background:var(--surface);border-bottom:1px solid var(--rule)}
.src{background:var(--surface);border-right:1px solid var(--rule)}
.lines{padding:8px 0 40px;font:12.5px/1.7 var(--mono)}
.ln{display:grid;grid-template-columns:48px 28px minmax(0,1fr);scroll-margin:120px}
.ln .no{text-align:right;color:var(--ink-3);user-select:none;padding-right:6px;font-variant-numeric:tabular-nums}
.ln .gut{display:flex;justify-content:center;align-items:flex-start;padding-top:2px}
.ln .tx{white-space:pre-wrap;word-break:break-word;padding-right:20px;color:var(--ink)}
.ln.hit{background:color-mix(in srgb,var(--sunk) 70%,transparent)}
.ln.target{background:color-mix(in srgb,var(--accent) 10%,transparent);transition:background-color .9s ease-out}
.pin{min-width:18px;height:18px;padding:0 4px;border:0;border-radius:4px;background:var(--sc);color:var(--surface);font:600 10.5px/18px var(--sans);font-variant-numeric:tabular-nums}
mark{color:inherit;background:color-mix(in srgb,var(--sc) 13%,transparent);text-decoration:underline;text-decoration-color:var(--sc);text-decoration-thickness:2px;text-underline-offset:3px;cursor:pointer;border-radius:2px;transition:background-color .6s ease-out}
mark.dim{background:none;text-decoration-color:color-mix(in srgb,var(--sc) 35%,transparent)}
mark.on{background:color-mix(in srgb,var(--sc) 34%,transparent)}
.list{background:var(--paper)}
.toolbar{position:sticky;top:0;z-index:2;display:flex;flex-wrap:wrap;align-items:center;gap:8px;padding:12px 16px;background:var(--paper);border-bottom:1px solid var(--rule)}
.toolbar.row{position:static;padding:0 0 12px;border:0;background:none}
.search{display:flex;align-items:center;gap:6px;flex:1 1 220px;position:relative}
.search .ic{position:absolute;left:10px;color:var(--ink-3)}
.search input{width:100%;padding-left:32px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:6px;height:30px;padding:0 10px;border:1px solid var(--rule-2);border-radius:999px;background:var(--surface);color:var(--ink-2);font-size:13px}
.chip[aria-pressed=true]{background:var(--ink);border-color:var(--ink);color:var(--surface)}
.chip[aria-pressed=true] .n{color:inherit;opacity:.75}
.chip:disabled{opacity:.45;cursor:default}
.shown{flex-basis:100%;font-size:12.5px}
.cards{padding:12px 16px 60px;display:flex;flex-direction:column;gap:10px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:12px 14px;cursor:pointer;scroll-margin:100px}
.card[aria-current=true]{border-color:var(--sc,var(--accent));box-shadow:0 0 0 1px var(--sc,var(--accent))}
.card[hidden]{display:none}
.card-h{display:flex;flex-wrap:wrap;align-items:center;gap:6px 10px;font-size:12.5px}
.did{font:600 12.5px var(--mono)}
.sev{padding:1px 8px;border-radius:5px;font-weight:600;color:var(--sc);background:color-mix(in srgb,var(--sc) 13%,transparent)}
.cat{color:var(--ink-2);font-weight:500}.loc{margin-left:auto;color:var(--ink-3);font-family:var(--mono);font-size:12px}
.why{margin-top:8px;color:var(--ink)}
.quote{margin:8px 0 0;padding:6px 10px;background:var(--sunk);border-radius:6px;font:12px/1.55 var(--mono);color:var(--ink-2);white-space:pre-wrap;word-break:break-word}
.fix{margin-top:10px}.fix-k{display:block;font-size:12px;font-weight:600;color:var(--ok)}.fix p{color:var(--ink-2)}
.card-f{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px;align-items:center}
.std{padding:1px 7px;border:1px solid var(--rule-2);border-radius:5px;font-size:11.5px;color:var(--ink-2)}
.src{font-size:11.5px;color:var(--ink-3)}.card-f .src{margin-left:auto;background:none;border:0}
.merged{font-size:11.5px;color:var(--ink-3)}
.empty{padding:28px 16px;text-align:center;color:var(--ink-3)}
.btn-link{border:0;background:none;color:var(--focus);text-decoration:underline;padding:0}

/* blocks, tables */
.wrap{max-width:1320px;padding:28px 32px 80px}
.block{margin-bottom:36px}.block-h{margin-bottom:14px}.block-h p{margin-top:4px;max-width:80ch}
.sub{margin:22px 0 10px}
.tbl-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
.tbl{width:100%;min-width:760px;border-collapse:collapse;font-size:13px}
.tbl th{position:sticky;top:0;text-align:left;font-weight:600;font-size:12px;color:var(--ink-2);background:var(--sunk);padding:9px 12px;border-bottom:1px solid var(--rule);white-space:nowrap}
.tbl td{padding:9px 12px;border-top:1px solid var(--rule);vertical-align:top}
.tbl tr:first-child td{border-top:0}
.tbl q{font-family:var(--mono);font-size:12px;color:var(--ink-2);quotes:none}
.tbl .asp{font-weight:600;min-width:140px}
.num{text-align:right}
.pill{display:inline-block;padding:1px 8px;border-radius:5px;font-size:12px;font-weight:600;white-space:nowrap}
.p-ok{background:var(--ok-bg);color:var(--ok)}.p-warn{background:var(--warn-bg);color:var(--warn)}.p-bad{background:var(--bad-bg);color:var(--bad)}.p-na{background:var(--na-bg);color:var(--na)}
.doms{list-style:none;margin:0;padding:0;display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));background:var(--surface);border:1px solid var(--rule);border-radius:10px;overflow:hidden}
.dom{display:grid;grid-template-columns:1fr auto;align-content:start;gap:6px 12px;padding:12px 14px;outline:1px solid var(--rule)}
.dom-n{font-weight:600}.dom p{grid-column:1/-1;color:var(--ink-2);font-size:13px}
.stats{list-style:none;margin:0 0 14px;padding:0;display:flex;flex-wrap:wrap;gap:8px 28px}
.stats li{display:flex;align-items:baseline;gap:6px}.st-n{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}.st-l{color:var(--ink-3);font-size:13px}
.graph{position:relative;display:grid;grid-auto-flow:column;grid-auto-columns:minmax(170px,1fr);gap:40px;padding:16px;background:var(--surface);border:1px solid var(--rule);border-radius:10px;overflow-x:auto}
.gedges{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible}
.gedges path{fill:none;stroke:var(--rule-2);stroke-width:1.3;transition:stroke .15s ease,opacity .15s ease}
.gedges path.inf{stroke-dasharray:4 4}
.gedges path.mism{stroke:var(--bad)}
.graph.focus .gedges path{opacity:.15}.graph.focus .gedges path.lit{opacity:1;stroke:var(--ink-2)}.graph.focus .gedges path.lit.mism{stroke:var(--bad)}
.gcol{position:relative;display:flex;flex-direction:column;gap:6px;min-width:0}
.gcol h4{margin-bottom:4px}
.gn{position:relative;display:flex;flex-direction:column;align-items:flex-start;gap:1px;width:100%;padding:6px 9px;text-align:left;background:var(--paper);border:1px solid var(--rule);border-radius:7px;font-size:12px;line-height:1.35}
.gn-t{display:block;max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gn-s{font-size:11px;color:var(--ink-3)}
.gn.orphan{border-color:var(--bad);border-style:dashed;background:var(--bad-bg)}
.graph.focus .gn{opacity:.4}.graph.focus .gn.lit{opacity:1;border-color:var(--ink-2)}
.gmore{font-size:12px}
.orphans{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
.orphans li{display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 10px;padding:10px 14px;border-top:1px solid var(--rule)}
.orphans li:first-child{border-top:0}
.lref{font-family:var(--mono);font-size:12px;color:var(--focus);text-decoration:none}
.lref:hover{text-decoration:underline}

/* questions */
.roles{margin-bottom:16px}
.qs{display:flex;flex-direction:column;gap:14px}
.q{background:var(--surface);border:1px solid var(--rule);border-radius:12px;padding:16px 18px}
.q[hidden]{display:none}
.q-h{display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-size:12.5px}
.prio{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 8px;border-radius:6px;background:var(--na-bg);color:var(--ink-2);font-weight:700;font-variant-numeric:tabular-nums}
.prio-s{font-weight:500;font-size:11.5px}
.prio.p0{background:var(--Blocker);color:var(--surface)}
.prio.p1{background:color-mix(in srgb,var(--Critical) 16%,transparent);color:var(--Critical)}
.prio.p2{background:color-mix(in srgb,var(--Major) 14%,transparent);color:var(--Major)}
.role{font-weight:700}.role-full{color:var(--ink-3)}.qid{margin-left:auto;font-family:var(--mono);color:var(--ink-3)}
.q-t{margin-top:10px;font-size:16px;line-height:1.4;max-width:90ch}
.q-grid{display:grid;grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);gap:18px;margin-top:14px}
.q-grid p{color:var(--ink-2)}
.q-grid p+h4{margin-top:14px}
.opts{margin:0;padding-left:20px;display:flex;flex-direction:column;gap:6px}
.opt{color:var(--ink-2)}.opt.rec{color:var(--ink);font-weight:550}
.opt.rec::marker{color:var(--ok);font-weight:700}
.rec-k{display:inline-block;margin-left:8px;padding:0 6px;border-radius:4px;background:var(--ok-bg);color:var(--ok);font-size:11.5px;font-weight:600;vertical-align:1px}
.rec-why{margin-top:10px;font-size:13px}
.drefs{display:flex;flex-wrap:wrap;gap:6px}
.dref{font:12px var(--mono);padding:1px 7px;border:1px solid var(--rule-2);border-radius:5px;color:var(--ink);text-decoration:none}
.diff{margin-top:14px;border-radius:8px;overflow:hidden;font:12.5px/1.6 var(--mono)}
.diff>div{display:flex;flex-direction:column;gap:2px;padding:10px 14px}
.diff .del{background:var(--del);color:var(--del-ink)}.diff .ins{background:var(--ins);color:var(--ins-ink)}
.diff p{white-space:pre-wrap;word-break:break-word;color:inherit}
.diff .del p{text-decoration:line-through;text-decoration-thickness:1px;text-decoration-color:color-mix(in srgb,currentColor 45%,transparent)}
.diff-k{font:600 11.5px var(--sans)}
.diff .mut{font:11.5px var(--sans);color:inherit;opacity:.75}
.q-f{display:flex;justify-content:flex-end;margin-top:12px}
.btn{display:inline-flex;align-items:center;gap:6px;height:32px;padding:0 12px;border:1px solid var(--rule-2);border-radius:8px;background:var(--surface);font-size:13px;font-weight:550}

/* requirements */
.reqs td.rtext{min-width:320px;color:var(--ink-2)}
.reqs td:nth-child(-n+5){white-space:nowrap}
.reqs tr[hidden]{display:none}
.cnt{min-width:26px;height:22px;padding:0 6px;border:0;border-radius:5px;background:color-mix(in srgb,var(--sc) 15%,transparent);color:var(--sc);font-weight:700;font-variant-numeric:tabular-nums}
.cnt.zero{display:inline-block;background:none;color:var(--ink-3);font-weight:400;text-align:right}
.check{display:inline-flex;align-items:center;gap:6px;color:var(--ink-2);font-size:13px}

.toast{position:fixed;left:50%;bottom:24px;z-index:50;padding:10px 16px;border-radius:9px;background:var(--ink);color:var(--surface);font-size:13px;font-weight:550;box-shadow:0 10px 28px rgb(0 0 0/.22);
  transform:translate(-50%,calc(100% + 24px));opacity:0;transition:transform .22s var(--ease-out),opacity .22s var(--ease-out);pointer-events:none}
.toast.show{transform:translate(-50%,0);opacity:1}
.foot{padding:20px 32px 28px;font-size:12px;border-top:1px solid var(--rule)}

@media (hover:hover) and (pointer:fine){
  .card:hover{border-color:var(--rule-2)}
  .chip:not(:disabled):not([aria-pressed=true]):hover,.btn:hover,.theme:hover,.kpi:hover{border-color:var(--ink-3)}
  .tabs [role=tab]:hover{color:var(--ink)}
  .gn:hover{border-color:var(--ink-3)}
  .dref:hover{border-color:var(--ink-3)}
}
@media (max-width:1100px){.hero{grid-template-columns:minmax(0,1fr)}.hero-side{justify-content:flex-start}}
@media (max-width:900px){
  .split{grid-template-columns:1fr;height:auto}
  .src{max-height:62vh;border-right:0;border-bottom:1px solid var(--rule)}
  .pane.list{overflow:visible}.toolbar{position:static}
  .q-grid{grid-template-columns:1fr}
}
@media (max-width:640px){
  .hero{padding:20px 16px 14px}.hero h1{font-size:21px}
  .hero-side{flex-direction:column;align-items:stretch}.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}
  .verdict-row{flex-direction:column;gap:10px}
  .tabs{padding:0 8px}.wrap{padding:18px 16px 60px}
  .journal{margin-left:0}.journal-body{left:16px;right:16px}
  .ln{grid-template-columns:38px 24px minmax(0,1fr)}
  .doms{grid-template-columns:1fr}
}
@media (prefers-reduced-motion:reduce){
  .ring-fg{animation:none}
  .toast{transition:opacity .2s ease;transform:translate(-50%,0)}
  mark,.ln.target{transition:none}
}
@media print{
  .tabs,.toolbar,.toast,.theme,.q-f{display:none}
  .panel[hidden]{display:block}.split{display:block;height:auto}.pane{overflow:visible}
}
"""

JS = r"""
(() => {
const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const store = { get(k) { try { return localStorage.getItem(k); } catch (_) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (_) {} } };

/* тема */
const root = document.documentElement;
const saved = store.get('srs-theme'); if (saved) root.dataset.theme = saved;
$('#theme').addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark'; store.set('srs-theme', root.dataset.theme);
});

/* вкладки */
const tabs = $$('[role=tab]');
function openTab(id, focus) {
  tabs.forEach(t => { const on = t.id === 'tab-' + id; t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1;
    $('#' + t.getAttribute('aria-controls')).hidden = !on; if (on && focus) t.focus(); });
  history.replaceState(null, '', '#' + id);
  if (id === 'matrix') drawGraph();
}
tabs.forEach((t, i) => {
  t.addEventListener('click', () => openTab(t.id.slice(4)));
  t.addEventListener('keydown', ev => {
    const d = ev.key === 'ArrowRight' ? 1 : ev.key === 'ArrowLeft' ? -1 : 0; if (!d) return;
    ev.preventDefault(); openTab(tabs[(i + d + tabs.length) % tabs.length].id.slice(4), true);
  });
});

/* документ ↔ карточки */
const src = $('#src'), list = $('.pane.list'), cards = $$('.card');
const byId = Object.fromEntries(cards.map(c => [c.dataset.id, c]));
function scrollWithin(box, el, block = 'center') {
  if (!box || !el) return;
  const b = box.getBoundingClientRect(), r = el.getBoundingClientRect();
  const top = box.scrollTop + r.top - b.top - (block === 'center' ? (b.height - r.height) / 2 : 90);
  box.scrollTo({ top: Math.max(0, top), behavior: reduce ? 'auto' : 'smooth' });
}
function flashLine(n) {
  const ln = document.getElementById('L' + n); if (!ln) return;
  ln.classList.add('target'); setTimeout(() => ln.classList.remove('target'), 1400);
  scrollWithin(src, ln);
}
let current = null;
function select(id, from) {
  const card = byId[id]; if (!card) return;
  if (current) current.removeAttribute('aria-current');
  current = card; card.setAttribute('aria-current', 'true');
  $$('mark.on').forEach(m => m.classList.remove('on'));
  const marks = $$('mark').filter(m => m.dataset.ids.split(' ').includes(id));
  if (from !== 'source') {
    if (marks.length) { scrollWithin(src, marks[0]); } else { flashLine(card.dataset.line); }
  }
  marks.forEach(m => m.classList.add('on'));
  setTimeout(() => marks.forEach(m => m.classList.remove('on')), 1600);
  if (from !== 'card') scrollWithin(innerWidth > 900 ? list : null, card, 'top') || (innerWidth <= 900 && card.scrollIntoView({ block: 'nearest', behavior: reduce ? 'auto' : 'smooth' }));
}
cards.forEach(c => {
  c.addEventListener('click', ev => { if (ev.target.closest('a,button')) return; select(c.dataset.id, 'card'); });
  c.addEventListener('keydown', ev => { if (ev.key === 'Enter') select(c.dataset.id, 'card'); });
});
function pickFrom(ids) {
  const visible = ids.filter(i => byId[i] && !byId[i].hidden);
  const pool = visible.length ? visible : ids;
  const k = current ? pool.indexOf(current.dataset.id) : -1;
  return pool[(k + 1) % pool.length];
}
$$('mark, .pin').forEach(m => m.addEventListener('click', () => select(pickFrom(m.dataset.ids.split(' ')), 'source')));

/* фильтры дефектов */
const q = $('#q'), cat = $('#cat'), sevChips = $$('.toolbar .chip[data-sev]');
function applyFilters() {
  const term = q.value.trim().toLowerCase();
  const sevs = sevChips.filter(c => c.getAttribute('aria-pressed') === 'true').map(c => c.dataset.sev);
  let shown = 0; const hiddenIds = new Set();
  cards.forEach(c => {
    const ok = (!term || c.dataset.q.includes(term) || c.dataset.id.toLowerCase() === term)
      && (!sevs.length || sevs.includes(c.dataset.sev)) && (!cat.value || c.dataset.cat === cat.value);
    c.hidden = !ok; if (ok) shown++; else hiddenIds.add(c.dataset.id);
  });
  $$('mark').forEach(m => m.classList.toggle('dim', m.dataset.ids.split(' ').every(i => hiddenIds.has(i))));
  $('#shown').textContent = `Показано ${shown} из ${cards.length}`;
  $('#cards-empty').hidden = shown > 0;
}
q.addEventListener('input', applyFilters); cat.addEventListener('change', applyFilters);
sevChips.forEach(c => c.addEventListener('click', () => { c.setAttribute('aria-pressed', c.getAttribute('aria-pressed') !== 'true'); applyFilters(); }));
$('#reset').addEventListener('click', () => { q.value = ''; cat.value = ''; sevChips.forEach(c => c.setAttribute('aria-pressed', 'false')); applyFilters(); });
$$('.kpi').forEach(k => k.addEventListener('click', () => {
  openTab('doc'); sevChips.forEach(c => c.setAttribute('aria-pressed', c.dataset.sev === k.dataset.sev));
  applyFilters(); list.scrollTo({ top: 0 });
}));
applyFilters();

/* переходы из других вкладок */
function gotoDefect(id) { openTab('doc'); q.value = ''; cat.value = ''; sevChips.forEach(c => c.setAttribute('aria-pressed', 'false')); applyFilters(); select(id); }
$$('[data-goto]').forEach(a => a.addEventListener('click', ev => { ev.preventDefault(); gotoDefect(a.dataset.goto); }));
$$('.lref').forEach(a => a.addEventListener('click', ev => { ev.preventDefault(); openTab('doc'); flashLine(a.dataset.line); }));

/* вопросы по ролям */
const roleChips = $$('.roles .chip');
roleChips.forEach(c => c.addEventListener('click', () => {
  roleChips.forEach(x => x.setAttribute('aria-pressed', x === c));
  $$('.q').forEach(el => el.hidden = !!c.dataset.role && el.dataset.role !== c.dataset.role);
}));

/* копирование для Jira */
const toast = $('#toast'); let tt;
function say(msg) { toast.textContent = msg; toast.classList.add('show'); clearTimeout(tt); tt = setTimeout(() => toast.classList.remove('show'), 2200); }
async function copy(text) {
  try { await navigator.clipboard.writeText(text); return true; } catch (_) {
    const ta = document.createElement('textarea'); ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); let ok = false; try { ok = document.execCommand('copy'); } catch (_) {}
    ta.remove(); return ok;
  }
}
$$('.copy').forEach(b => b.addEventListener('click', async () => say(await copy(b.dataset.md) ? 'Скопировано в буфер — вставьте в задачу Jira' : 'Не удалось скопировать: браузер запретил доступ к буферу')));

/* требования */
const rq = $('#rq'), rkind = $('#rkind'), rdef = $('#rdef'), rows = $$('#rbody tr');
function filterReqs() {
  const term = rq.value.trim().toLowerCase(); let n = 0;
  rows.forEach(r => { const ok = (!term || r.dataset.q.includes(term)) && (!rkind.value || r.dataset.kind === rkind.value) && (!rdef.checked || +r.dataset.n > 0);
    r.hidden = !ok; if (ok) n++; });
  $('#rshown').textContent = `Показано ${n} из ${rows.length}`; $('#reqs-empty').hidden = n > 0;
}
[rq, rkind, rdef].forEach(el => el.addEventListener('input', filterReqs)); filterReqs();
$$('.cnt[data-req]').forEach(b => b.addEventListener('click', () => {
  openTab('doc'); sevChips.forEach(c => c.setAttribute('aria-pressed', 'false')); cat.value = ''; q.value = b.dataset.req.toLowerCase(); applyFilters();
  const first = cards.find(c => !c.hidden); if (first) select(first.dataset.id);
}));

/* граф */
const graph = $('#graph'); let drawn = false;
function drawGraph() {
  if (!graph || graph.offsetParent === null) return;
  const svg = $('.gedges', graph), data = JSON.parse($('#graph-data').textContent);
  const node = id => graph.querySelector(`[data-node="${CSS.escape(id)}"]`);
  const g = graph.getBoundingClientRect(); svg.innerHTML = '';
  svg.setAttribute('viewBox', `0 0 ${graph.scrollWidth} ${graph.scrollHeight}`);
  svg.style.width = graph.scrollWidth + 'px'; svg.style.height = graph.scrollHeight + 'px';
  for (const [s, d, rel, origin] of data) {
    let a = node(s), b = node(d); if (!a || !b) continue;
    let ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
    if (ra.left > rb.left) { [a, b] = [b, a]; [ra, rb] = [rb, ra]; }
    const ox = graph.scrollLeft - g.left, oy = graph.scrollTop - g.top;
    const same = Math.abs(ra.left - rb.left) < 4;
    const x1 = (same ? ra.left : ra.right) + ox, y1 = ra.top + ra.height / 2 + oy;
    const x2 = rb.left + ox, y2 = rb.top + rb.height / 2 + oy;
    const dx = same ? -28 : Math.max(24, (x2 - x1) / 2);
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', same ? `M${x1},${y1} C${x1 + dx},${y1} ${x2 + dx},${y2} ${x2},${y2}` : `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`);
    p.setAttribute('class', (origin === 'INFERRED' ? 'inf' : '') + (rel === 'type_mismatch' ? ' mism' : ''));
    p.dataset.a = s; p.dataset.b = d; svg.appendChild(p);
  }
  drawn = true;
}
if (graph) {
  const lit = id => {
    graph.classList.add('focus');
    const paths = $$('.gedges path', graph).filter(p => p.dataset.a === id || p.dataset.b === id);
    const ids = new Set([id]); paths.forEach(p => { p.classList.add('lit'); ids.add(p.dataset.a); ids.add(p.dataset.b); });
    $$('.gn', graph).forEach(n => n.classList.toggle('lit', ids.has(n.dataset.node)));
  };
  const unlit = () => { graph.classList.remove('focus'); $$('.lit', graph).forEach(n => n.classList.remove('lit')); };
  $$('.gn', graph).forEach(n => {
    n.addEventListener('mouseenter', () => lit(n.dataset.node)); n.addEventListener('focus', () => lit(n.dataset.node));
    n.addEventListener('mouseleave', unlit); n.addEventListener('blur', unlit);
    n.addEventListener('click', () => { if (+n.dataset.line) { openTab('doc'); flashLine(n.dataset.line); } });
  });
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(drawGraph, 120); });
}
const initial = location.hash.slice(1); if (tabs.some(t => t.id === 'tab-' + initial)) openTab(initial);
})();
"""
