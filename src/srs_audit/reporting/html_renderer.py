"""Интерактивная HTML-подсветка исходного документа по верифицированным спанам дефектов."""

from __future__ import annotations

import html
import json

from ..grounding import Grounder
from ..models import CATEGORY_RU, SEVERITY_ORDER, AuditReport

CSS = """
:root{--bg:#fff;--fg:#1d2433;--muted:#6b7280;--line:#e5e7eb;--panel:#f8fafc;
--Blocker:#7f1d1d;--Critical:#dc2626;--Major:#ea580c;--Minor:#ca8a04}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e5e7eb;--muted:#9ca3af;--line:#262b36;--panel:#161a22}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:16px 20px;border-bottom:1px solid var(--line)}h1{font-size:18px;margin:0 0 4px}
.v{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:700;color:#fff}
.v.NO-GO{background:#dc2626}.v.GO{background:#16a34a}.v.CONDITIONAL{background:#ca8a04}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;min-height:calc(100vh - 70px)}
@media (max-width:900px){main{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}}
pre{margin:0;padding:12px 0;font:12.5px/1.6 ui-monospace,Consolas,monospace;white-space:pre-wrap;word-break:break-word}
.ln{display:flex}.ln .no{flex:0 0 52px;text-align:right;padding-right:12px;color:var(--muted);user-select:none}
.ln .tx{flex:1;padding-right:12px}.ln.hit{background:color-mix(in srgb,var(--Major) 6%,transparent)}
mark{background:transparent;border-bottom:2px solid;cursor:pointer;color:inherit;padding:0 1px}
mark.Blocker{border-color:var(--Blocker);background:color-mix(in srgb,var(--Blocker) 18%,transparent)}
mark.Critical{border-color:var(--Critical);background:color-mix(in srgb,var(--Critical) 14%,transparent)}
mark.Major{border-color:var(--Major);background:color-mix(in srgb,var(--Major) 12%,transparent)}
mark.Minor{border-color:var(--Minor);background:color-mix(in srgb,var(--Minor) 12%,transparent)}
aside{border-left:1px solid var(--line);background:var(--panel);padding:12px;overflow:auto;max-height:calc(100vh - 70px);position:sticky;top:0}
.f{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}.f button{border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:6px;padding:3px 8px;cursor:pointer}
.f button.on{outline:2px solid var(--Major)}
.card{border:1px solid var(--line);border-left:4px solid;border-radius:6px;padding:8px 10px;margin-bottom:8px;background:var(--bg);cursor:pointer}
.card.Blocker{border-left-color:var(--Blocker)}.card.Critical{border-left-color:var(--Critical)}.card.Major{border-left-color:var(--Major)}.card.Minor{border-left-color:var(--Minor)}
.card .h{font-weight:600}.card .m{color:var(--muted);font-size:12px}.card.active{box-shadow:0 0 0 2px var(--Major)}
.flash{animation:fl 1.2s}@keyframes fl{0%{background:color-mix(in srgb,var(--Major) 40%,transparent)}}
"""

JS = """
const cards=[...document.querySelectorAll('.card')];
function go(id){const m=document.querySelector('mark[data-ids~="'+id+'"]');if(m){m.scrollIntoView({block:'center',behavior:'smooth'});m.classList.add('flash');setTimeout(()=>m.classList.remove('flash'),1300);}
cards.forEach(c=>c.classList.toggle('active',c.dataset.id===id));}
cards.forEach(c=>c.addEventListener('click',()=>go(c.dataset.id)));
document.querySelectorAll('mark').forEach(m=>m.addEventListener('click',()=>{const id=m.dataset.ids.split(' ')[0];const c=document.querySelector('.card[data-id="'+id+'"]');if(c){c.scrollIntoView({block:'center',behavior:'smooth'});go(id);}}));
document.querySelectorAll('.f button').forEach(b=>b.addEventListener('click',()=>{b.classList.toggle('on');const on=[...document.querySelectorAll('.f button.on')].map(x=>x.dataset.s);
cards.forEach(c=>c.style.display=(!on.length||on.includes(c.dataset.s))?'':'none');}));
"""


def render_html(report: AuditReport) -> str:
    text = report.source_text
    g = Grounder(text)
    n = len(text)
    owner: list[list[str]] = [[] for _ in range(n)]
    sev_of = {d.defect_id: d.severity for d in report.defects}
    for d in report.defects:
        res = g.ground(d.exact_quote, (d.line_start, d.line_end))
        if res.found:
            for i in range(res.char_start, min(res.char_end, n)):
                owner[i].append(d.defect_id)

    def worst(ids: list[str]) -> str:
        return min((sev_of[i] for i in ids), key=lambda s: SEVERITY_ORDER[s])

    rows = []
    pos = 0
    for no, line in enumerate(text.split("\n"), start=1):
        parts = []
        hit = False
        i = 0
        while i < len(line):
            ids = owner[pos + i] if pos + i < n else []
            j = i
            while j < len(line) and (owner[pos + j] if pos + j < n else []) == ids:
                j += 1
            chunk = html.escape(line[i:j])
            if ids:
                hit = True
                title = html.escape("; ".join(ids))
                parts.append(f'<mark class="{worst(ids)}" data-ids="{" ".join(ids)}" title="{title}">{chunk}</mark>')
            else:
                parts.append(chunk)
            i = j
        rows.append(f'<div class="ln{" hit" if hit else ""}" id="L{no}"><span class="no">{no}</span><span class="tx">{"".join(parts) or "&nbsp;"}</span></div>')
        pos += len(line) + 1

    cards = []
    for d in report.defects:
        cards.append(
            f'<div class="card {d.severity}" data-id="{d.defect_id}" data-s="{d.severity}">'
            f'<div class="h">{d.defect_id} · {d.severity} · {html.escape(CATEGORY_RU[d.category])}</div>'
            f'<div class="m">{html.escape(d.verified_location)} · {html.escape(d.req_id)} · {html.escape(d.standard_ref)}</div>'
            f"<div>{html.escape(d.reasoning)}</div>"
            f'<div class="m">Исправить: {html.escape(d.recommendation)}</div></div>'
        )
    m = report.metrics
    vcls = "CONDITIONAL" if m.verdict == "CONDITIONAL GO" else m.verdict
    sev = " · ".join(f"{k}: {v}" for k, v in m.by_severity.items())
    buttons = "".join(f'<button data-s="{s}">{s} ({m.by_severity.get(s, 0)})</button>' for s in ("Blocker", "Critical", "Major", "Minor"))
    data = json.dumps({"verdict": m.verdict, "score": m.score}, ensure_ascii=False)
    return f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SRS Audit · подсветка</title><style>{CSS}</style></head><body>
<header><h1>{html.escape(report.passport.document_title)}</h1>
<span class="v {vcls}">{m.verdict}</span> Индекс качества {m.score}/100 · дефектов {m.n_defects} ({sev}) · модель {html.escape(report.passport.model)}</header>
<main><pre>{"".join(rows)}</pre><aside><div class="f">{buttons}</div>{"".join(cards)}</aside></main>
<script>const REPORT={data};{JS}</script></body></html>"""
