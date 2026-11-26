"""Лёгкий граф трассировки документа (детерминированно, без внешних зависимостей).

Узлы: требования, критерии приёмки, поля UI, бизнес-операции (чтение/запись), эндпоинты API,
параметры с типами, HTTP-статусы. Рёбра помечены происхождением:

* ``EXTRACTED`` — связь прямо записана в документе (ссылка «Ф-2», параметр в разделе метода,
  JSON-пример под заголовком статуса);
* ``INFERRED`` — связь выведена кодом по совпадению основ слов и семантике операций
  (критерий приёмки ↔ функциональное правило, поле UI ↔ параметр API, операция ↔ эндпоинт).

Из графа выводятся orphan-узлы: функциональные правила без критерия приёмки, поля UI без
параметра API, операции бизнес-части без эндпоинта, статусы без описания или примера.
Компактная сводка (``summary_for_llm``) уходит в CrossCheckFinTechAgent вместо того, чтобы
заставлять модель самой искать эти связи по всему тексту — на 100-страничных ТЗ это главное
средство удержать промпт и внимание модели в рамках.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .analyzers.deterministic import MUTATION_WORD, STATUS_MENTION, URL_PATH
from .models import GraphEdge, GraphNode, GraphOrphan, ParsedDocument, Requirement, TraceGraph

TYPE_WORD = re.compile(
    r"^(?:string|str|integer|int|int32|int64|long|number|float|double|decimal|boolean|bool|object|array|date|datetime|uuid)\b",
    re.I,
)
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
REQUEST_SECTION = re.compile(r"параметр|query|path|request|запрос|header|заголов|тело", re.I)
RESPONSE_SECTION = re.compile(r"ответ|response|результат", re.I)
STATUS_HEADING = re.compile(r"^\s*(?:HTTP\s*)?([1-5]\d\d)\b")
DOC_REF = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{1,6}[-‑]\d{1,4}(?:\.\d+)*)(?![\w-])")
READ_VERB = re.compile(r"чита|чтени|получа\w*\s+(?:настро|данн|параметр)|показ|отображ|просмотр|загруж|\bread|\bfetch|\bload", re.I)
SETTINGS_OBJECT = re.compile(r"настро|порог|лимит|параметр|профил|данн\w*\s+(?:карт|клиент)", re.I)
WRITE_VERB = re.compile(r"сохран|измен|задаё|задае|задат|созда|удал|обнов|установ|отключ|включ|save|update|set|create|delete", re.I)
UI_ENUM = re.compile(r"(?:видн\w*|отображ\w*|содерж\w*|есть|располож\w*|показ\w*)\s*:?\s*(.+)$", re.I)
TOGGLE = re.compile(r"переключател|чекбокс|флаж|toggle|switch", re.I)
BOOLISH = re.compile(r"включ|enabled|enable|active|flag|признак|0\s*(?:или|/)\s*1|true|false|bool", re.I)
BUTTON = re.compile(r"кнопк", re.I)
STOP_STEMS = {
    "котор", "этого", "также", "долже", "должн", "может", "проис", "необх", "систе", "серви", "клиен",
    "получ", "прове", "через", "после", "перед", "между", "когда", "всего", "чтобы",
}


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-zа-яё]{4,}", text.lower().replace("ё", "е"))
    return {w[:5] for w in words if w[:5] not in STOP_STEMS}


def _short(text: str, n: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


class _Builder:
    def __init__(self) -> None:
        self.g = TraceGraph()
        self._ids: set[str] = set()

    def node(self, node_id: str, kind: str, label: str, line: int = 0, req_id: str = "") -> str:
        if node_id not in self._ids:
            self._ids.add(node_id)
            self.g.nodes.append(GraphNode(node_id=node_id, kind=kind, label=label, line=line, req_id=req_id))  # type: ignore[arg-type]
        return node_id

    def edge(self, src: str, dst: str, relation: str, origin: str, note: str = "") -> None:
        if not any(e.src == src and e.dst == dst and e.relation == relation for e in self.g.edges):
            self.g.edges.append(GraphEdge(src=src, dst=dst, relation=relation, origin=origin, note=note))  # type: ignore[arg-type]

    def orphan(self, node_id: str, kind: str, label: str, line: int, explanation: str) -> None:
        self.g.orphans.append(GraphOrphan(node_id=node_id, kind=kind, label=label, line=line, explanation=explanation))


def build_graph(doc: ParsedDocument, reqs: list[Requirement]) -> TraceGraph:
    b = _Builder()
    by_source = {r.source_id: r for r in reqs if r.source_id}

    # --- требования, критерии приёмки, явные ссылки «Ф-2»
    functional = [r for r in reqs if r.kind == "functional"]
    acceptance = [r for r in reqs if r.kind == "acceptance"]
    for r in reqs:
        if r.kind in ("functional", "nonfunctional", "business", "data", "integration", "ui", "process"):
            body = re.sub(rf"^\s*{re.escape(r.source_id)}[.):]?\s*", "", r.text) if r.source_id else r.text
            b.node(r.req_id, "requirement", f"{r.source_id or r.req_id}: {_short(body, 60)}", r.line_start, r.req_id)
        elif r.kind == "acceptance":
            b.node(r.req_id, "acceptance", _short(r.text, 70), r.line_start, r.req_id)
    for r in reqs:
        for m in DOC_REF.finditer(r.text):
            tgt = by_source.get(m.group(1))
            if tgt and tgt.req_id != r.req_id and r.req_id in b._ids and tgt.req_id in b._ids:
                rel = "verifies" if r.kind == "acceptance" else "references"
                b.edge(r.req_id, tgt.req_id, rel, "EXTRACTED", f"ссылка «{m.group(1)}» в тексте")

    # критерий приёмки ↔ функциональное правило: основы слов с весом IDF по функциональным правилам
    if functional and acceptance:
        stems = {r.req_id: _stems(r.text) for r in functional}
        df: dict[str, int] = defaultdict(int)
        for s in stems.values():
            for st in s:
                df[st] += 1
        idf = {st: math.log(1 + len(functional) / c) for st, c in df.items()}
        for a in acceptance:
            if any(e.src == a.req_id and e.relation == "verifies" for e in b.g.edges):
                continue
            sa = _stems(a.text)
            scores = {fid: sum(idf[s] for s in sa & st) for fid, st in stems.items()}
            top = max(scores.values(), default=0.0)
            if top < 1.0:
                continue
            for fid, sc in scores.items():
                if sc >= max(1.0, 0.6 * top):
                    shared = sorted(sa & stems[fid], key=lambda s: -idf[s])[:3]
                    b.edge(a.req_id, fid, "verifies", "INFERRED", "общие термины: " + ", ".join(shared))
        for f in functional:
            if not any(e.dst == f.req_id and e.relation == "verifies" for e in b.g.edges):
                b.orphan(f.req_id, "requirement_without_acceptance", f.source_id or f.req_id, f.line_start,
                         "ни один критерий приёмки не проверяет это правило")

    # --- API: эндпоинты, параметры, статусы
    api_blocks = [x for x in doc.blocks if re.search(r"\bapi\b|rest|endpoint|эндпоинт|метод", x.section_path, re.I)]
    endpoints: list[tuple[str, bool]] = []  # (node_id, изменяет состояние)
    if api_blocks:
        method = next((x.cells[1].strip().upper() for x in api_blocks if x.kind == "table_row" and x.cells
                       and len(x.cells) > 1 and re.match(r"(rest type|http method|method|метод)", x.cells[0], re.I)), "")
        descr = next((x.cells[1] for x in api_blocks if x.kind == "table_row" and x.cells and len(x.cells) > 1
                      and re.match(r"(description|описание)", x.cells[0], re.I)), "")
        paths: dict[str, int] = {}
        for x in api_blocks:
            if x.kind == "table_row" and x.cells and re.match(r"(request url|url|endpoint|путь)", x.cells[0], re.I):
                for m in URL_PATH.finditer(" ".join(x.cells[1:])):
                    paths.setdefault(m.group(0).rstrip("."), x.line_start)
        for path, line in paths.items():
            first = descr.split(".")[0] if descr else ""
            mut = bool(MUTATION_WORD.search(path + " " + first)) or method in ("POST", "PUT", "PATCH", "DELETE")
            nid = b.node(f"EP {method or '?'} {path}", "endpoint", f"{method or '?'} {path}" + (f" — {_short(first, 40)}" if first else ""), line)
            endpoints.append((nid, mut))

        ep_default = endpoints[0][0] if len(endpoints) == 1 else ""
        params: dict[tuple[str, str], tuple[str, str, str]] = {}  # (side, name) -> (node_id, type, description)
        for x in api_blocks:
            if x.kind != "table_row" or not x.cells or len(x.cells) < 2:
                continue
            name, typ = x.cells[0].strip().strip("`"), x.cells[1].strip()
            if not IDENT.match(name) or not TYPE_WORD.match(typ):
                continue
            leaf = x.section_path.split("›")[-1]
            side = "resp" if RESPONSE_SECTION.search(leaf) else "req" if REQUEST_SECTION.search(x.section_path) else ""
            if not side:
                continue
            desc = " ".join(c for c in x.cells[2:] if c and c not in ("—", "-"))
            nid = b.node(f"P {side}:{name}", "param", f"{'запрос' if side == 'req' else 'ответ'} · {name}: {typ.lower()}", x.line_start)
            params[(side, name)] = (nid, typ.lower(), desc)
            if ep_default:
                b.edge(ep_default, nid, "has_param" if side == "req" else "returns", "EXTRACTED", f"раздел «{leaf.strip()}»")
        for (side, name), (nid, typ, _d) in params.items():
            if side == "req" and ("resp", name) in params:
                rid, rtyp, _ = params[("resp", name)]
                if TYPE_WORD.match(typ).group(0) != TYPE_WORD.match(rtyp).group(0):  # type: ignore[union-attr]
                    b.edge(nid, rid, "type_mismatch", "INFERRED", f"{typ} → {rtyp}")

        # статусы: описанные заголовком, с примером, упомянутые в тексте
        described: dict[str, int] = {}
        examples: set[str] = set()
        for x in api_blocks:
            if x.kind == "heading":
                m = STATUS_HEADING.match(x.text)
                if m:
                    described.setdefault(m.group(1), x.line_start)
            if x.kind == "code":
                for code in described:
                    if re.search(rf"(?:^|›)\s*{code}\b", x.section_path):
                        examples.add(code)
        mentioned: dict[str, int] = {}
        for x in api_blocks:
            if x.kind in ("paragraph", "list_item", "table_row"):
                for m in STATUS_MENTION.finditer(x.text):
                    mentioned.setdefault(m.group(1), x.line_start)
        for code in sorted(set(described) | set(mentioned)):
            nid = b.node(f"S {code}", "status", f"HTTP {code}", described.get(code) or mentioned.get(code, 0))
            if ep_default:
                b.edge(ep_default, nid, "returns", "EXTRACTED" if code in described else "INFERRED",
                       "заголовок статуса" if code in described else "упомянут в тексте")
            if code in examples:
                b.edge(nid, nid, "has_example", "EXTRACTED", "JSON-пример под заголовком")
            if code not in described:
                b.orphan(nid, "status_undescribed", f"HTTP {code}", mentioned[code], "код упомянут, но не описан в разделе статусов")
            elif code not in examples:
                b.orphan(nid, "status_without_example", f"HTTP {code}", described[code], "у статуса нет примера ответа")

        # --- UI-поля → параметры / эндпоинты
        for r in reqs:
            if r.kind != "ui":
                continue
            m = UI_ENUM.search(r.text)
            if not m:
                continue
            # делим перечисление по запятым и «и», не разрывая подписи в кавычках («Порог, ₸»)
            items = [i.strip(" .;") for i in re.split(r",\s*(?![^«]*»)|\s+и\s+(?![^«]*»)", m.group(1))
                     if len(i.strip(" .;")) > 2]
            for item in items:
                fid = b.node(f"UI {item}", "ui_field", item, r.line_start, r.req_id)
                b.edge(r.req_id, fid, "contains", "EXTRACTED")
                target = None
                if BUTTON.search(item):
                    verb_w = WRITE_VERB.search(item)
                    target = next((nid for nid, mut in endpoints if mut == bool(verb_w)), None)
                    rel = "implements"
                else:
                    rel = "maps_to"
                    st_item = _stems(item)
                    best, best_sc = None, 0
                    for (side, name), (pid, typ, desc) in params.items():
                        if side != "req":
                            continue
                        sc = len(st_item & _stems(desc + " " + name))
                        if TOGGLE.search(item) and BOOLISH.search(f"{name} {typ} {desc}"):
                            sc += 2
                        if sc > best_sc:
                            best, best_sc = pid, sc
                    target = best
                if target:
                    b.edge(fid, target, rel, "INFERRED")
                else:
                    b.orphan(fid, "ui_field_without_api", item, r.line_start,
                             "элемент экрана не связан ни с одним параметром или методом API")

        # --- операции бизнес-части (чтение/запись) → эндпоинты
        for r in reqs:
            # операции, которые бизнес-часть ждёт от API: только интеграционные и UI-требования
            # про данные настроек (функциональные правила описывают поведение сервиса, а не вызовы API)
            if r.kind not in ("integration", "ui") or not SETTINGS_OBJECT.search(r.text):
                continue
            ops = []
            if READ_VERB.search(r.text):
                ops.append(("read", "чтение"))
            if WRITE_VERB.search(r.text):
                ops.append(("write", "запись"))
            for op, ru in ops:
                oid = b.node(f"OP {op} {r.req_id}", "operation", f"{ru}: {_short(r.text, 50)}", r.line_start, r.req_id)
                b.edge(r.req_id, oid, "requires", "EXTRACTED")
                match = [nid for nid, mut in endpoints if mut == (op == "write")]
                for nid in match:
                    b.edge(oid, nid, "implements", "INFERRED")
                if not match:
                    b.orphan(oid, "operation_without_endpoint", f"{ru} ({r.source_id or r.req_id})", r.line_start,
                             f"бизнес-часть требует операцию «{ru}», но в API нет подходящего метода")
    return b.g


ORPHAN_RU = {
    "requirement_without_acceptance": "Правило без критерия приёмки",
    "ui_field_without_api": "Поле UI без API",
    "operation_without_endpoint": "Операция без метода API",
    "status_undescribed": "Статус упомянут, но не описан",
    "status_without_example": "Статус без примера",
}
NODE_KIND_RU = {
    "requirement": "требования", "acceptance": "критерии приёмки", "ui_field": "поля UI", "operation": "операции",
    "endpoint": "эндпоинты", "param": "параметры API", "status": "HTTP-статусы",
}


def summary_for_llm(g: TraceGraph, limit: int = 40) -> str:
    """Компактная сводка графа для промпта CrossCheck (не больше ~limit строк)."""
    if not g.nodes:
        return "(граф не построен)"
    counts = defaultdict(int)
    for n in g.nodes:
        counts[n.kind] += 1
    ext = sum(e.origin == "EXTRACTED" for e in g.edges)
    lines = [
        "Узлы: " + ", ".join(f"{NODE_KIND_RU[k]} {v}" for k, v in counts.items())
        + f". Рёбра: {len(g.edges)} (EXTRACTED {ext}, INFERRED {len(g.edges) - ext}).",
    ]
    labels = {n.node_id: n.label for n in g.nodes}
    for e in g.edges:
        if e.relation in ("verifies", "type_mismatch", "implements") and len(lines) < limit:
            lines.append(f"- [{e.origin}] {labels.get(e.src, e.src)} —{e.relation}→ {labels.get(e.dst, e.dst)}"
                         + (f" ({e.note})" if e.note else ""))
    if g.orphans:
        lines.append("ORPHAN-УЗЛЫ (кандидаты в пропуски — проверь и подтверди цитатой):")
        for o in g.orphans[: max(5, limit - len(lines))]:
            lines.append(f"- {ORPHAN_RU.get(o.kind, o.kind)}: {o.label} (стр. {o.line}) — {o.explanation}")
    return "\n".join(lines)
