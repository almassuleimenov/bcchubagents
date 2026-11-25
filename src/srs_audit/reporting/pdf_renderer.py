"""PDF-экспорт отчёта через typst со встроенным кириллическим шрифтом (DejaVu Sans).

Отчёт генерируется нами, поэтому достаточно конвертера подмножества Markdown
(заголовки, абзацы, списки, таблицы, цитаты, **жирный**, `код`, <br>, <sub>, <details>)
в разметку Typst — без внешних пакетов и сетевого доступа.
"""

from __future__ import annotations

import math
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FONT_DIRS = [ROOT / "assets" / "fonts", Path(__file__).resolve().parents[1] / "assets" / "fonts"]

EMOJI = {
    "⛔": "",
    "🔴": "",
    "🟠": "",
    "🟡": "",
    "🟢": "",
    "⚫": "",
    "✅": "✓",
    "⚠️": "!",
    "❌": "✗",
    "📍": "",
    "›": "›",
}


def find_font() -> Path | None:
    for d in FONT_DIRS:
        fonts = sorted(d.glob("*.ttf")) if d.exists() else []
        if fonts:
            return d
    return None


_LONG_TOKEN = re.compile(r"\S{28,}")


def _breakable(text: str) -> str:
    """Typst переносит строки только по пробелам: в длинных токенах (URL, curl) добавляем точки переноса."""
    return _LONG_TOKEN.sub(lambda m: re.sub(r"([/&=?,.:;_-])", "\\1\u200b", m.group(0)), text)


def _esc(text: str) -> str:
    return re.sub(r'([\\#$*_`<>@\[\]~=/+\-"])', r"\\\1", _breakable(text))


def _inline(text: str) -> str:
    for k, v in EMOJI.items():
        text = text.replace(k, v)
    text = text.replace("\\|", "|")
    out: list[str] = []
    pos = 0
    pattern = re.compile(r"\*\*(.+?)\*\*|`([^`]+)`|<br\s*/?>|<sub>(.*?)</sub>|<summary>(.*?)</summary>|</?details>")
    for m in pattern.finditer(text):
        out.append(_esc(text[pos : m.start()]))
        if m.group(1) is not None:
            out.append(f"#strong[{_inline(m.group(1))}]")
        elif m.group(2) is not None:
            code = m.group(2).replace("\\", "\\\\").replace('"', '\\"')
            out.append(f'#raw("{code}")')
        elif m.group(0).startswith("<br"):
            out.append(" #linebreak() ")
        elif m.group(3) is not None:
            out.append(f"#text(size: 7pt, fill: gray)[{_inline(m.group(3))}]")
        elif m.group(4) is not None:
            out.append(f"#strong[{_inline(m.group(4))}]")
        pos = m.end()
    out.append(_esc(text[pos:]))
    return "".join(out)


def _table(rows: list[str]) -> str:
    cells = [[c.strip() for c in re.split(r"(?<!\\)\|", r.strip().strip("|"))] for r in rows]
    cells = [c for c in cells if not all(re.fullmatch(r":?-{2,}:?", x) for x in c if x)]
    if not cells:
        return ""
    n = max(len(r) for r in cells)
    # ширина колонки пропорциональна средней длине текста (короткие ID и статусы — auto)
    widths = []
    for j in range(n):
        vals = [len(r[j]) for r in cells if j < len(r)]
        avg = sum(vals) / max(1, len(vals))
        longest = max(vals, default=0)
        base = max(avg, longest / 2.5)  # учитываем редкие длинные ячейки
        widths.append("auto" if longest < 16 else f"{min(8, max(1, round(math.sqrt(base) / 1.6)))}fr")
    if all(w == "auto" for w in widths):
        widths[-1] = "1fr"
    cols = "(" + ", ".join(widths) + ("," if n == 1 else "") + ")"
    parts = [f"#table(columns: {cols}, inset: 4pt, stroke: 0.4pt + gray,"]
    for i, r in enumerate(cells):
        r = r + [""] * (n - len(r))
        for c in r:
            content = _inline(c)
            parts.append(f"  [#strong[{content}]]," if i == 0 else f"  [{content}],")
    parts.append(")")
    return "#text(size: 8pt)[\n" + "\n".join(parts) + "\n]"


def markdown_to_typst(md: str) -> str:
    lines = md.splitlines()
    out = [
        '#set page(paper: "a4", margin: (x: 1.6cm, y: 1.6cm), numbering: "1 / 1")',
        '#set text(font: "DejaVu Sans", size: 9.5pt, lang: "ru")',
        "#set par(justify: false, leading: 0.55em)",
        "#show heading: set block(above: 1.1em, below: 0.6em)",
        "",
    ]
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out.append(_table(block))
            continue
        if not s:
            out.append("")
        elif s == "---":
            out.append("#line(length: 100%, stroke: 0.5pt + gray)")
        elif m := re.match(r"^(#{1,6})\s+(.*)$", s):
            out.append("=" * len(m.group(1)) + " " + _inline(m.group(2)))
        elif s.startswith(">") and not s.lstrip("> ").strip():
            pass  # пустая строка-разделитель внутри цитаты
        elif s.startswith(">"):
            out.append(f"#block(fill: luma(240), inset: 6pt, radius: 3pt, width: 100%)[{_inline(s.lstrip('> ').strip()) or ' '}]")
        elif m := re.match(r"^(\s*)[-*]\s+\[ \]\s+(.*)$", line):
            out.append(f"{m.group(1)}- ☐ {_inline(m.group(2))}")
        elif m := re.match(r"^(\s*)[-*]\s+(.*)$", line):
            out.append(f"{m.group(1)}- {_inline(m.group(2))}")
        elif m := re.match(r"^(\s*)\d+\.\s+(.*)$", line):
            out.append(f"{m.group(1)}+ {_inline(m.group(2))}")
        else:
            out.append(_inline(s))
        i += 1
    return "\n".join(out)


def render_pdf(markdown: str, pdf_path: Path) -> tuple[bool, str]:
    try:
        import typst
    except ImportError:
        return False, "пакет typst не установлен (pip install typst)"
    fonts = find_font()
    src = markdown_to_typst(markdown)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "report.typ"
            tpath.write_text(src, encoding="utf-8")
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            typst.compile(str(tpath), output=str(pdf_path), font_paths=[str(fonts)] if fonts else [])
        return True, ""
    except Exception as exc:  # typst сообщает об ошибках разметки исключением
        return False, f"{type(exc).__name__}: {str(exc)[:300]}"
