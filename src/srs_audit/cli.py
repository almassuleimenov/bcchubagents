"""CLI: ``python -m srs_audit check <input> <output>``.

Коды возврата: 0 — GO / CONDITIONAL GO, 1 — NO-GO, 2 — UNPROCESSABLE (битый файл / ошибка).
"""

from __future__ import annotations

import asyncio
import os
import sys
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import __version__

os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="SRS Audit — ИИ-агент статической проверки требований (ISO 29148 / IEEE 830 / финтех РК).",
)
console = Console(stderr=True)

EXIT = {"GO": 0, "CONDITIONAL GO": 0, "NO-GO": 1, "UNPROCESSABLE": 2}
VERDICT_STYLE = {"GO": "bold green", "CONDITIONAL GO": "bold yellow", "NO-GO": "bold red", "UNPROCESSABLE": "bold white on red"}


class Profile(str, Enum):
    kz_banking = "kz-banking"
    generic = "generic"


class Fmt(str, Enum):
    md = "md"
    json = "json"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@app.command()
def check(
    input_path: Path = typer.Argument(..., help="Файл требований: .md, .pdf, .docx, .txt"),
    output_path: Path = typer.Argument(..., help="Куда записать отчёт (например, reports/report.md)"),
    model: str = typer.Option(None, "--model", "-m", help="Модель: google:gemini-2.5-flash, openai:gpt-4o-mini, anthropic:…, ollama:…"),
    profile: Profile = typer.Option(Profile.kz_banking, "--profile", help="Профиль правил"),
    pdf: bool = typer.Option(False, "--pdf", help="Дополнительно сохранить отчёт в PDF (typst)"),
    html_highlight: bool = typer.Option(False, "--html-highlight", help="Сохранить HTML с подсветкой дефектов в исходном тексте"),
    offline_lint: bool = typer.Option(False, "--offline-lint", help="Без LLM: только детерминированные правила (0 токенов)"),
    fmt: Fmt = typer.Option(Fmt.md, "--format", "-f", help="Формат отчёта"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Не использовать кэш ответов LLM"),
    cache_path: Path = typer.Option(Path(".cache.db"), "--cache-path", help="Файл SQLite-кэша"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Минимум вывода в консоль"),
) -> None:
    """Проверить документ требований и записать отчёт."""
    from .pipeline import AuditOptions, run_audit
    from .reporting.md_renderer import render_json, render_markdown

    from .progress import Tracker

    live_ui = not quiet and console.is_terminal
    tracker = None
    if not quiet:
        # в терминале — живой индикатор; при выводе в файл/CI — обычные строки
        tracker = Tracker(on_line=None if live_ui else (lambda line: console.print(f"[cyan]›[/cyan] {line}")))

    opts = AuditOptions(
        model=model,
        profile=profile.value,
        offline=offline_lint,
        use_cache=not no_cache,
        cache_path=str(cache_path),
        tracker=tracker,
    )
    try:
        if live_ui and tracker is not None:
            from rich.live import Live

            with Live(LiveView(tracker), console=console, refresh_per_second=12, transient=False):
                report = asyncio.run(run_audit(input_path, opts))
        else:
            report = asyncio.run(run_audit(input_path, opts))
    except KeyboardInterrupt:
        console.print("[red]Прервано пользователем.[/red] Завершённые вызовы LLM сохранены в кэше — "
                      "повторный запуск продолжит с них.")
        raise typer.Exit(130) from None

    verdict = report.metrics.verdict
    try:
        body = render_json(report) if fmt == Fmt.json else render_markdown(report)
        _write(output_path, body)
        extras: list[Path] = []
        if verdict != "UNPROCESSABLE":
            if html_highlight:
                from .reporting.html_renderer import render_html

                html_path = output_path.with_suffix(".html")
                _write(html_path, render_html(report))
                extras.append(html_path)
            if pdf:
                from .reporting.pdf_renderer import render_pdf

                pdf_path = output_path.with_suffix(".pdf")
                ok, msg = render_pdf(render_markdown(report) if fmt == Fmt.json else body, pdf_path)
                if ok:
                    extras.append(pdf_path)
                else:
                    console.print(f"[yellow]PDF не создан:[/yellow] {msg}")
    except OSError as exc:
        console.print(f"[red]Не удалось записать отчёт:[/red] {exc}")
        raise typer.Exit(2) from None

    if not quiet:
        _summary(report, output_path, extras)
    raise typer.Exit(EXIT.get(verdict, 2))


class LiveView:
    """Живой индикатор: бегущий сегмент =#=#=#, общий таймер, этап и статус каждого агента."""

    WIDTH = 30
    SEGMENT = "=#=#=#=#"
    SPIN = "|/-\\"

    def __init__(self, tracker) -> None:
        self.t = tracker

    def _bar(self, frame: int) -> str:
        free = self.WIDTH - len(self.SEGMENT)
        p = frame % (2 * free)
        p = p if p <= free else 2 * free - p  # «отскок» от краёв
        seg = self.SEGMENT if frame % 2 == 0 else self.SEGMENT[::-1]  # =#=# ↔ #=#= — «вращение»
        return "·" * p + seg + "·" * (free - p)

    def __rich__(self):
        import time

        from rich.console import Group
        from rich.text import Text

        from .progress import fmt_time

        frame = int(time.monotonic() * 12)
        head = Text.assemble(
            ("[", "dim"), (self._bar(frame), "bold cyan"), ("] ", "dim"),
            (fmt_time(self.t.elapsed), "bold"), "  ", (self.t.stage or "Запуск…", "bold white"),
        )
        rows = Table(show_header=False, box=None, pad_edge=False, padding=(0, 1))
        rows.add_column(no_wrap=True)
        rows.add_column(no_wrap=True, overflow="ellipsis")
        rows.add_column(no_wrap=True)
        rows.add_column(overflow="fold")
        style = {"run": "yellow", "done": "green", "cached": "magenta", "failed": "red", "wait": "dim"}
        for task in self.t.tasks.values():
            icon = self.SPIN[frame % 4] if task.status == "run" else {"done": "✓", "cached": "⚡", "failed": "✗"}.get(task.status, "·")
            note = task.note or ("ждём ответ модели…" if task.status == "run" else "")
            rows.add_row(Text(f" {icon}", style=style[task.status]), Text(task.label), Text(fmt_time(task.elapsed), style="bold"),
                         Text(note, style="dim" if task.status != "failed" else "red"))
        return Group(head, rows) if self.t.tasks else head


def _summary(report, output_path: Path, extras: list[Path]) -> None:
    m, p, c = report.metrics, report.passport, report.citation
    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_row("Документ", p.document_title)
    t.add_row("Здоровье файла", report.health.status)
    t.add_row("Модель", p.model)
    if m.verdict != "UNPROCESSABLE":
        t.add_row("Требований", str(m.n_requirements))
        sev = " · ".join(f"{k}: {v}" for k, v in m.by_severity.items())
        t.add_row("Дефектов", f"{m.n_defects}  ({sev})")
        t.add_row("Индекс качества", f"{m.score}/100")
        if p.mode == "llm":
            t.add_row("Цитаты LLM", f"проверено {c.checked} → подтверждено {c.confirmed} → исправлено {c.corrected_via_retry} → карантин {c.quarantined}")
            t.add_row("LLM", f"{p.llm_requests} запросов, {p.input_tokens}+{p.output_tokens} токенов, кэш: {p.cache_hits}")
    t.add_row("Отчёт", str(output_path))
    for e in extras:
        t.add_row("", str(e))
    for n in p.notes:
        t.add_row("Примечание", n)
    console.print(Panel(t, title=f"[{VERDICT_STYLE[m.verdict]}]{m.verdict}[/]", subtitle=f"{p.duration_s} с", expand=False))


@app.command()
def doctor() -> None:
    """Проверить окружение: Python, ключи LLM, выбранную модель, шрифт для PDF."""
    from .llm.client import _ORIGINAL_KEYS, fallback_names, has_credentials, load_env, resolve_model_name
    from .reporting.pdf_renderer import find_font

    load_env()
    name = resolve_model_name(None)
    t = Table(title="SRS Audit · doctor", show_header=False)
    t.add_row("Python", sys.version.split()[0])
    t.add_row("Версия агента", __version__)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OLLAMA_BASE_URL"):
        v = os.getenv(k, "")
        is_set = k in _ORIGINAL_KEYS or bool(v and not v.startswith("your_"))
        t.add_row(k, "задан" if is_set else "—")
    t.add_row("Модель по умолчанию", f"{name} ({'ключ найден' if name and has_credentials(name) else 'офлайн-режим'})" if name else "— (офлайн-режим)")
    if name:
        t.add_row("Резервные модели", " → ".join(fallback_names(name)) or "—")
    try:
        import typst  # noqa: F401

        t.add_row("typst (PDF)", "установлен")
    except ImportError:
        t.add_row("typst (PDF)", "не установлен — pip install typst")
    t.add_row("Шрифт PDF", str(find_font() or "не найден"))
    console.print(t)


@app.command()
def version() -> None:
    """Показать версию агента."""
    from .versioning import agent_version

    console.print(agent_version())


def main() -> None:
    app()


if __name__ == "__main__":
    main()
