"""Трекер прогресса конвейера (без зависимости от UI).

Конвейер сообщает этапы и состояние каждого вызова агента; CLI рисует это живым
индикатором (rich.live), а в тестах/без терминала — простыми строками.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

STATUS_ICON = {"wait": "·", "run": "⟳", "done": "✓", "cached": "⚡", "failed": "✗"}


@dataclass
class TaskState:
    label: str
    status: str = "wait"  # wait | run | done | cached | failed
    started: float = 0.0
    finished: float = 0.0
    note: str = ""

    @property
    def elapsed(self) -> float:
        if not self.started:
            return 0.0
        return (self.finished or time.monotonic()) - self.started


@dataclass
class Tracker:
    on_line: Callable[[str], None] | None = None  # текстовый режим (без живого индикатора)
    started: float = field(default_factory=time.monotonic)
    stage: str = ""
    tasks: dict[str, TaskState] = field(default_factory=dict)

    def _emit(self, line: str) -> None:
        if self.on_line:
            self.on_line(line)

    def set_stage(self, text: str) -> None:
        self.stage = text
        self._emit(text)

    def start(self, key: str, label: str) -> None:
        self.tasks[key] = TaskState(label=label, status="run", started=time.monotonic())
        self._emit(f"  ⟳ {label} — запрос к модели…")

    def note(self, key: str, text: str) -> None:
        if key in self.tasks:
            self.tasks[key].note = text
            self._emit(f"  … {self.tasks[key].label}: {text}")

    def finish(self, key: str, status: str = "done", note: str = "") -> None:
        t = self.tasks.get(key)
        if t is None:
            return
        t.status, t.finished = status, time.monotonic()
        if note:
            t.note = note
        word = {"done": "готово", "cached": "из кэша", "failed": "ошибка"}.get(status, status)
        self._emit(f"  {STATUS_ICON.get(status, '?')} {t.label} — {word} за {fmt_time(t.elapsed)}" + (f" ({t.note})" if t.note else ""))

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started


def fmt_time(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"
