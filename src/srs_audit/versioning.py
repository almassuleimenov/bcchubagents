"""Версия агента для паспорта прогона: git-тег/коммит репозитория агента."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from . import __version__

REPO = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, timeout=5, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def agent_version() -> str:
    commit = _git("rev-parse", "--short", "HEAD")
    if not commit:
        return f"v{__version__} (git недоступен)"
    tag = _git("describe", "--tags", "--exact-match")
    # отчёты — выходные артефакты прогона, а не код агента: их правки версию агента не «пачкают»
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)reports")
    dirty = " +незакоммиченные изменения" if changed else ""
    return f"v{__version__} · {tag or 'commit'} {commit}{dirty}"
