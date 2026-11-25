"""Фабрика моделей, лимитер Free Tier, ретраи с Retry-After и SQLite-кэш ответов.

* Провайдеры: ``google:`` (Gemini API), ``openai:``, ``anthropic:``, ``ollama:``.
* ``FallbackModel``: основная модель + запасные (``SRS_FALLBACK_MODELS``) — при 429/квоте/
  недоступности запрос автоматически уходит в следующую модель.
* ``aiolimiter.AsyncLimiter(8, 60)``: не более 8 запросов в минуту (ниже лимита Free Tier).
* ``tenacity``: экспоненциальный backoff, уважающий ``Retry-After``/``retryDelay`` провайдера.
* Кэш на встроенном ``sqlite3`` (ключ SHA-256 от модели, агента, версии промпта и входа):
  повторный прогон того же документа воспроизводим и стоит 0 токенов.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiolimiter import AsyncLimiter
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay

# Замер 25.11.2026 на docs/requirements.md (экстрактор): gemini-2.5-flash — 38 с, 52/52 требований,
# 0 ретраев цитат; gemini-flash-latest — 503 «high demand»; gemini-flash-lite-latest — 6 с, но 16 требований.
DEFAULT_GEMINI = "gemini-2.5-flash"
# Цепочка резервов: при 503/429/таймауте FallbackModel мгновенно переключается на следующую модель
GEMINI_CHAIN = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]
DEFAULT_OPENAI = "gpt-4o-mini"
DEFAULT_ANTHROPIC = "claude-haiku-4-5"
REQUEST_TIMEOUT_S = 180.0  # таймаут одного HTTP-запроса к модели
AGENT_RETRY_BUDGET_S = 300.0  # общий бюджет повторов одного агента
MAX_ATTEMPTS = 3
_ORIGINAL_KEYS: set[str] = set()


# ---------------------------------------------------------------------------
# Окружение и выбор модели
# ---------------------------------------------------------------------------


def load_env(env_file: str | Path | None = None) -> None:
    """Читает .env и приводит переменные к ожиданиям провайдеров."""
    from dotenv import load_dotenv

    load_dotenv(env_file or Path.cwd() / ".env", override=False)
    _ORIGINAL_KEYS.update(k for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY") if _key(k))
    # google-genai ругается «Both GOOGLE_API_KEY and GEMINI_API_KEY are set», если заданы оба:
    # оставляем один GOOGLE_API_KEY (изменение только в памяти процесса, .env не трогаем)
    gem = _key("GEMINI_API_KEY")
    if gem:
        if not _key("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = gem
        os.environ.pop("GEMINI_API_KEY", None)
    base = os.getenv("OLLAMA_BASE_URL", "").strip()
    if base and not base.rstrip("/").endswith("/v1"):
        os.environ["OLLAMA_BASE_URL"] = base.rstrip("/") + "/v1"
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")


def _placeholder(value: str) -> bool:
    v = value.strip().lower()
    return not v or v.startswith("your_") or v in {"changeme", "xxx", "none", "null"}


def _key(name: str) -> str:
    v = os.getenv(name, "").strip()
    return "" if _placeholder(v) else v


def normalize_model_name(name: str) -> str:
    """Принимает «gemini-2.5-flash», «google-gla:…», «gpt-4o-mini», «claude-…» и т.п."""
    name = name.strip()
    if name.startswith(("google-gla:", "gemini:")):
        return "google:" + name.split(":", 1)[1]
    if ":" in name:
        return name
    low = name.lower()
    if low.startswith("gemini") or low.startswith("gemma"):
        return f"google:{name}"
    if low.startswith(("gpt", "o1", "o3", "o4")):
        return f"openai:{name}"
    if low.startswith("claude"):
        return f"anthropic:{name}"
    return f"ollama:{name}"


def resolve_model_name(cli_model: str | None = None) -> str | None:
    """CLI → SRS_MODEL → ключи провайдеров из .env. None = ключей нет (офлайн-режим)."""
    if cli_model:
        return normalize_model_name(cli_model)
    if os.getenv("SRS_MODEL"):
        return normalize_model_name(os.environ["SRS_MODEL"])
    if _key("GOOGLE_API_KEY") or _key("GEMINI_API_KEY"):
        return normalize_model_name(os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI)
    if _key("OPENAI_API_KEY"):
        return normalize_model_name(os.getenv("OPENAI_MODEL") or DEFAULT_OPENAI)
    if _key("ANTHROPIC_API_KEY"):
        return normalize_model_name(os.getenv("ANTHROPIC_MODEL") or DEFAULT_ANTHROPIC)
    if os.getenv("OLLAMA_MODEL"):
        return "ollama:" + os.environ["OLLAMA_MODEL"]
    return None


def has_credentials(model_name: str) -> bool:
    provider = model_name.split(":", 1)[0]
    if provider in ("google", "google-gla"):
        return bool(_key("GOOGLE_API_KEY") or _key("GEMINI_API_KEY"))
    if provider in ("openai", "openai-chat", "openai-responses"):
        return bool(_key("OPENAI_API_KEY"))
    if provider == "anthropic":
        return bool(_key("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        return bool(os.getenv("OLLAMA_BASE_URL"))
    if provider in ("test", "function"):
        return True
    return True  # прочие провайдеры pydantic-ai сами проверят окружение


def fallback_names(primary: str) -> list[str]:
    env = os.getenv("SRS_FALLBACK_MODELS", "").strip()
    if env:
        names = [normalize_model_name(x) for x in env.split(",") if x.strip()]
    elif primary.startswith("google:"):
        names = [f"google:{m}" for m in GEMINI_CHAIN]
    else:
        names = []
    return [n for n in names if n != primary and has_credentials(n)]


def build_model(primary: str, *, use_fallback: bool = True):
    """Создаёт pydantic-ai Model (с FallbackModel, если заданы запасные модели)."""
    from pydantic_ai.models import infer_model

    main = infer_model(primary)
    if not use_fallback:
        return main
    extras = fallback_names(primary)
    if not extras:
        return main
    from pydantic_ai.models.fallback import FallbackModel

    return FallbackModel(main, *[infer_model(n) for n in extras])


# ---------------------------------------------------------------------------
# SQLite-кэш
# ---------------------------------------------------------------------------


class SQLiteCache:
    """Простой потокобезопасный JSON-кэш на встроенном sqlite3 (без pickle — безопасно)."""

    def __init__(self, path: str | Path = ".cache.db", enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = str(path)
        self._lock = threading.Lock()
        self.hits = 0
        if enabled:
            with self._conn() as c:
                c.execute(
                    "CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, created REAL, agent TEXT, value TEXT)"
                )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    @staticmethod
    def make_key(*parts: Any) -> str:
        blob = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock, self._conn() as c:
            row = c.execute("SELECT value FROM llm_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            self.hits += 1
            return json.loads(row[0])
        except json.JSONDecodeError:
            return None

    def set(self, key: str, agent: str, value: dict) -> None:
        if not self.enabled:
            return
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO llm_cache (key, created, agent, value) VALUES (?, ?, ?, ?)",
                (key, time.time(), agent, json.dumps(value, ensure_ascii=False)),
            )


# ---------------------------------------------------------------------------
# Ретраи и лимиты
# ---------------------------------------------------------------------------

RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _is_retryable(exc: BaseException) -> bool:
    from pydantic_ai import exceptions as E

    if isinstance(exc, (E.ModelQuotaExceededError, E.ModelAuthenticationError, E.ModelNotFoundError)):
        return False
    if isinstance(exc, (E.ModelRateLimitError, E.ModelServerError, E.ModelUnavailableError, E.ModelConnectionError)):
        return True
    status = getattr(exc, "status_code", None)
    if status in RETRYABLE_STATUS:
        return True
    from pydantic_ai.exceptions import FallbackExceptionGroup

    if isinstance(exc, FallbackExceptionGroup):
        return any(_is_retryable(e) for e in exc.exceptions)
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Retry-After из исключения/заголовков/тела Gemini («retryDelay": "37s")."""
    ra = getattr(exc, "retry_after", None)
    if isinstance(ra, (int, float)) and ra > 0:
        return float(ra)
    headers = getattr(exc, "headers", None) or {}
    try:
        h = {k.lower(): v for k, v in dict(headers).items()}
        if "retry-after" in h:
            return float(h["retry-after"])
    except (TypeError, ValueError):
        pass
    body = getattr(exc, "body", None)
    if body is not None:
        m = re.search(r'retryDelay"?\s*[:=]\s*"?(\d+(?:\.\d+)?)s', json.dumps(body, default=str))
        if m:
            return float(m.group(1))
    return None


def _wait(retry_state) -> float:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    ra = retry_after_seconds(exc) if exc else None
    if ra:
        return min(ra + 1.0, 60.0)
    return min(5.0 * (2 ** (retry_state.attempt_number - 1)), 30.0)


def describe_error(exc: BaseException) -> str:
    """Короткое человекочитаемое описание ошибки провайдера для индикатора прогресса."""
    from pydantic_ai import exceptions as E
    from pydantic_ai.exceptions import FallbackExceptionGroup

    if isinstance(exc, FallbackExceptionGroup):
        return "все модели цепочки недоступны: " + "; ".join(describe_error(e) for e in exc.exceptions[:4])
    status = getattr(exc, "status_code", None)
    model = getattr(exc, "model_name", "") or ""
    prefix = f"{model}: " if model else ""
    if isinstance(exc, E.ModelTimeoutError) or isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return f"{prefix}таймаут ответа модели"
    if isinstance(exc, E.ModelQuotaExceededError):
        return f"{prefix}429 — исчерпана дневная квота"
    if status == 503 or isinstance(exc, E.ModelUnavailableError):
        return f"{prefix}503 — модель перегружена"
    if status == 429 or isinstance(exc, E.ModelRateLimitError):
        return f"{prefix}429 — лимит запросов в минуту"
    if status:
        return f"{prefix}HTTP {status}"
    return f"{prefix}{type(exc).__name__}"


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    per_agent: dict[str, int] = field(default_factory=dict)

    def add(self, agent: str, usage: Any) -> None:
        if usage is None:
            return
        self.requests += int(getattr(usage, "requests", 0) or 0)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.per_agent[agent] = self.per_agent.get(agent, 0) + int(getattr(usage, "requests", 0) or 0)


def make_rate_limited(model: Any, limiter: AsyncLimiter, on_response=None):
    """Оборачивает модель: КАЖДЫЙ HTTP-запрос (включая повторы после ModelRetry) идёт через лимитер."""
    from pydantic_ai.models.wrapper import WrapperModel

    class RateLimitedModel(WrapperModel):
        async def request(self, messages, model_settings, model_request_parameters):  # type: ignore[override]
            async with limiter:
                response = await super().request(messages, model_settings, model_request_parameters)
            if on_response is not None:
                on_response(response)
            return response

    return RateLimitedModel(model)


class LLMRuntime:
    """Общий контекст вызовов: модель, лимитер, кэш, учёт токенов."""

    def __init__(
        self,
        model: Any,
        model_name: str,
        *,
        cache: SQLiteCache | None = None,
        rpm: int | None = None,
        concurrency: int = 4,
        max_attempts: int = MAX_ATTEMPTS,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.cache = cache or SQLiteCache(enabled=False)
        rpm = rpm or int(os.getenv("SRS_RPM", "8"))
        self.limiter = AsyncLimiter(rpm, 60)
        self.sem = asyncio.Semaphore(concurrency)
        self.max_attempts = max_attempts
        self.usage = UsageTotals()
        self.served_models: set[str] = set()
        self.model = make_rate_limited(model, self.limiter, self._on_response)

    def _on_response(self, response: Any) -> None:
        name = getattr(response, "model_name", None)
        if name:
            self.served_models.add(str(name))

    async def call(self, fn, *args, on_retry=None, **kwargs):
        """Выполняет корутину-фабрику ``fn`` под семафором и ретраями (лимитер — внутри модели).

        ``on_retry(text)`` получает причину и паузу перед повтором — для индикатора прогресса.
        """

        def before_sleep(rs) -> None:
            if on_retry is not None and rs.outcome is not None:
                exc = rs.outcome.exception()
                wait = rs.next_action.sleep if rs.next_action else 0
                on_retry(f"{describe_error(exc)}; повтор {rs.attempt_number + 1}/{self.max_attempts} через {wait:.0f} с")

        async for attempt in AsyncRetrying(
            retry=retry_if_exception(_is_retryable),
            wait=_wait,
            stop=stop_after_attempt(self.max_attempts) | stop_after_delay(AGENT_RETRY_BUDGET_S),
            before_sleep=before_sleep,
            reraise=True,
        ):
            with attempt:
                async with self.sem:
                    return await fn(*args, **kwargs)
        raise RuntimeError("unreachable")  # pragma: no cover
