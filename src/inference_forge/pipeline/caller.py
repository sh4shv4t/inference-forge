"""httpx-based async Sarvam AI API caller with retry, backoff, and circuit breaker."""
from __future__ import annotations

import asyncio
import json
import random
import time
from typing import Any

import httpx

from inference_forge.config import settings
from inference_forge.observability.logger import get_logger
from inference_forge.observability.metrics import record_api_call, record_latency
from inference_forge.pipeline.cache import DeduplicationCache, ticket_hash
from inference_forge.pipeline.circuit_breaker import (
    CircuitBreakerOpenError,
    circuit_breaker,
)

logger = get_logger(__name__)

SYSTEM_PROMPT = (
    'You are a support ticket classifier. Respond ONLY in valid JSON, no markdown, '
    'no explanation: '
    '{"category": "<one of: hardware_issue|software_issue|model_quality|billing|other>", '
    '"priority": "<one of: low|medium|high|critical>", '
    '"summary": "<one sentence, max 20 words>"}'
)

# Errors that should never be retried
FATAL_STATUS_CODES = {400, 401, 403, 422}
# Errors eligible for retry
RETRYABLE_STATUS_CODES = {429, 500, 503}

_rate_limit_lock = asyncio.Lock()
_last_call_ts = 0.0


def _backoff_for_attempt(attempt: int) -> float:
    return settings.retry_backoff_base * (2 ** (attempt - 1))


def _strip_think_tags(text: str) -> str:
    """
    Remove <think>...</think> reasoning blocks emitted by some models (e.g. sarvam-m).

    Strategy:
    1. If text contains '</think>', return everything after the LAST occurrence.
    2. Otherwise remove all <think>...</think> blocks via regex.
    3. Falls back to original text if stripping produces empty string.
    """
    if "</think>" in text:
        after = text[text.rfind("</think>") + len("</think>") :].strip()
        return after if after else text.strip()
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    """
    Best-effort extraction of the first valid JSON object from a text response.

    1. Uses _strip_think_tags to isolate the actual response portion.
    2. Scans left-to-right for '{', builds balanced candidates, tries json.loads.
    3. Tries every balanced '{...}' block until one parses correctly.
    """
    cleaned = _strip_think_tags(text)
    search_text = cleaned if cleaned else text

    search_start = 0
    while True:
        start = search_text.find("{", search_start)
        if start == -1:
            break

        depth = 0
        for idx in range(start, len(search_text)):
            ch = search_text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = search_text[start : idx + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        search_start = idx + 1
                        break
        else:
            break  # for loop exhausted without finding balanced '}'

    raise json.JSONDecodeError("No valid JSON object found", text, 0)


def _jittered(base: float) -> float:
    """Apply ±20% jitter to a backoff duration."""
    jitter = random.uniform(-0.2, 0.2)
    return base * (1 + jitter)


async def _single_api_call(
    client: httpx.AsyncClient,
    ticket: str,
    attempt: int,
) -> tuple[dict[str, Any], int]:
    """
    Make one HTTP request to the Sarvam AI chat completions endpoint.

    Returns (parsed_result_dict, total_tokens).
    Raises httpx.HTTPStatusError on bad status codes (caller decides retry).
    Raises json.JSONDecodeError if model output is not valid JSON.
    """
    if settings.sarvam_mock_mode:
        if settings.sarvam_mock_latency_ms > 0:
            await asyncio.sleep(settings.sarvam_mock_latency_ms / 1000.0)
        return (
            {
                "category": "other",
                "priority": "low",
                "summary": "Mock classification response.",
            },
            0,
        )

    payload = {
        "model": settings.sarvam_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
        ],
        # 1024 tokens: sarvam-m emits a <think> block (200-600 tokens) before the
        # 50-token JSON response. 256 tokens was too small for complex tickets,
        # causing truncation before </think> and leaving unparseable output.
        "max_tokens": 1024,
        "temperature": 0.2,
        "reasoning_effort": "low",
        # Encourage the API to return strict JSON.
        "response_format": {"type": "json_object"},
    }

    # Optional global throttle to avoid upstream rate limits.
    if settings.sarvam_min_interval_ms > 0:
        min_interval = settings.sarvam_min_interval_ms / 1000.0
        async with _rate_limit_lock:
            global _last_call_ts
            now = time.monotonic()
            wait = min_interval - (now - _last_call_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            _last_call_ts = time.monotonic()

    start = time.monotonic()
    response = await client.post(
        f"{settings.sarvam_api_base}/chat/completions",
        json=payload,
        headers={"api-subscription-key": settings.sarvam_api_key},
    )
    elapsed_ms = (time.monotonic() - start) * 1000

    t_hash = ticket_hash(ticket)
    logger.info(
        "api_call",
        ticket_hash=t_hash,
        attempt=attempt,
        latency_ms=round(elapsed_ms, 2),
        status_code=response.status_code,
        model=settings.sarvam_model,
        cache_hit=False,
    )

    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    total_tokens = data.get("usage", {}).get("total_tokens", 0)

    # sarvam-m wraps answers in <think>...</think>; strip before parsing.
    content_clean = _strip_think_tags(content)
    try:
        result = json.loads(content_clean)
    except json.JSONDecodeError:
        result = _extract_json_object(content)

    record_latency(elapsed_ms)
    record_api_call(status="success", model=settings.sarvam_model, token_count=total_tokens)

    return result, total_tokens


async def _call_with_retry(
    client: httpx.AsyncClient,
    ticket: str,
) -> dict[str, Any]:
    """
    Execute a Sarvam API call with retry + exponential backoff + jitter.

    Wraps each attempt with the circuit breaker.
    """
    last_exc: Exception | None = None

    for attempt in range(1, settings.max_retries + 1):
        try:
            if settings.cb_enabled:
                result, tokens = await circuit_breaker.call(
                    _single_api_call, client, ticket, attempt
                )
            else:
                result, tokens = await _single_api_call(client, ticket, attempt)
            return {**result, "tokens": tokens}

        except CircuitBreakerOpenError as exc:
            logger.warning(
                "circuit_breaker_rejected",
                ticket_hash=ticket_hash(ticket),
                attempt=attempt,
            )
            return _failure_result(ticket, "circuit_breaker_open")

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            record_api_call(
                status=str(status_code), model=settings.sarvam_model, token_count=0
            )

            if status_code in FATAL_STATUS_CODES:
                logger.error(
                    "api_fatal_error",
                    ticket_hash=ticket_hash(ticket),
                    status_code=status_code,
                )
                return _failure_result(ticket, f"http_{status_code}")

            if status_code == 429 and attempt < settings.max_retries:
                retry_after = exc.response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else _jittered(_backoff_for_attempt(attempt))
                logger.warning(
                    "rate_limited",
                    ticket_hash=ticket_hash(ticket),
                    wait_s=round(wait, 2),
                )
                await asyncio.sleep(wait)
                last_exc = exc
                continue

            if status_code in RETRYABLE_STATUS_CODES and attempt < settings.max_retries:
                wait = _jittered(_backoff_for_attempt(attempt))
                await asyncio.sleep(wait)
                last_exc = exc
                continue

            # Final attempt or non-retryable error
            last_exc = exc

        except json.JSONDecodeError as exc:
            # Model returned malformed JSON — retry
            logger.warning(
                "malformed_json_response",
                ticket_hash=ticket_hash(ticket),
                attempt=attempt,
            )
            if attempt < settings.max_retries:
                wait = _jittered(_backoff_for_attempt(attempt))
                await asyncio.sleep(wait)
            last_exc = exc

        except Exception as exc:
            logger.exception(
                "unexpected_api_error",
                ticket_hash=ticket_hash(ticket),
                attempt=attempt,
            )
            last_exc = exc
            if attempt < settings.max_retries:
                wait = _jittered(_backoff_for_attempt(attempt))
                await asyncio.sleep(wait)

    logger.error(
        "max_retries_exceeded",
        ticket_hash=ticket_hash(ticket),
        attempts=settings.max_retries,
    )
    return _failure_result(ticket, "max_retries_exceeded")


def _failure_result(ticket: str, error: str) -> dict[str, Any]:
    return {
        "category": None,
        "priority": None,
        "summary": None,
        "error": error,
        "ticket": ticket,
        "tokens": 0,
    }


class SarvamCaller:
    """
    High-level caller that combines deduplication cache + retry + circuit breaker
    and enforces the global concurrency semaphore.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        cache: DeduplicationCache,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self._client = client
        self._cache = cache
        self._semaphore = semaphore

    async def process_ticket(self, ticket: str) -> dict[str, Any]:
        """
        Classify a single support ticket.

        1. Check dedup cache → return cached result if found.
        2. Acquire global semaphore slot.
        3. Call Sarvam API with retry logic.
        4. Populate cache with result.
        """
        cached = await self._cache.get(ticket)
        if cached is not None:
            return {**cached, "cache_hit": True}

        async with self._semaphore:
            result = await _call_with_retry(self._client, ticket)

        if result.get("error") is None:
            to_store = {k: v for k, v in result.items() if k not in ("cache_hit",)}
            await self._cache.set(ticket, to_store)

        return {**result, "cache_hit": False}

    async def process_batch(self, tickets: list[str]) -> list[dict[str, Any]]:
        """Process a batch concurrently; one ticket failing must not abort the whole batch."""
        tasks = [self.process_ticket(t) for t in tickets]
        raw = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[dict[str, Any]] = []
        for ticket, item in zip(tickets, raw, strict=True):
            if isinstance(item, BaseException):
                logger.exception(
                    "ticket_processing_failed",
                    ticket_preview=ticket[:120],
                    exc_type=type(item).__name__,
                )
                err_msg = f"{type(item).__name__}: {item}"
                out.append(_failure_result(ticket, err_msg[:500]))
            else:
                out.append(item)
        return out


def build_http_client() -> httpx.AsyncClient:
    """Build a shared httpx async client with connection pooling and timeouts."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
        http2=True,
    )
