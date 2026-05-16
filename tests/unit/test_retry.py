"""Tests for retry logic, Retry-After handling, and jitter bounds."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from inference_forge.pipeline.caller import (
    FATAL_STATUS_CODES,
    RETRYABLE_STATUS_CODES,
    _call_with_retry,
    _jittered,
)
from inference_forge.pipeline.circuit_breaker import circuit_breaker, CBState


def _make_response(status: int, body: dict | None = None, headers: dict | None = None) -> httpx.Response:
    if body is None:
        if status == 200:
            body = {
                "choices": [{"message": {"content": json.dumps({"category": "billing", "priority": "low", "summary": "test"})}}],
                "usage": {"total_tokens": 10},
            }
        else:
            body = {"error": "err"}
    return httpx.Response(
        status_code=status,
        content=json.dumps(body).encode(),
        headers=headers or {"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.sarvam.ai/v1/chat/completions"),
    )


class TestJitter:
    def test_jitter_within_bounds(self) -> None:
        for _ in range(100):
            result = _jittered(1.0)
            assert 0.8 <= result <= 1.2, f"Jitter out of bounds: {result}"

    def test_jitter_nonzero_base(self) -> None:
        result = _jittered(2.0)
        assert 1.6 <= result <= 2.4


class TestRetryLogic:
    async def test_success_on_first_attempt(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        client.post = AsyncMock(return_value=_make_response(200))

        with patch("inference_forge.pipeline.caller.circuit_breaker") as mock_cb:
            mock_cb.call = AsyncMock(
                return_value=({"category": "billing", "priority": "low", "summary": "ok"}, 10)
            )
            result = await _call_with_retry(client, "test ticket")
        assert result["category"] == "billing"
        assert result.get("error") is None

    async def test_fatal_error_no_retry(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)

        for fatal_code in [400, 401, 403, 422]:
            call_count = 0

            async def mock_call(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise httpx.HTTPStatusError(
                    "fatal",
                    request=httpx.Request("POST", "https://api.sarvam.ai"),
                    response=_make_response(fatal_code),
                )

            with patch("inference_forge.pipeline.caller.circuit_breaker") as mock_cb:
                mock_cb.call = mock_call
                result = await _call_with_retry(client, "ticket")

            assert result["error"] == f"http_{fatal_code}"
            assert call_count == 1, f"Expected 1 call for {fatal_code}, got {call_count}"

    async def test_retry_after_header_respected(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        sleep_durations = []

        async def mock_sleep(duration: float) -> None:
            sleep_durations.append(duration)

        response_429 = _make_response(429, headers={"Retry-After": "3", "content-type": "application/json"})
        response_200 = _make_response(200)

        attempts = [0]

        async def mock_call(fn, client, ticket, attempt):
            attempts[0] += 1
            if attempts[0] == 1:
                raise httpx.HTTPStatusError(
                    "429",
                    request=httpx.Request("POST", "https://api.sarvam.ai"),
                    response=response_429,
                )
            return {"category": "billing", "priority": "low", "summary": "ok"}, 10

        with patch("inference_forge.pipeline.caller.circuit_breaker") as mock_cb:
            mock_cb.call = mock_call
            with patch("inference_forge.pipeline.caller.asyncio.sleep", side_effect=mock_sleep):
                result = await _call_with_retry(client, "ticket")

        assert sleep_durations[0] == pytest.approx(3.0)

    async def test_json_decode_error_triggers_retry(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)
        attempt_count = [0]

        async def mock_call(fn, client, ticket, attempt):
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise json.JSONDecodeError("bad", "", 0)
            return {"category": "other", "priority": "low", "summary": "ok"}, 5

        with patch("inference_forge.pipeline.caller.circuit_breaker") as mock_cb:
            mock_cb.call = mock_call
            with patch("inference_forge.pipeline.caller.asyncio.sleep", new_callable=AsyncMock):
                result = await _call_with_retry(client, "ticket")

        assert attempt_count[0] == 3
        assert result.get("error") is None

    async def test_max_retries_exceeded_returns_failure(self) -> None:
        client = AsyncMock(spec=httpx.AsyncClient)

        async def mock_call(fn, client, ticket, attempt):
            raise json.JSONDecodeError("bad", "", 0)

        with patch("inference_forge.pipeline.caller.circuit_breaker") as mock_cb:
            mock_cb.call = mock_call
            with patch("inference_forge.pipeline.caller.asyncio.sleep", new_callable=AsyncMock):
                result = await _call_with_retry(client, "my ticket")

        assert result["error"] == "max_retries_exceeded"
        assert result["ticket"] == "my ticket"
        assert result["category"] is None
