"""Shared pytest fixtures for inference-forge tests."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock  # noqa: F401 (re-exported for test use)

import fakeredis.aioredis as fakeredis
import httpx
import pytest
import pytest_asyncio

from inference_forge.pipeline.cache import DeduplicationCache
from inference_forge.pipeline.circuit_breaker import CircuitBreaker
from inference_forge.pipeline.job_store import JobStore


# ---------------------------------------------------------------------------
# Event loop
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy():
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Redis fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def fake_redis():
    """In-memory Redis mock via fakeredis."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=False)
    yield client
    await client.aclose()


@pytest_asyncio.fixture
async def cache(fake_redis):
    return DeduplicationCache(fake_redis)


@pytest_asyncio.fixture
async def job_store(fake_redis):
    return JobStore(fake_redis)


# ---------------------------------------------------------------------------
# Circuit breaker fixture (isolated instance per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def cb():
    """Fresh CircuitBreaker with tight thresholds for fast tests."""
    return CircuitBreaker(failure_threshold=3, window_seconds=5, recovery_timeout=1)


# ---------------------------------------------------------------------------
# Mock httpx client
# ---------------------------------------------------------------------------


def make_mock_response(
    content: dict[str, Any],
    status_code: int = 200,
    tokens: int = 50,
) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": json.dumps(content)}}],
        "usage": {"total_tokens": tokens},
    }
    return httpx.Response(
        status_code=status_code,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.sarvam.ai/v1/chat/completions"),
    )


@pytest.fixture
def mock_http_client():
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def test_app(fake_redis):
    """Full FastAPI app with mocked Redis and httpx."""
    from inference_forge.main import create_app

    application = create_app()

    # Patch the lifespan's resource creation
    server = fakeredis.FakeServer()
    redis_client = fakeredis.FakeRedis(server=server, decode_responses=False)
    sub_redis_client = fakeredis.FakeRedis(server=server, decode_responses=False)

    mock_http = AsyncMock(spec=httpx.AsyncClient)
    mock_http.aclose = AsyncMock()

    from inference_forge.pipeline.cache import DeduplicationCache
    from inference_forge.pipeline.caller import SarvamCaller
    from inference_forge.pipeline.job_store import JobStore

    import asyncio as _asyncio

    semaphore = _asyncio.Semaphore(10)
    cache_inst = DeduplicationCache(redis_client)
    job_store_inst = JobStore(redis_client)
    caller_inst = SarvamCaller(mock_http, cache_inst, semaphore)

    application.state.redis = redis_client
    application.state.sub_redis = sub_redis_client
    application.state.http_client = mock_http
    application.state.cache = cache_inst
    application.state.job_store = job_store_inst
    application.state.caller = caller_inst
    application.state.semaphore = semaphore

    from httpx import AsyncClient, ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test"
    ) as client:
        yield client, mock_http
