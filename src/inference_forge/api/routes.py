"""FastAPI route handlers for inference-forge."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from inference_forge.api.schemas import (
    HealthMetrics,
    HealthResponse,
    ProcessRequest,
    ProcessResponse,
    ResultsResponse,
)
from inference_forge.observability.logger import get_logger
from inference_forge.observability.metrics import (
    active_jobs,
    get_summary,
    prometheus_output,
    queue_depth,
)
from inference_forge.pipeline.batcher import Batch, PriorityBatcher
from inference_forge.pipeline.cache import DeduplicationCache
from inference_forge.pipeline.circuit_breaker import circuit_breaker
from inference_forge.pipeline.estimator import (
    ComplexityBreakdown,
    breakdown,
    eta_estimator,
)
from inference_forge.pipeline.job_store import JobStore

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_active_job_count: int = 0


def _get_cache(request: Request) -> DeduplicationCache:
    return request.app.state.cache


def _get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _get_caller(request: Request):  # noqa: ANN202
    return request.app.state.caller


def _get_sub_redis(request: Request):  # noqa: ANN202
    """Return a dedicated Redis client for pub/sub subscriptions."""
    return request.app.state.sub_redis


# ---------------------------------------------------------------------------
# Background processing task
# ---------------------------------------------------------------------------


async def _process_job(
    job_id: str,
    unique_tickets: list[str],
    total_tickets: int,
    pre_cached: int,
    job_store: JobStore,
    cache: DeduplicationCache,
    caller: Any,
) -> None:
    """Background coroutine: batches and processes unique tickets, publishes events."""
    global _active_job_count

    _active_job_count += 1
    active_jobs.inc()

    job_start = time.monotonic()
    batcher = PriorityBatcher()
    await batcher.enqueue_many(unique_tickets)

    await job_store.start(job_id)

    all_results: list[dict[str, Any]] = []
    completed = pre_cached  # cache hits are already "done"
    failed_count = 0
    cache_hits_in_job = pre_cached

    try:
        while not batcher.empty:
            batch: Batch | None = await batcher.drain_batch()
            if batch is None:
                break

            batch_start = time.monotonic()

            # Process entire batch concurrently
            batch_results = await caller.process_batch(batch.tickets)
            batch_duration = time.monotonic() - batch_start

            # Update EWMA estimator with observed batch latency
            eta_estimator.update(batch_duration)

            batch_completed = 0
            batch_failed = 0
            batch_cache_hits = 0

            for ticket, result in zip(batch.tickets, batch_results):
                result["ticket"] = ticket
                all_results.append(result)
                if result.get("error"):
                    batch_failed += 1
                else:
                    batch_completed += 1
                if result.get("cache_hit"):
                    batch_cache_hits += 1

            completed += batch_completed + batch_failed
            failed_count += batch_failed
            cache_hits_in_job += batch_cache_hits

            await job_store.increment_progress(
                job_id,
                completed=batch_completed,
                failed=batch_failed,
                cache_hits=batch_cache_hits,
            )

            remaining = total_tickets - completed
            eta = eta_estimator.estimate(max(0, remaining), batch.complexity)

            await job_store.publish_progress(
                job_id=job_id,
                completed=completed,
                total=total_tickets,
                cache_hits=cache_hits_in_job,
                failed=failed_count,
                eta_seconds=eta,
            )

            logger.info(
                "batch_complete",
                job_id=job_id,
                batch_size=len(batch.tickets),
                bucket=batch.complexity.value,
                duration_ms=round(batch_duration * 1000, 2),
                failures=batch_failed,
            )

        # Build final stats
        duration = time.monotonic() - job_start
        summary = get_summary()
        total_tokens = sum(r.get("tokens", 0) for r in all_results)
        failure_rate = failed_count / total_tickets if total_tickets else 0.0
        cache_hit_rate = cache_hits_in_job / total_tickets if total_tickets else 0.0

        from inference_forge.config import settings  # local to avoid circular
        cost_usd = (total_tokens / 1000) * settings.cost_per_1k_tokens

        percentiles = get_summary()

        stats = {
            "total_tokens": total_tokens,
            "cost_usd": round(cost_usd, 6),
            "p50_ms": percentiles.get("p50_ms", 0.0),
            "p95_ms": percentiles.get("p95_ms", 0.0),
            "p99_ms": percentiles.get("p99_ms", 0.0),
            "cache_hit_rate": round(cache_hit_rate, 4),
            "failure_rate": round(failure_rate, 4),
            "duration_seconds": round(duration, 2),
        }

        await job_store.finish(job_id, all_results)
        await job_store.publish_done(job_id, all_results, stats)

        logger.info(
            "job_complete",
            job_id=job_id,
            total=total_tickets,
            completed=completed,
            failed=failed_count,
            duration_s=round(duration, 2),
            **stats,
        )

    except Exception as exc:
        logger.exception("job_failed", job_id=job_id, error=str(exc))
        await job_store.fail(job_id, reason=str(exc))
    finally:
        _active_job_count -= 1
        active_jobs.dec()


# ---------------------------------------------------------------------------
# POST /process
# ---------------------------------------------------------------------------


@router.post("/process", response_model=ProcessResponse, status_code=202)
async def process_tickets(body: ProcessRequest, request: Request) -> ProcessResponse:
    """
    Accept a batch of support tickets, deduplicate, estimate complexity, and
    kick off async processing. Returns immediately with a job_id.
    """
    cache = _get_cache(request)
    job_store = _get_job_store(request)
    caller = _get_caller(request)

    tickets = body.tickets
    job_id = str(uuid.uuid4())

    # Step 1: Deduplicate — find cache hits and unique tickets
    seen: dict[str, str] = {}  # normalized ticket → first occurrence
    unique_tickets: list[str] = []
    pre_cached_results: list[dict[str, Any]] = []
    cached_count = 0

    for ticket in tickets:
        normalized = ticket.strip().lower()
        if normalized in seen:
            # exact duplicate within request
            cached_count += 1
            continue
        seen[normalized] = ticket

        cached = await cache.get(ticket)
        if cached is not None:
            cached["ticket"] = ticket
            cached["cache_hit"] = True
            pre_cached_results.append(cached)
            cached_count += 1
        else:
            unique_tickets.append(ticket)

    total = len(tickets)
    unique = len(unique_tickets)

    # Step 2: Complexity breakdown
    bd: ComplexityBreakdown = breakdown(unique_tickets)

    # Step 3: ETA estimate
    estimated_seconds = eta_estimator.estimate_initial(unique, bd)

    # Step 4: Create job in Redis
    await job_store.create(job_id, total)
    # Mark pre-cached tickets as already completed
    if cached_count:
        await job_store.increment_progress(job_id, completed=cached_count, cache_hits=cached_count)

    # Step 5: Launch background task
    asyncio.create_task(
        _process_job(
            job_id=job_id,
            unique_tickets=unique_tickets,
            total_tickets=total,
            pre_cached=cached_count,
            job_store=job_store,
            cache=cache,
            caller=caller,
        )
    )

    queue_depth.set(unique)
    logger.info(
        "job_submitted",
        job_id=job_id,
        total=total,
        cached=cached_count,
        unique=unique,
        estimated_s=round(estimated_seconds, 1),
    )

    return ProcessResponse(
        job_id=job_id,
        total=total,
        cached=cached_count,
        unique=unique,
        estimated_seconds=round(estimated_seconds, 1),
        complexity_breakdown=bd.as_dict(),
    )


# ---------------------------------------------------------------------------
# GET /stream/{job_id}
# ---------------------------------------------------------------------------


async def _sse_generator(
    job_id: str,
    job_store: JobStore,
    sub_redis: Any,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events from Redis pub/sub."""
    ps = await job_store.subscribe_async(sub_redis, job_id)

    try:
        async for message in ps.listen():
            if message["type"] != "message":
                continue
            data = message["data"]
            if isinstance(data, bytes):
                data = data.decode()

            yield f"data: {data}\n\n"

            # Stop streaming once the done event is published
            try:
                parsed = json.loads(data)
                if parsed.get("status") == "done":
                    break
            except json.JSONDecodeError:
                pass
    finally:
        await ps.unsubscribe()
        await ps.aclose()


@router.get("/stream/{job_id}")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """SSE stream of job progress events."""
    job_store = _get_job_store(request)
    sub_redis = _get_sub_redis(request)

    if not await job_store.exists(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    return StreamingResponse(
        _sse_generator(job_id, job_store, sub_redis),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# GET /results/{job_id}
# ---------------------------------------------------------------------------


@router.get("/results/{job_id}", response_model=ResultsResponse)
async def get_results(job_id: str, request: Request) -> ResultsResponse:
    """Return full results if done, or progress info if still processing."""
    job_store = _get_job_store(request)
    state = await job_store.get_state(job_id)

    if state is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    status = state.get("status", "pending")

    if status == "done":
        results = await job_store.get_results(job_id)
        return ResultsResponse(status="done", results=results)

    total = int(state.get("total", 1))
    completed = int(state.get("completed", 0))
    progress = round(completed / total, 4) if total else 0.0

    return ResultsResponse(status=status, progress=progress)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return system health, circuit breaker state, and key metrics."""
    job_store = _get_job_store(request)

    # Redis ping
    redis_status = "connected"
    try:
        await request.app.state.redis.ping()
    except Exception:
        redis_status = "disconnected"

    summary = get_summary()
    cb_state = circuit_breaker.state.value

    return HealthResponse(
        status="ok",
        circuit_breaker=cb_state,
        redis=redis_status,
        cache_hit_rate=summary.get("cache_hit_rate", 0.0),
        queue_depth=int(queue_depth._value.get() if hasattr(queue_depth, "_value") else 0),
        active_jobs=_active_job_count,
        metrics=HealthMetrics(
            total_api_calls=summary["total_api_calls"],
            total_tokens_used=summary["total_tokens_used"],
            estimated_cost_usd=summary["estimated_cost_usd"],
            p95_latency_ms=summary.get("p95_ms", 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


@router.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus text-format metrics."""
    body, content_type = prometheus_output()
    return Response(content=body, media_type=content_type)
