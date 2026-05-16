"""Redis-backed job state management and pub/sub event publisher."""
from __future__ import annotations

import json
import time
from typing import Any

import redis.asyncio as aioredis

from inference_forge.config import settings
from inference_forge.observability.logger import get_logger

logger = get_logger(__name__)

JobStatus = str  # "pending" | "processing" | "done" | "failed"


def _job_key(job_id: str) -> str:
    return f"job:{job_id}"


def _event_channel(job_id: str) -> str:
    return f"job:{job_id}:events"


def _results_key(job_id: str) -> str:
    return f"job:{job_id}:results"


class JobStore:
    """
    Manages job lifecycle in Redis using HSET/HGET and pub/sub.

    Schema (Redis hash  key=job:{job_id}):
        status:        pending | processing | done | failed
        total:         int
        completed:     int
        failed_count:  int
        cache_hits:    int
        created_at:    float (unix timestamp)
        started_at:    float | ""
        completed_at:  float | ""

    Results are stored separately in  job:{job_id}:results  as JSON to keep
    the hash small and avoid HGET overhead on the critical path.
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def create(self, job_id: str, total: int) -> None:
        key = _job_key(job_id)
        now = time.time()
        await self._redis.hset(
            key,
            mapping={
                "status": "pending",
                "total": total,
                "completed": 0,
                "failed_count": 0,
                "cache_hits": 0,
                "created_at": now,
                "started_at": "",
                "completed_at": "",
            },
        )
        await self._redis.expire(key, settings.job_ttl_seconds)
        logger.info("job_created", event="job_created", job_id=job_id, total=total)

    async def start(self, job_id: str) -> None:
        key = _job_key(job_id)
        await self._redis.hset(key, mapping={"status": "processing", "started_at": time.time()})

    async def increment_progress(
        self,
        job_id: str,
        completed: int = 0,
        failed: int = 0,
        cache_hits: int = 0,
    ) -> None:
        key = _job_key(job_id)
        pipe = self._redis.pipeline()
        if completed:
            pipe.hincrby(key, "completed", completed)
        if failed:
            pipe.hincrby(key, "failed_count", failed)
        if cache_hits:
            pipe.hincrby(key, "cache_hits", cache_hits)
        await pipe.execute()

    async def finish(self, job_id: str, results: list[dict[str, Any]]) -> None:
        key = _job_key(job_id)
        results_key = _results_key(job_id)
        now = time.time()

        pipe = self._redis.pipeline()
        pipe.hset(key, mapping={"status": "done", "completed_at": now})
        # Store results atomically in a separate key
        pipe.set(results_key, json.dumps(results, ensure_ascii=False))
        pipe.expire(results_key, settings.job_ttl_seconds)
        await pipe.execute()
        logger.info("job_finished", event="job_finished", job_id=job_id, result_count=len(results))

    async def fail(self, job_id: str, reason: str = "") -> None:
        key = _job_key(job_id)
        await self._redis.hset(
            key, mapping={"status": "failed", "completed_at": time.time(), "error": reason}
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get_state(self, job_id: str) -> dict[str, Any] | None:
        key = _job_key(job_id)
        raw = await self._redis.hgetall(key)
        if not raw:
            return None
        # Decode bytes → str (redis-py returns bytes by default)
        return {k.decode(): v.decode() for k, v in raw.items()}

    async def get_results(self, job_id: str) -> list[dict[str, Any]] | None:
        raw = await self._redis.get(_results_key(job_id))
        if raw is None:
            return None
        return json.loads(raw)

    async def exists(self, job_id: str) -> bool:
        return bool(await self._redis.exists(_job_key(job_id)))

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def publish_progress(
        self,
        job_id: str,
        completed: int,
        total: int,
        cache_hits: int,
        failed: int,
        eta_seconds: float,
    ) -> None:
        payload = json.dumps(
            {
                "completed": completed,
                "total": total,
                "cache_hits": cache_hits,
                "failed": failed,
                "eta_seconds": round(eta_seconds, 1),
            }
        )
        await self._redis.publish(_event_channel(job_id), payload)

    async def publish_done(
        self,
        job_id: str,
        results: list[dict[str, Any]],
        stats: dict[str, Any],
    ) -> None:
        payload = json.dumps({"status": "done", "results": results, "stats": stats})
        await self._redis.publish(_event_channel(job_id), payload)

    def subscribe(self, redis: aioredis.Redis, job_id: str) -> aioredis.client.PubSub:
        """Return a PubSub object already subscribed to this job's channel."""
        ps = redis.pubsub()
        return ps

    async def subscribe_async(self, redis: aioredis.Redis, job_id: str) -> aioredis.client.PubSub:
        ps = redis.pubsub()
        await ps.subscribe(_event_channel(job_id))
        return ps
