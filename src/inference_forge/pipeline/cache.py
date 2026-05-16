"""Redis-backed SHA-256 deduplication cache for inference results."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from inference_forge.config import settings
from inference_forge.observability.logger import get_logger
from inference_forge.observability.metrics import record_cache_hit, record_cache_miss

logger = get_logger(__name__)

_PROCESSED_COUNT = 0
_LOG_EVERY_N = 50


def _ticket_key(ticket: str) -> str:
    """Return the Redis cache key for a given ticket."""
    digest = hashlib.sha256(ticket.strip().lower().encode()).hexdigest()
    return f"cache:{digest}"


def ticket_hash(ticket: str) -> str:
    """Return only the SHA-256 hex digest (for logging / dedup checks)."""
    return hashlib.sha256(ticket.strip().lower().encode()).hexdigest()


class DeduplicationCache:
    """
    Wraps a redis.asyncio.Redis client with SETNX-based dedup logic.

    Usage
    -----
        cache = DeduplicationCache(redis_client)
        result = await cache.get(ticket)     # None on miss
        await cache.set(ticket, result_dict) # stores with TTL
    """

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._hits = 0
        self._misses = 0

    async def get(self, ticket: str) -> dict[str, Any] | None:
        """Return cached result or None on cache miss."""
        global _PROCESSED_COUNT

        key = _ticket_key(ticket)
        raw = await self._redis.get(key)

        if raw is not None:
            self._hits += 1
            record_cache_hit()
            logger.debug("cache_hit", key=key)
            _PROCESSED_COUNT += 1
            self._maybe_log_rate()
            return json.loads(raw)

        self._misses += 1
        record_cache_miss()
        _PROCESSED_COUNT += 1
        self._maybe_log_rate()
        return None

    async def set(self, ticket: str, result: dict[str, Any]) -> None:
        """Store result in Redis with SETNX semantics and 24h TTL."""
        key = _ticket_key(ticket)
        value = json.dumps(result, ensure_ascii=False)
        # SETNX: only set if key does not exist (prevents race conditions)
        await self._redis.set(key, value, nx=True, ex=settings.dedup_ttl_seconds)

    async def exists(self, ticket: str) -> bool:
        """Return True if a cached result exists for *ticket*."""
        return bool(await self._redis.exists(_ticket_key(ticket)))

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total else 0.0

    def _maybe_log_rate(self) -> None:
        global _PROCESSED_COUNT
        if _PROCESSED_COUNT % _LOG_EVERY_N == 0:
            logger.info(
                "cache_hit_rate",
                processed=_PROCESSED_COUNT,
                hit_rate=round(self.hit_rate, 4),
                hits=self._hits,
                misses=self._misses,
            )
