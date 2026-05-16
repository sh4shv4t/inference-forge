"""Tests for the Redis deduplication cache."""
from __future__ import annotations

import hashlib

import pytest

from inference_forge.pipeline.cache import DeduplicationCache, ticket_hash, _ticket_key


class TestHashCorrectness:
    def test_hash_is_sha256(self) -> None:
        ticket = "My invoice is wrong"
        expected = hashlib.sha256(ticket.strip().lower().encode()).hexdigest()
        assert ticket_hash(ticket) == expected

    def test_hash_normalizes_whitespace_and_case(self) -> None:
        t1 = "  My Invoice IS Wrong  "
        t2 = "my invoice is wrong"
        assert ticket_hash(t1) == ticket_hash(t2)

    def test_different_tickets_different_hashes(self) -> None:
        assert ticket_hash("ticket one") != ticket_hash("ticket two")

    def test_key_prefix(self) -> None:
        key = _ticket_key("hello")
        assert key.startswith("cache:")


class TestCacheGetSet:
    async def test_miss_returns_none(self, cache: DeduplicationCache) -> None:
        result = await cache.get("non-existent ticket")
        assert result is None

    async def test_set_then_get(self, cache: DeduplicationCache) -> None:
        ticket = "My account is locked"
        result = {"category": "account", "priority": "high", "summary": "Account locked"}
        await cache.set(ticket, result)
        cached = await cache.get(ticket)
        assert cached == result

    async def test_case_insensitive_retrieval(self, cache: DeduplicationCache) -> None:
        ticket = "my invoice is wrong"
        result = {"category": "billing", "priority": "medium", "summary": "Invoice wrong"}
        await cache.set(ticket, result)
        # Retrieve with different casing
        cached = await cache.get("MY INVOICE IS WRONG")
        assert cached == result

    async def test_setnx_prevents_overwrite(self, cache: DeduplicationCache) -> None:
        ticket = "duplicate ticket"
        first = {"category": "billing", "priority": "low", "summary": "First"}
        second = {"category": "technical", "priority": "high", "summary": "Second"}
        await cache.set(ticket, first)
        await cache.set(ticket, second)  # should be no-op due to SETNX
        cached = await cache.get(ticket)
        assert cached == first  # first write wins

    async def test_hit_rate_tracking(self, cache: DeduplicationCache) -> None:
        ticket = "track me"
        result = {"category": "other", "priority": "low", "summary": "Test"}
        await cache.set(ticket, result)

        await cache.get("miss")  # miss
        await cache.get(ticket)  # hit

        assert cache.hit_rate == 0.5

    async def test_exists(self, cache: DeduplicationCache) -> None:
        ticket = "check existence"
        assert not await cache.exists(ticket)
        await cache.set(ticket, {"category": "other", "priority": "low", "summary": "x"})
        assert await cache.exists(ticket)
