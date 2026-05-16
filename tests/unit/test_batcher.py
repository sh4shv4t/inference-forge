"""Tests for PriorityBatcher and complexity classification."""
from __future__ import annotations

import pytest

from inference_forge.pipeline.batcher import PriorityBatcher
from inference_forge.pipeline.estimator import (
    BATCH_SIZE,
    Complexity,
    classify,
    classify_many,
)


class TestClassify:
    def test_simple_ticket(self) -> None:
        assert classify("My bill is wrong") == Complexity.SIMPLE

    def test_medium_ticket(self) -> None:
        words = " ".join(["word"] * 30)
        assert classify(words) == Complexity.MEDIUM

    def test_complex_ticket(self) -> None:
        words = " ".join(["word"] * 70)
        assert classify(words) == Complexity.COMPLEX

    def test_boundary_14_words_is_simple(self) -> None:
        assert classify(" ".join(["w"] * 14)) == Complexity.SIMPLE

    def test_boundary_15_words_is_medium(self) -> None:
        assert classify(" ".join(["w"] * 15)) == Complexity.MEDIUM

    def test_boundary_60_words_is_medium(self) -> None:
        assert classify(" ".join(["w"] * 60)) == Complexity.MEDIUM

    def test_boundary_61_words_is_complex(self) -> None:
        assert classify(" ".join(["w"] * 61)) == Complexity.COMPLEX


class TestBatchSizes:
    def test_simple_batch_size(self) -> None:
        assert BATCH_SIZE[Complexity.SIMPLE] == 20

    def test_medium_batch_size(self) -> None:
        assert BATCH_SIZE[Complexity.MEDIUM] == 10

    def test_complex_batch_size(self) -> None:
        assert BATCH_SIZE[Complexity.COMPLEX] == 5


class TestPriorityBatcher:
    async def test_complex_before_simple(self) -> None:
        batcher = PriorityBatcher()
        simple = "short"
        complex_ticket = " ".join(["word"] * 70)
        await batcher.enqueue_many([simple, complex_ticket])

        first_batch = await batcher.drain_batch()
        assert first_batch is not None
        assert first_batch.complexity == Complexity.COMPLEX

    async def test_complex_before_medium_before_simple(self) -> None:
        batcher = PriorityBatcher()
        simple = "short ticket"
        medium = " ".join(["word"] * 30)
        complex_ticket = " ".join(["word"] * 70)
        await batcher.enqueue_many([simple, medium, complex_ticket])

        batches = await batcher.drain_all_batches()
        order = [b.complexity for b in batches]
        assert order[0] == Complexity.COMPLEX
        assert Complexity.SIMPLE not in order[:1]

    async def test_batch_size_respected_simple(self) -> None:
        batcher = PriorityBatcher()
        tickets = ["short"] * 25  # 25 simple tickets
        await batcher.enqueue_many(tickets)

        batch = await batcher.drain_batch()
        assert batch is not None
        assert len(batch.tickets) == BATCH_SIZE[Complexity.SIMPLE]  # cap at 20

    async def test_batch_size_respected_complex(self) -> None:
        batcher = PriorityBatcher()
        complex_ticket = " ".join(["word"] * 70)
        tickets = [complex_ticket] * 10
        await batcher.enqueue_many(tickets)

        batch = await batcher.drain_batch()
        assert batch is not None
        assert len(batch.tickets) == BATCH_SIZE[Complexity.COMPLEX]  # cap at 5

    async def test_empty_queue_returns_none(self) -> None:
        batcher = PriorityBatcher()
        result = await batcher.drain_batch()
        assert result is None

    async def test_original_indices_preserved(self) -> None:
        batcher = PriorityBatcher()
        tickets = ["a", "b", "c"]
        await batcher.enqueue_many(tickets, start_index=10)
        batch = await batcher.drain_batch()
        assert batch is not None
        assert all(idx >= 10 for idx in batch.indices)
