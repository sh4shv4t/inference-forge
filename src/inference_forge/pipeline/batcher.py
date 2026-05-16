"""Priority queue and adaptive batch assembler."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from inference_forge.observability.logger import get_logger
from inference_forge.observability.metrics import queue_depth
from inference_forge.pipeline.estimator import (
    BATCH_SIZE,
    COMPLEXITY_PRIORITY,
    Complexity,
    classify,
)

logger = get_logger(__name__)


@dataclass(order=True)
class PrioritizedTicket:
    """Wrapper that makes (ticket, metadata) sortable by complexity priority."""

    priority: int
    ticket: str = field(compare=False)
    complexity: Complexity = field(compare=False)
    original_index: int = field(compare=False, default=0)
    extra: dict[str, Any] = field(compare=False, default_factory=dict)


@dataclass
class Batch:
    tickets: list[str]
    complexity: Complexity
    indices: list[int]


class PriorityBatcher:
    """
    Maintains an asyncio.PriorityQueue of tickets, drains them into
    complexity-homogeneous batches respecting the per-bucket group size.

    Processing order: COMPLEX → MEDIUM → SIMPLE
    """

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[PrioritizedTicket] = asyncio.PriorityQueue()

    async def enqueue_many(self, tickets: list[str], start_index: int = 0) -> None:
        """Enqueue tickets with their complexity-derived priority."""
        for i, ticket in enumerate(tickets):
            complexity = classify(ticket)
            priority = COMPLEXITY_PRIORITY[complexity]
            item = PrioritizedTicket(
                priority=priority,
                ticket=ticket,
                complexity=complexity,
                original_index=start_index + i,
            )
            await self._queue.put(item)

        queue_depth.set(self._queue.qsize())
        logger.debug(
            "tickets_enqueued",
            event="tickets_enqueued",
            count=len(tickets),
            queue_size=self._queue.qsize(),
        )

    async def drain_batch(self) -> Batch | None:
        """
        Drain up to BATCH_SIZE items of the same complexity from the front
        of the queue. Returns None if queue is empty.

        Because PriorityQueue is ordered, items of the same priority cluster
        together; we peek at the top and collect same-complexity items greedily.
        """
        if self._queue.empty():
            return None

        first: PrioritizedTicket = await self._queue.get()
        complexity = first.complexity
        max_size = BATCH_SIZE[complexity]

        tickets = [first.ticket]
        indices = [first.original_index]

        # Collect up to max_size - 1 more same-complexity items (non-blocking)
        while len(tickets) < max_size and not self._queue.empty():
            try:
                # Peek: get_nowait then put back if different complexity
                candidate = self._queue.get_nowait()
                if candidate.complexity == complexity:
                    tickets.append(candidate.ticket)
                    indices.append(candidate.original_index)
                else:
                    # Different complexity — put it back and stop
                    await self._queue.put(candidate)
                    break
            except asyncio.QueueEmpty:
                break

        queue_depth.set(self._queue.qsize())
        return Batch(tickets=tickets, complexity=complexity, indices=indices)

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    async def drain_all_batches(self) -> list[Batch]:
        """Drain the entire queue into a list of batches (for testing)."""
        batches: list[Batch] = []
        while not self.empty:
            batch = await self.drain_batch()
            if batch:
                batches.append(batch)
        return batches
