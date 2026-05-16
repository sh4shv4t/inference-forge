"""Complexity classifier and ETA estimator with EWMA latency tracking."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence


class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


# Priority values for asyncio.PriorityQueue (lower = higher priority)
COMPLEXITY_PRIORITY: dict[Complexity, int] = {
    Complexity.COMPLEX: 0,
    Complexity.MEDIUM: 1,
    Complexity.SIMPLE: 2,
}

# Max concurrent batch size per complexity bucket
BATCH_SIZE: dict[Complexity, int] = {
    Complexity.SIMPLE: 20,
    Complexity.MEDIUM: 10,
    Complexity.COMPLEX: 5,
}


def classify(ticket: str) -> Complexity:
    """Classify a ticket by word count into simple / medium / complex."""
    word_count = len(ticket.split())
    if word_count < 15:
        return Complexity.SIMPLE
    if word_count <= 60:
        return Complexity.MEDIUM
    return Complexity.COMPLEX


def classify_many(tickets: Sequence[str]) -> dict[Complexity, list[str]]:
    """Return tickets grouped by complexity bucket."""
    groups: dict[Complexity, list[str]] = {
        Complexity.SIMPLE: [],
        Complexity.MEDIUM: [],
        Complexity.COMPLEX: [],
    }
    for t in tickets:
        groups[classify(t)].append(t)
    return groups


@dataclass
class ComplexityBreakdown:
    simple: int = 0
    medium: int = 0
    complex: int = 0

    def as_dict(self) -> dict[str, int]:
        return {"simple": self.simple, "medium": self.medium, "complex": self.complex}


def breakdown(tickets: Sequence[str]) -> ComplexityBreakdown:
    bd = ComplexityBreakdown()
    for t in tickets:
        c = classify(t)
        if c == Complexity.SIMPLE:
            bd.simple += 1
        elif c == Complexity.MEDIUM:
            bd.medium += 1
        else:
            bd.complex += 1
    return bd


@dataclass
class ETAEstimator:
    """
    Estimates remaining processing time using an Exponentially Weighted
    Moving Average (α=0.3) of observed batch durations.

    Formula
    -------
        estimated_seconds = (remaining / avg_group_size) * ewma_latency_s
    """

    alpha: float = 0.3
    default_latency_s: float = 1.2
    _ewma: float = field(default=1.2, init=False)
    _has_data: bool = field(default=False, init=False)

    def update(self, observed_latency_s: float) -> None:
        """Record a new batch latency sample and update the EWMA."""
        if not self._has_data:
            self._ewma = observed_latency_s
            self._has_data = True
        else:
            self._ewma = self.alpha * observed_latency_s + (1 - self.alpha) * self._ewma

    def estimate(self, remaining_tickets: int, bucket: Complexity | None = None) -> float:
        """Return estimated seconds to process *remaining_tickets*."""
        group_size = BATCH_SIZE[bucket] if bucket is not None else 10
        latency = self._ewma if self._has_data else self.default_latency_s
        batches = max(1, remaining_tickets / group_size)
        return batches * latency

    def estimate_initial(
        self, unique_tickets: int, breakdown_obj: ComplexityBreakdown
    ) -> float:
        """
        Estimate total processing time before any data is collected.
        Uses default_latency_s and weighted average group size.
        """
        total = unique_tickets or 1
        weighted_batches = (
            (breakdown_obj.complex / BATCH_SIZE[Complexity.COMPLEX])
            + (breakdown_obj.medium / BATCH_SIZE[Complexity.MEDIUM])
            + (breakdown_obj.simple / BATCH_SIZE[Complexity.SIMPLE])
        )
        latency = self._ewma if self._has_data else self.default_latency_s
        return weighted_batches * latency


# Module-level singleton shared across the process lifetime
eta_estimator = ETAEstimator()
