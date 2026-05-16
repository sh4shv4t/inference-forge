"""Tests for the ETA estimator and EWMA latency tracking."""
from __future__ import annotations

import pytest

from inference_forge.pipeline.estimator import (
    ComplexityBreakdown,
    ETAEstimator,
    breakdown,
    classify,
    Complexity,
)


class TestETAEstimator:
    def test_initial_estimate_uses_default_latency(self) -> None:
        est = ETAEstimator(default_latency_s=1.2)
        # 10 simple tickets → ceil(10/20) = 1 batch × 1.2s = 1.2s
        result = est.estimate(10, Complexity.SIMPLE)
        # estimate uses max(1, remaining/group_size) batches
        assert result == pytest.approx(1.2, rel=0.01)

    def test_ewma_updates_on_first_observation(self) -> None:
        est = ETAEstimator()
        est.update(2.0)
        assert est._ewma == pytest.approx(2.0)

    def test_ewma_converges(self) -> None:
        est = ETAEstimator(alpha=0.3)
        for _ in range(50):
            est.update(3.0)
        assert est._ewma == pytest.approx(3.0, abs=0.01)

    def test_ewma_weighted_update(self) -> None:
        est = ETAEstimator(alpha=0.3, default_latency_s=1.0)
        est.update(1.0)  # sets _ewma = 1.0
        est.update(4.0)  # 0.3 * 4 + 0.7 * 1 = 1.9
        assert est._ewma == pytest.approx(1.9, abs=0.001)

    def test_estimate_after_update(self) -> None:
        est = ETAEstimator(alpha=0.3)
        est.update(2.0)
        # 5 complex tickets → 5/5 = 1 batch × 2.0s
        result = est.estimate(5, Complexity.COMPLEX)
        assert result == pytest.approx(2.0, rel=0.01)

    def test_estimate_initial_zero_tickets(self) -> None:
        est = ETAEstimator()
        bd = ComplexityBreakdown(simple=0, medium=0, complex=0)
        result = est.estimate_initial(0, bd)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_estimate_initial_mixed_breakdown(self) -> None:
        est = ETAEstimator(default_latency_s=1.0)
        # 20 simple, 10 medium, 5 complex
        # weighted_batches = 20/20 + 10/10 + 5/5 = 1 + 1 + 1 = 3
        bd = ComplexityBreakdown(simple=20, medium=10, complex=5)
        result = est.estimate_initial(35, bd)
        assert result == pytest.approx(3.0, rel=0.01)


class TestBreakdown:
    def test_breakdown_counts(self) -> None:
        tickets = (
            ["short"] * 5
            + [" ".join(["w"] * 30)] * 3
            + [" ".join(["w"] * 70)] * 2
        )
        bd = breakdown(tickets)
        assert bd.simple == 5
        assert bd.medium == 3
        assert bd.complex == 2

    def test_breakdown_as_dict(self) -> None:
        bd = ComplexityBreakdown(simple=1, medium=2, complex=3)
        d = bd.as_dict()
        assert d == {"simple": 1, "medium": 2, "complex": 3}
