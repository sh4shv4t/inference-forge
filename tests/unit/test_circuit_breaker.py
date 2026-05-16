"""Tests for the hand-rolled CircuitBreaker state machine."""
from __future__ import annotations

import asyncio
import time

import pytest

from inference_forge.pipeline.circuit_breaker import (
    CBState,
    CircuitBreaker,
    CircuitBreakerOpenError,
)


async def _ok() -> str:
    return "ok"


async def _fail() -> None:
    raise ValueError("simulated failure")


class TestClosedState:
    async def test_initial_state_is_closed(self, cb: CircuitBreaker) -> None:
        assert cb.state == CBState.CLOSED

    async def test_success_keeps_closed(self, cb: CircuitBreaker) -> None:
        result = await cb.call(_ok)
        assert result == "ok"
        assert cb.state == CBState.CLOSED

    async def test_failure_increments_count(self, cb: CircuitBreaker) -> None:
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.failure_count == 1

    async def test_threshold_triggers_open(self, cb: CircuitBreaker) -> None:
        for _ in range(cb._failure_threshold):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CBState.OPEN

    async def test_failures_outside_window_do_not_open(self) -> None:
        cb = CircuitBreaker(failure_threshold=3, window_seconds=1, recovery_timeout=1)
        # Two failures…
        for _ in range(2):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        # …then sleep past the window
        await asyncio.sleep(1.1)
        # One more failure should NOT open (window evicted the old ones)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CBState.CLOSED


class TestOpenState:
    async def test_open_rejects_immediately(self, cb: CircuitBreaker) -> None:
        for _ in range(cb._failure_threshold):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CBState.OPEN

        with pytest.raises(CircuitBreakerOpenError):
            await cb.call(_ok)

    async def test_open_transitions_to_half_open_after_timeout(
        self, cb: CircuitBreaker
    ) -> None:
        for _ in range(cb._failure_threshold):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        assert cb.state == CBState.OPEN

        await asyncio.sleep(cb._recovery_timeout + 0.1)
        # _maybe_transition is called inside cb.call
        with pytest.raises(CircuitBreakerOpenError):
            # First call after sleep should have transitioned to HALF_OPEN
            # but the probe itself might succeed or fail
            pass
        # Force a call to trigger transition
        try:
            await cb.call(_ok)
        except Exception:
            pass
        # After timeout the state should be HALF_OPEN or CLOSED (if probe succeeded)
        assert cb.state in (CBState.HALF_OPEN, CBState.CLOSED)


class TestHalfOpenState:
    async def _force_half_open(self, cb: CircuitBreaker) -> None:
        for _ in range(cb._failure_threshold):
            with pytest.raises(ValueError):
                await cb.call(_fail)
        await asyncio.sleep(cb._recovery_timeout + 0.05)

    async def test_successful_probe_closes_circuit(self, cb: CircuitBreaker) -> None:
        await self._force_half_open(cb)
        result = await cb.call(_ok)
        assert result == "ok"
        assert cb.state == CBState.CLOSED

    async def test_failed_probe_reopens_circuit(self, cb: CircuitBreaker) -> None:
        await self._force_half_open(cb)
        with pytest.raises(ValueError):
            await cb.call(_fail)
        assert cb.state == CBState.OPEN

    async def test_failure_count_reset_on_close(self, cb: CircuitBreaker) -> None:
        await self._force_half_open(cb)
        await cb.call(_ok)
        assert cb.failure_count == 0
        assert cb.state == CBState.CLOSED
