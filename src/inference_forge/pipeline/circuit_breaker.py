"""Hand-rolled circuit breaker: CLOSED → OPEN → HALF_OPEN → CLOSED."""
from __future__ import annotations

import asyncio
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from inference_forge.config import settings
from inference_forge.observability.logger import get_logger
from inference_forge.observability.metrics import circuit_breaker_state

logger = get_logger(__name__)

T = TypeVar("T")


class CBState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is in OPEN state."""


class CircuitBreaker:
    """
    Sliding-window circuit breaker with CLOSED / OPEN / HALF_OPEN states.

    Parameters
    ----------
    failure_threshold:
        Number of failures in *window_seconds* that trigger OPEN.
    window_seconds:
        Width of the sliding failure-count window.
    recovery_timeout:
        Seconds to wait in OPEN before transitioning to HALF_OPEN.
    """

    def __init__(
        self,
        failure_threshold: int = settings.cb_failure_threshold,
        window_seconds: int = settings.cb_window_seconds,
        recovery_timeout: int = settings.cb_recovery_timeout,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._window_seconds = window_seconds
        self._recovery_timeout = recovery_timeout

        self._state = CBState.CLOSED
        self._failure_timestamps: deque[float] = deque()
        self._opened_at: float | None = None
        self._last_state_change: float = time.monotonic()
        self._probe_lock = asyncio.Lock()

        # Update Prometheus gauge
        circuit_breaker_state.set(0)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> CBState:
        return self._state

    @property
    def failure_count(self) -> int:
        self._evict_old_failures()
        return len(self._failure_timestamps)

    @property
    def last_state_change_ts(self) -> float:
        return self._last_state_change

    # ------------------------------------------------------------------
    # Core call wrapper
    # ------------------------------------------------------------------

    async def call(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Execute *coro_fn* if the circuit allows; otherwise raise CircuitBreakerOpenError."""
        await self._maybe_transition()

        if self._state == CBState.OPEN:
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN. Retry after {self._recovery_timeout}s."
            )

        if self._state == CBState.HALF_OPEN:
            return await self._half_open_probe(coro_fn, *args, **kwargs)

        # CLOSED — normal path
        return await self._execute(coro_fn, *args, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _maybe_transition(self) -> None:
        """Check if OPEN → HALF_OPEN timeout has elapsed."""
        if self._state == CBState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            if elapsed >= self._recovery_timeout:
                await self._transition(CBState.HALF_OPEN)

    async def _half_open_probe(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Allow exactly ONE probe through; all others are rejected while probing."""
        async with self._probe_lock:
            if self._state != CBState.HALF_OPEN:
                # Another coroutine already resolved the probe
                if self._state == CBState.OPEN:
                    raise CircuitBreakerOpenError("Circuit breaker re-opened during probe.")
                return await self._execute(coro_fn, *args, **kwargs)

            try:
                result = await self._execute(coro_fn, *args, **kwargs)
                await self._on_success()
                return result
            except Exception:
                await self._on_failure()
                raise

    async def _execute(
        self,
        coro_fn: Callable[..., Coroutine[Any, Any, T]],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run the coroutine and record success/failure."""
        try:
            result: T = await coro_fn(*args, **kwargs)
            await self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        if self._state == CBState.HALF_OPEN:
            self._failure_timestamps.clear()
            await self._transition(CBState.CLOSED)

    async def _on_failure(self) -> None:
        now = time.monotonic()
        self._failure_timestamps.append(now)
        self._evict_old_failures()

        if self._state == CBState.HALF_OPEN:
            await self._transition(CBState.OPEN)
            return

        if self._state == CBState.CLOSED and len(self._failure_timestamps) >= self._failure_threshold:
            await self._transition(CBState.OPEN)

    def _evict_old_failures(self) -> None:
        cutoff = time.monotonic() - self._window_seconds
        while self._failure_timestamps and self._failure_timestamps[0] < cutoff:
            self._failure_timestamps.popleft()

    async def _transition(self, new_state: CBState) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.monotonic()

        if new_state == CBState.OPEN:
            self._opened_at = time.monotonic()
            circuit_breaker_state.set(2)
        elif new_state == CBState.HALF_OPEN:
            circuit_breaker_state.set(1)
        else:  # CLOSED
            self._opened_at = None
            circuit_breaker_state.set(0)

        logger.warning(
            "circuit_breaker_transition",
            event="circuit_breaker_transition",
            from_state=old_state.value,
            to_state=new_state.value,
            failure_count=len(self._failure_timestamps),
            window_seconds=self._window_seconds,
        )


# Module-level singleton shared across the whole process
circuit_breaker = CircuitBreaker()
