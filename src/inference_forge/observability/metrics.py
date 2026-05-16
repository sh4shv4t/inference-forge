from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Shared registry (keeps metrics isolated from the default global registry
# so tests can create fresh instances without collision)
# ---------------------------------------------------------------------------
REGISTRY = CollectorRegistry(auto_describe=True)

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
api_calls_total = Counter(
    "inference_forge_api_calls_total",
    "Total Sarvam AI API calls made",
    labelnames=["status", "model"],
    registry=REGISTRY,
)

tokens_total = Counter(
    "inference_forge_tokens_total",
    "Total tokens consumed across all API calls",
    registry=REGISTRY,
)

cache_hits_total = Counter(
    "inference_forge_cache_hits_total",
    "Total deduplication cache hits",
    registry=REGISTRY,
)

cache_misses_total = Counter(
    "inference_forge_cache_misses_total",
    "Total deduplication cache misses",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Histograms
# ---------------------------------------------------------------------------
latency_ms = Histogram(
    "inference_forge_latency_ms",
    "Per-ticket API call latency in milliseconds",
    buckets=[100, 250, 500, 1000, 2000, 5000],
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# Gauges
# ---------------------------------------------------------------------------
circuit_breaker_state = Gauge(
    "inference_forge_circuit_breaker_state",
    "Circuit breaker state: 0=CLOSED, 1=HALF_OPEN, 2=OPEN",
    registry=REGISTRY,
)

active_jobs = Gauge(
    "inference_forge_active_jobs",
    "Number of currently active inference jobs",
    registry=REGISTRY,
)

queue_depth = Gauge(
    "inference_forge_queue_depth",
    "Current depth of the priority queue",
    registry=REGISTRY,
)

# ---------------------------------------------------------------------------
# In-process latency ring buffer (for percentile computation)
# ---------------------------------------------------------------------------
_latency_window: deque[float] = deque(maxlen=1000)

# Running totals for cost / token tracking
_total_tokens: int = 0
_total_api_calls: int = 0
_total_cache_hits: int = 0
_total_cache_misses: int = 0


def record_latency(ms: float) -> None:
    """Record a per-ticket latency sample."""
    _latency_window.append(ms)
    latency_ms.observe(ms)


def record_api_call(status: str, model: str, token_count: int) -> None:
    global _total_tokens, _total_api_calls
    api_calls_total.labels(status=status, model=model).inc()
    tokens_total.inc(token_count)
    _total_tokens += token_count
    _total_api_calls += 1


def record_cache_hit() -> None:
    global _total_cache_hits
    cache_hits_total.inc()
    _total_cache_hits += 1


def record_cache_miss() -> None:
    global _total_cache_misses
    cache_misses_total.inc()
    _total_cache_misses += 1


def get_percentiles() -> dict[str, float]:
    """Compute p50 / p95 / p99 from the in-memory latency window."""
    if not _latency_window:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.array(_latency_window)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def get_summary() -> dict:
    """Return a metrics summary dict for /health."""
    from inference_forge.config import settings  # avoid circular at module level

    cost_usd = (_total_tokens / 1000) * settings.cost_per_1k_tokens
    total_requests = _total_cache_hits + _total_cache_misses
    cache_hit_rate = _total_cache_hits / total_requests if total_requests else 0.0
    return {
        "total_api_calls": _total_api_calls,
        "total_tokens_used": _total_tokens,
        "estimated_cost_usd": round(cost_usd, 6),
        "cache_hit_rate": round(cache_hit_rate, 4),
        **get_percentiles(),
    }


def prometheus_output() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
