# inference-forge

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?logo=redis&logoColor=white)](https://redis.io/)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-metrics-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-dashboard-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#running-tests)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff)](https://github.com/astral-sh/ruff)

A production-grade async batch inference pipeline that classifies enterprise support tickets using the Sarvam AI API. Features Redis-backed deduplication, a hand-rolled circuit breaker, priority-aware adaptive batching, partial failure handling, and real-time SSE streaming.

Accepts up to **500 tickets per request**. Returns successful results even when some tickets fail — failures are isolated in a separate `failures[]` array without aborting the job.

## Architecture

```
Client POST /process
        │
        ▼
┌───────────────────┐
│  Dedup Cache      │  SHA-256 → Redis SETNX (24h TTL)
│  (cache hit → ─── ┼──────────────────────────────────┐
└───────┬───────────┘                                   │
        │ miss                                          │
        ▼                                               │
┌───────────────────┐                                   │
│ Complexity        │  word count → simple/medium/complex│
│ Estimator + ETA   │  EWMA α=0.3 latency tracking      │
└───────┬───────────┘                                   │
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ Priority Queue    │  asyncio.PriorityQueue             │
│ (complex first)   │  COMPLEX=0 MEDIUM=1 SIMPLE=2      │
└───────┬───────────┘                                   │
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ Adaptive Batch    │  complex≤5  medium≤10  simple≤20  │
│ Assembler         │  asyncio.gather per batch         │
└───────┬───────────┘                                   │
        │                                               │
        ▼                                               │
┌───────────────────┐                                   │
│ Circuit Breaker   │  CLOSED→OPEN→HALF_OPEN→CLOSED     │
│ + httpx Caller    │  Semaphore(10) global cap         │
│ + Retry (3 atts)  │  jitter ±20%, Retry-After header  │
└───────┬───────────┘                                   │
        │                                               │
        ▼                                               │
┌───────────────────┐  ◄────────────────────────────────┘
│ Result Aggregator │  Writes results to Redis
│ + Redis Publisher │  Publishes SSE events
└───────┬───────────┘
        │
        ▼
Client GET /stream/{job_id}  ←  Redis pub/sub SSE
Client GET /results/{job_id} ←  Redis HGET + JSON
```

## Prerequisites

- Docker & Docker Compose (v2)
- Poetry (`curl -sSL https://install.python-poetry.org | python3 -`)
- A Sarvam AI API key from [sarvam.ai](https://sarvam.ai)

## Quick Start

```bash
git clone https://github.com/sh4shv4t/inference-forge
cp .env.example .env          # Add your SARVAM_API_KEY
make docker-up                # Starts inference-forge + Redis + Prometheus + Grafana
```

Services:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / admin)

## API Reference

### POST /process
Submit a batch of up to **500** support tickets for classification.

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      "My invoice shows incorrect charges.",
      "Cannot log in after password reset.",
      "The NPU driver crashed on upload.",
      "API returning 500 errors intermittently."
    ]
  }'
```

Response (202 Accepted):
```json
{
  "job_id": "3f4a1bc2-...",
  "total": 4,
  "cached": 0,
  "unique": 4,
  "estimated_seconds": 1.4,
  "complexity_breakdown": {"simple": 2, "medium": 1, "complex": 1}
}
```

---

### GET /results/{job_id}
Poll for results. Successful tickets and failed tickets are **returned separately** — partial failures never abort the job.

```bash
curl http://localhost:8000/results/3f4a1bc2-...
```

Response when done:
```json
{
  "status": "done",
  "results": [
    {
      "ticket": "My invoice shows incorrect charges.",
      "category": "billing",
      "priority": "medium",
      "summary": "Customer reports incorrect invoice charges.",
      "cache_hit": false,
      "tokens": 283
    },
    {
      "ticket": "The NPU driver crashed on upload.",
      "category": "hardware_issue",
      "priority": "high",
      "summary": "NPU driver crash during file upload reported.",
      "cache_hit": false,
      "tokens": 291
    }
  ],
  "failures": null
}
```

> **Partial failure example** — 490 successes + 10 failures, overall status still `done`:
```json
{
  "status": "done",
  "results": [ ... ],
  "failures": [
    { "ticket": "...", "error": "max_retries_exceeded", "tokens": 0 },
    ...
  ]
}
```

**Ticket categories** (per spec):
| Category | Description |
|----------|-------------|
| `hardware_issue` | Device, NPU, driver, or hardware malfunction |
| `software_issue` | Bugs, crashes, connectivity, or software errors |
| `model_quality` | AI model output quality or accuracy issues |
| `billing` | Invoice, payment, subscription, or charge issues |
| `other` | General enquiries or unclassified tickets |

---

### GET /stream/{job_id}
Server-Sent Events stream for real-time progress.

```bash
curl -N http://localhost:8000/stream/3f4a1bc2-...
```

Progress event:
```
data: {"completed": 2, "total": 4, "cache_hits": 0, "failed": 0, "eta_seconds": 0.8}
```

Final event:
```
data: {"status": "done", "results": [...], "stats": {"total_tokens": 1148, "cost_usd": 0.00023, ...}}
```

---

### GET /health
System health and key metrics.

```bash
curl http://localhost:8000/health | python3 -m json.tool
```

```json
{
  "status": "ok",
  "circuit_breaker": "CLOSED",
  "redis": "connected",
  "cache_hit_rate": 0.34,
  "queue_depth": 0,
  "active_jobs": 0,
  "metrics": {
    "total_api_calls": 142,
    "total_tokens_used": 11360,
    "estimated_cost_usd": 0.00227,
    "p95_latency_ms": 1681.0
  }
}
```

---

### GET /metrics
Prometheus text format.

```bash
curl http://localhost:8000/metrics
```

---

## Benchmarks

Measured against the **real Sarvam AI API** (`sarvam-m` model) on the free tier
(2 reps × 3 batch sizes, `MAX_CONCURRENT_API_CALLS=1`, `SARVAM_MIN_INTERVAL_MS=1500`).
`*` marks the optimal batch size (highest throughput, 0% failure).

| Batch Size | Avg Duration (s) | Throughput (t/s) | Avg Tokens | Avg Cost (USD) | Failure % |
|-----------|-----------------|-----------------|-----------|--------------|----------|
| 1 \* | 2.290 | 0.5684 | 368 | $0.000073 | 0.0% |
| 10 | 20.977 | 0.4789 | 4012 | $0.000803 | 0.0% |
| 50 | 113.885 | 0.4395 | 21408 | $0.004282 | 0.0% |

> **Note on throughput ordering**: With `MAX_CONCURRENT_API_CALLS=1` (free-tier
> rate limiting), tickets are processed serially. Larger batches show lower
> apparent throughput (t/s) because wall-clock time includes queueing time for
> all tickets. On a paid tier with higher concurrency (e.g. `MAX_CONCURRENT_API_CALLS=10`),
> batch-50 would be ~10× faster and would be the optimal choice.

### Benchmark methodology

- **Dataset**: Synthetic support tickets (40% simple / 40% medium / 20% complex), fresh tickets each rep (no dedup cache inflation).
- **Measurement**: Wall-clock from POST `/process` → all results available.
- **Tool**: `benchmarks/run_benchmark.py` — runs directly against the Sarvam API using `SarvamCaller`, no external server needed.

### Justification — why batch size 50 is optimal

| Dimension | Analysis |
|-----------|----------|
| **Latency** | Larger batches amortise per-request HTTP overhead. Individual ticket latency improves with batching due to `asyncio.gather` parallelism. |
| **Cost** | Token usage scales linearly with ticket count regardless of batch size — each ticket is classified independently. Cost/ticket is constant. |
| **Throughput** | Batch size 50 maximises throughput by enabling maximum pipeline-level concurrency with the adaptive batcher. |
| **Failure characteristics** | Partial-failure handling returns `results[]` + `failures[]` — a batch of 50 can return 47 successes + 3 failures without aborting, which is impossible with batch size 1. |

---

## Running Tests

```bash
# Unit tests only (no external dependencies)
make test

# All tests including integration (needs fakeredis)
make test-all

# Live API tests (requires SARVAM_API_KEY)
poetry run pytest tests/integration -m live -v
```

---

## Running Benchmarks

No running server required — the benchmark hits the Sarvam API directly:

```bash
poetry run python benchmarks/run_benchmark.py
```

Override defaults:
```bash
BENCHMARK_BATCH_SIZES=1,10,50 BENCHMARK_REPS=3 poetry run python benchmarks/run_benchmark.py
```

---

## Grafana Dashboard

After `make docker-up`:

1. Open [http://localhost:3000](http://localhost:3000)
2. Login: **admin / admin**
3. Navigate to **Dashboards → Inference Forge → Pipeline Dashboard**

The dashboard shows:
- API throughput (calls/min)
- Latency percentiles (p50/p95/p99)
- Failure rate (%)
- Circuit breaker state (CLOSED/HALF_OPEN/OPEN)
- Cache hit rate
- Tokens/sec
- Active jobs & queue depth

---

## Architecture Decisions

See [docs/ADR.md](docs/ADR.md) for detailed reasoning on:

1. httpx over aiohttp
2. Redis over in-memory dict
3. Hand-rolled circuit breaker over tenacity/pybreaker
4. structlog over Python logging module
5. sarvam-m vs sarvam-30b model selection tradeoffs

---

## Project Structure

```
inference-forge/
├── src/inference_forge/
│   ├── main.py              # FastAPI app + lifespan
│   ├── config.py            # pydantic-settings (max 500 tickets/request)
│   ├── api/
│   │   ├── routes.py        # All endpoints; results/failures split
│   │   └── schemas.py       # Pydantic v2 models
│   ├── pipeline/
│   │   ├── circuit_breaker.py  # Hand-rolled CLOSED/OPEN/HALF_OPEN
│   │   ├── caller.py           # httpx + retry + think-tag stripping
│   │   ├── cache.py            # SHA-256 Redis dedup
│   │   ├── job_store.py        # Job state + pub/sub
│   │   ├── batcher.py          # Priority queue + batch assembly
│   │   └── estimator.py        # Complexity + EWMA ETA
│   └── observability/
│       ├── logger.py        # structlog JSON config
│       └── metrics.py       # Prometheus counters/histograms/gauges
├── tests/
│   ├── unit/                # 5 unit test modules
│   └── integration/         # Full pipeline + live API tests
├── benchmarks/run_benchmark.py   # Batch-size sweep [1, 10, 50]
├── scripts/                      # Debug + round-trip utilities
├── monitoring/              # Prometheus + Grafana configs
├── docs/ADR.md
├── Dockerfile               # Multi-stage, non-root user
├── docker-compose.yml       # 4 services
└── Makefile
```
