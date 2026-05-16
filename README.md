# inference-forge

A production-grade async batch inference pipeline that classifies support tickets using the Sarvam AI API, featuring Redis-backed deduplication, a hand-rolled circuit breaker, priority-aware semantic batching, and real-time SSE streaming.

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
│ + Retry (3 atts) │  jitter ±20%, Retry-After header  │
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
git clone https://github.com/your-org/inference-forge
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
Submit a batch of support tickets for classification.

```bash
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{
    "tickets": [
      "My invoice shows incorrect charges.",
      "Cannot log in after password reset.",
      "API returning 500 errors intermittently."
    ]
  }'
```

Response (202 Accepted):
```json
{
  "job_id": "3f4a1bc2-...",
  "total": 3,
  "cached": 0,
  "unique": 3,
  "estimated_seconds": 1.2,
  "complexity_breakdown": {"simple": 1, "medium": 1, "complex": 1}
}
```

---

### GET /stream/{job_id}
Server-Sent Events stream for real-time progress.

```bash
curl -N http://localhost:8000/stream/3f4a1bc2-...
```

Progress event:
```
data: {"completed": 2, "total": 3, "cache_hits": 0, "failed": 0, "eta_seconds": 0.4}
```

Final event:
```
data: {"status": "done", "results": [...], "stats": {"total_tokens": 126, "cost_usd": 0.0000252, ...}}
```

---

### GET /results/{job_id}
Poll for results (alternative to SSE).

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
      "tokens": 42
    }
  ]
}
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
    "p95_latency_ms": 643.0
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

Requires the server to be running (`make dev` or `make docker-up`):

```bash
make benchmark
```

The benchmark generates 200 synthetic tickets (40% simple, 40% medium, 20% complex, 25% duplicates) and sweeps concurrency levels [2, 5, 10, 15, 20].

---

## Benchmarks

> Run `make benchmark` after starting the server to populate this table.
> Results below are representative targets based on architecture design.

| Concurrency | Duration (s) | Throughput (t/s) | P95 Latency (ms) | Total Tokens | Cost (USD) | Failure % | Cache Hit % |
|-------------|-------------|-----------------|-----------------|-------------|-----------|----------|------------|
| 2           | ~120.00     | ~1.67           | ~850            | ~16,000     | $0.0032   | 0.0%     | 23.0%      |
| 5           | ~52.00      | ~3.85           | ~780            | ~16,000     | $0.0032   | 0.0%     | 23.0%      |
| 10 ★        | ~28.00      | ~7.14           | ~920            | ~16,000     | $0.0032   | 0.5%     | 23.0%      |
| 15          | ~24.00      | ~8.33           | ~1200           | ~16,000     | $0.0032   | 2.1%     | 23.0%      |
| 20          | ~22.00      | ~9.09           | ~1800           | ~16,000     | $0.0032   | 5.0%     | 23.0%      |

**★ Optimal concurrency: 10** — peak throughput before failure rate rises above 1%.

**Cache savings:** ~50 cache hits × 50 avg tokens = ~2,500 tokens saved ≈ $0.0005 per 200-ticket run.

*Run `make benchmark` against a live server to get real numbers.*

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
│   ├── config.py            # pydantic-settings
│   ├── api/
│   │   ├── routes.py        # All endpoints
│   │   └── schemas.py       # Pydantic v2 models
│   ├── pipeline/
│   │   ├── circuit_breaker.py  # Hand-rolled CLOSED/OPEN/HALF_OPEN
│   │   ├── caller.py           # httpx + retry + semaphore
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
├── benchmarks/run_benchmark.py
├── monitoring/              # Prometheus + Grafana configs
├── docs/ADR.md
├── Dockerfile               # Multi-stage, non-root user
├── docker-compose.yml       # 4 services
└── Makefile
```
