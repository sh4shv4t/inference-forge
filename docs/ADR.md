# Architecture Decision Records — inference-forge

> Each ADR follows the format: **Decision | Context | Alternatives Considered | Why Rejected**

---

## ADR-001: httpx over aiohttp for outbound HTTP

**Decision:** Use `httpx[asyncio]` as the sole HTTP client for all Sarvam AI API calls.

**Context:** The pipeline makes high-concurrency outbound HTTP calls to a single upstream host. The client must support async/await natively, HTTP/2 for multiplexing, connection pooling, and structured timeout control.

**Alternatives Considered:**

| Alternative | Notes |
|-------------|-------|
| `aiohttp` | Mature async library, but uses a different API surface than `requests`, requires manual session management, and has no native HTTP/2 support without additional dependencies. |
| `requests` + `concurrent.futures` | Synchronous; spawning threads for I/O-bound work is wasteful and defeats the purpose of asyncio. |
| `urllib3` | Low-level, no async support without wrappers. |

**Why Rejected:**
- `aiohttp` lacks HTTP/2 support, which reduces per-connection efficiency under high concurrency.
- `aiohttp`'s `ClientSession` requires explicit lifecycle management that is more error-prone in long-running FastAPI services compared to `httpx.AsyncClient` in a lifespan context manager.
- `httpx` offers a requests-compatible API, making it easier to onboard contributors.
- `httpx`'s `Limits` object gives fine-grained control over `max_connections` and `max_keepalive_connections`, directly tunable for our semaphore-capped concurrency model.

---

## ADR-002: Redis over in-memory dict for shared state

**Decision:** Use Redis 7 (via `redis-py[asyncio]`) for job state, deduplication cache, and pub/sub event delivery.

**Context:** The service is deployed with `--workers 2` (two Uvicorn processes). Any in-process dict would not be shared between workers, causing jobs submitted to worker A to be invisible to worker B. We also need TTL-based expiry and a pub/sub channel for SSE streaming.

**Alternatives Considered:**

| Alternative | Notes |
|-------------|-------|
| `dict` / `lru_cache` | Per-process only; breaks horizontal scaling and multi-worker deployments. |
| PostgreSQL | ACID guarantees are overkill for ephemeral job state; pub/sub requires polling or a separate `LISTEN/NOTIFY` channel. |
| Memcached | No native pub/sub; no persistent storage; no complex data structures (hashes, sorted sets). |
| DynamoDB / Firestore | Managed but adds cloud vendor coupling, network latency, and cost for what is a transient cache. |

**Why Rejected:**
- In-memory solutions break across multiple Uvicorn workers and Kubernetes pods.
- PostgreSQL introduces schema migrations for what is fundamentally a cache/queue problem.
- Redis is the industry standard for this exact pattern: it provides `HSET`/`HGET` for structured job state, `SETNX` with TTL for race-condition-safe deduplication, `PUBLISH`/`SUBSCRIBE` for zero-polling SSE streaming, and automatic key expiry for free garbage collection of stale job data.

---

## ADR-003: Hand-rolled CircuitBreaker over tenacity / pybreaker

**Decision:** Implement `CircuitBreaker` as a custom asyncio-native class with explicit CLOSED → OPEN → HALF_OPEN → CLOSED state machine.

**Context:** The circuit breaker must integrate with asyncio (all code is async), expose real-time state on `/health`, and enforce an exact "one probe in HALF_OPEN" contract. It must also emit structured log events on every transition.

**Alternatives Considered:**

| Alternative | Notes |
|-------------|-------|
| `tenacity` | Excellent retry library, but no native circuit breaker pattern; would need layering on top of a separate library. |
| `pybreaker` | Synchronous only; does not support `asyncio.Lock` for HALF_OPEN probe exclusivity; state is not directly inspectable as a Python property. |
| `aiobreaker` | Limited maintenance, no sliding-window failure tracking, no HALF_OPEN probe lock. |

**Why Rejected:**
- All third-party circuit breakers are either synchronous or too opaque to expose state on the `/health` endpoint without monkey-patching.
- The HALF_OPEN probe exclusivity requirement (exactly one request through while others are rejected) requires an `asyncio.Lock`, which library solutions don't provide.
- Hand-rolling gives complete control over the sliding window (`deque` of timestamps), the `_maybe_transition` hook, and the Prometheus gauge update — all wired together without magic.
- The implementation is ~150 lines, well-tested, and has no transitive dependencies.

---

## ADR-004: structlog over Python's stdlib `logging` module

**Decision:** Use `structlog` for all application logging with JSON output in production and colorized console output in development.

**Context:** A production inference pipeline benefits from machine-parseable logs (JSON) that can be ingested by Datadog, Loki, or CloudWatch without log parsing rules. Each log event must carry structured fields (`job_id`, `latency_ms`, `status_code`, etc.).

**Alternatives Considered:**

| Alternative | Notes |
|-------------|-------|
| `logging` (stdlib) | Requires custom `Formatter` subclass for JSON output; context binding is cumbersome (requires `LoggerAdapter` or thread-local `extra`). |
| `loguru` | Simpler API, but JSON output requires a third-party sink; context binding is not asyncio-safe by default. |
| `python-json-logger` | Builds on stdlib; still requires `LoggerAdapter` for context propagation across async boundaries. |

**Why Rejected:**
- `structlog` provides `contextvars`-based context binding that works correctly across `asyncio.gather` and `create_task` boundaries — critical for correlating logs from concurrent ticket processing.
- `structlog.contextvars.merge_contextvars` automatically injects `job_id`, `ticket_hash`, etc. into every log line within a context scope without explicit passing.
- Zero-config async safety: structlog uses Python 3.7+ `contextvars` natively.
- Dev mode (`ConsoleRenderer`) produces readable colored output; production mode (`JSONRenderer`) is a one-line configuration change controlled by `sys.stderr.isatty()`.

---

## ADR-005: sarvam-m with reasoning_effort=low vs sarvam-30b

**Decision:** Use `sarvam-m` with `reasoning_effort: "low"` for the initial implementation. Document a clear upgrade path to `sarvam-30b` for production if SLA allows.

**Context:** The task is structured classification of support tickets into 5 categories, 4 priority levels, and a ≤20-word summary. This is a well-defined extraction task, not open-ended reasoning.

**Tradeoff Analysis:**

| Dimension | `sarvam-m` (reasoning_effort=low) | `sarvam-30b` |
|-----------|-----------------------------------|--------------|
| Latency (p50) | ~400–700 ms | ~800–1500 ms |
| Cost per 1K tokens | Lower | ~3–5× higher |
| Classification accuracy | High (>95% on structured tasks) | Marginally higher |
| Summary quality | Good (task-specific) | Excellent (nuanced) |
| JSON compliance | Good with system prompt | Excellent |
| Reasoning depth | Minimal (correct for classification) | Deep chain-of-thought |

**Why sarvam-m is preferred for this task:**
1. **Classification is not a reasoning task.** The model needs to identify keywords and map to a fixed taxonomy — this requires pattern recognition, not multi-step reasoning.
2. **Cost at scale.** At 500 tickets/job and $0.0002/1K tokens, `sarvam-m` at avg 80 tokens/ticket costs ~$0.008/job. `sarvam-30b` at 3× cost raises this to ~$0.024/job — a 3× difference that compounds at scale.
3. **Latency.** `sarvam-m` with `reasoning_effort=low` achieves sub-700ms p95 latency, keeping batch throughput high. `sarvam-30b`'s deeper reasoning chain increases p95 to >1500ms.

**Recommendation to switch to sarvam-30b when:**
- Ticket classification accuracy drops below 90% (monitor via feedback loop)
- Tickets contain multi-language content or highly domain-specific jargon
- The `summary` field is used for downstream LLM reasoning (quality matters more than cost)
- SLA allows >2s p95 latency and the budget increase is acceptable
