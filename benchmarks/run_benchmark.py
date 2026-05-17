"""
Benchmark runner for inference-forge.

Generates a synthetic dataset of 200 tickets (40% simple, 40% medium, 20% complex,
25% duplicates) and sweeps concurrency levels [2, 5, 10, 15, 20].

Usage:
    poetry run python benchmarks/run_benchmark.py

Requires the inference-forge server to be running:
    make dev   OR   make docker-up
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("BENCHMARK_BASE_URL", "http://localhost:8000")
DUPLICATE_RATIO = 0.25
COST_PER_1K_TOKENS = 0.0002


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.getenv(name)
    if not value:
        return default
    levels: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            level = int(item)
        except ValueError:
            continue
        if level > 0:
            levels.append(level)
    return levels or default


TOTAL_TICKETS = _env_int("BENCHMARK_TOTAL_TICKETS", 20)
CONCURRENCY_LEVELS = _env_int_list(
    "BENCHMARK_CONCURRENCY_LEVELS", [2, 5, 10, 15, 20]
)
POLL_INTERVAL = _env_float("BENCHMARK_POLL_INTERVAL", 0.5)
MAX_WAIT_S = _env_int("BENCHMARK_MAX_WAIT_S", 1800)
CB_COOLDOWN_S = _env_int("BENCHMARK_CB_COOLDOWN_S", 120)
WARMUP_ENABLED = os.getenv("BENCHMARK_WARMUP", "1") == "1"

console = Console()

# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

_SIMPLE_TEMPLATES = [
    "My invoice is wrong.",
    "I can't log in.",
    "My account is locked.",
    "Please cancel my subscription.",
    "I need a refund.",
    "How do I reset my password?",
    "My payment failed.",
    "I was charged twice.",
    "Update my email address.",
    "I forgot my username.",
]

_MEDIUM_TEMPLATES = [
    (
        "I've been experiencing issues with the {feature} feature for the past {days} days. "
        "When I try to {action}, the system returns an error message saying '{error}'. "
        "I've tried {workaround} but the problem persists. My account ID is {account_id}."
    ),
    (
        "Hello support team, I'm writing about a billing discrepancy in my account. "
        "According to my last invoice dated {date}, I was charged {amount} for {plan} plan. "
        "However, I should only be on the {expected_plan} plan which costs {expected_amount}. "
        "Please investigate and issue a correction."
    ),
    (
        "The {api_endpoint} endpoint has been returning {status_code} errors intermittently "
        "since yesterday's deployment. Our monitoring shows {error_rate}% error rate in the "
        "last hour. This is affecting {customer_count} of our downstream customers. "
        "Please prioritize this."
    ),
]

_COMPLEX_TEMPLATES = [
    (
        "I am writing to report a critical production issue that started occurring on {date}. "
        "When users attempt to upload files larger than {size}MB through our web interface at "
        "{url}, the upload progress bar reaches 100% and shows a success message, but the file "
        "does not appear in the user's account dashboard. We have confirmed this behavior on "
        "Chrome {chrome_version}, Firefox {firefox_version}, and Safari {safari_version}. "
        "The issue does not occur for files smaller than {small_size}MB. "
        "We have checked the server logs and see the following error: {error_log}. "
        "Our DevOps team has tried {attempted_fix} without success. "
        "This is blocking {blocked_users} active users from completing their work. "
        "Our SLA requires a response within 4 hours. Account ID: {account_id}. "
        "Contact: {email}. Priority: CRITICAL."
    ),
    (
        "We are an enterprise customer on the {plan} plan (contract ID: {contract_id}) and "
        "we are experiencing severe performance degradation affecting all {user_count} of our "
        "users since {start_time}. Specifically: (1) API response times have increased from "
        "an average of {normal_latency}ms to {degraded_latency}ms, (2) the {feature} feature "
        "is completely unavailable, returning {error_code} for all requests, (3) our "
        "integration with {integration} is broken - webhooks are not being delivered. "
        "We have attempted the following troubleshooting steps: {steps}. "
        "None of these have resolved the issue. We need an urgent escalation to your "
        "infrastructure team. Our business is losing approximately ${loss_per_hour} per hour. "
        "Please assign a dedicated support engineer immediately."
    ),
]


def _fill(template: str) -> str:
    rng = random.Random()
    replacements = {
        "{feature}": random.choice(["upload", "export", "bulk import", "API", "dashboard"]),
        "{days}": str(random.randint(1, 7)),
        "{action}": random.choice(["submit a form", "upload a file", "call the API", "log in"]),
        "{error}": random.choice(["500 Internal Server Error", "403 Forbidden", "Timeout", "ECONNREFUSED"]),
        "{workaround}": random.choice(["clearing cache", "using a different browser", "VPN"]),
        "{account_id}": str(random.randint(10000, 99999)),
        "{date}": f"2026-{random.randint(1,5):02d}-{random.randint(1,28):02d}",
        "{amount}": f"${random.randint(50, 500)}.00",
        "{plan}": random.choice(["Pro", "Business", "Enterprise"]),
        "{expected_plan}": random.choice(["Starter", "Basic", "Pro"]),
        "{expected_amount}": f"${random.randint(10, 100)}.00",
        "{api_endpoint}": random.choice(["/v1/orders", "/v1/users", "/v1/payments", "/v1/webhooks"]),
        "{status_code}": random.choice(["500", "503", "429", "502"]),
        "{error_rate}": str(random.randint(10, 80)),
        "{customer_count}": str(random.randint(10, 500)),
        "{size}": str(random.randint(5, 50)),
        "{url}": "https://app.example.com/upload",
        "{chrome_version}": f"12{random.randint(0,4)}.0",
        "{firefox_version}": f"12{random.randint(0,4)}.0",
        "{safari_version}": f"1{random.randint(5,8)}.0",
        "{small_size}": str(random.randint(1, 4)),
        "{error_log}": "StorageException: multipart upload aborted, ETag mismatch",
        "{attempted_fix}": random.choice(["restarting the service", "rolling back the deployment"]),
        "{blocked_users}": str(random.randint(50, 5000)),
        "{email}": "admin@enterprise.com",
        "{contract_id}": f"ENT-{random.randint(1000,9999)}",
        "{user_count}": str(random.randint(100, 10000)),
        "{start_time}": f"2026-05-{random.randint(10,16):02d} {random.randint(0,23):02d}:00 UTC",
        "{normal_latency}": str(random.randint(50, 200)),
        "{degraded_latency}": str(random.randint(2000, 10000)),
        "{error_code}": random.choice(["503", "500", "504"]),
        "{integration}": random.choice(["Slack", "Salesforce", "Zapier", "Datadog"]),
        "{steps}": "restarting workers, checking network, rolling back config changes",
        "{loss_per_hour}": f"{random.randint(500, 50000):,}",
    }
    result = template
    for k, v in replacements.items():
        result = result.replace(k, v)
    return result


def generate_dataset(seed: int = 42) -> list[str]:
    random.seed(seed)
    n_simple = int(TOTAL_TICKETS * 0.40)
    n_medium = int(TOTAL_TICKETS * 0.40)
    n_complex = TOTAL_TICKETS - n_simple - n_medium

    base_tickets: list[str] = []
    base_tickets += [random.choice(_SIMPLE_TEMPLATES) for _ in range(n_simple)]
    base_tickets += [_fill(random.choice(_MEDIUM_TEMPLATES)) for _ in range(n_medium)]
    base_tickets += [_fill(random.choice(_COMPLEX_TEMPLATES)) for _ in range(n_complex)]

    random.shuffle(base_tickets)

    # Seed 25% duplicates
    n_dupes = int(TOTAL_TICKETS * DUPLICATE_RATIO)
    all_tickets = base_tickets.copy()
    for _ in range(n_dupes):
        all_tickets.append(random.choice(base_tickets))

    random.shuffle(all_tickets)
    return all_tickets[: TOTAL_TICKETS]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    concurrency: int
    duration_s: float = 0.0
    throughput: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    total_tokens: int = 0
    cost_usd: float = 0.0
    failure_rate: float = 0.0
    cache_hit_rate: float = 0.0
    error: str = ""


async def run_single_benchmark(
    concurrency: int,
    tickets: list[str],
    client: httpx.AsyncClient,
) -> BenchmarkResult:
    result = BenchmarkResult(concurrency=concurrency)

    # Patch concurrency via env (requires server restart) — for standalone benchmark,
    # we pass it as a query param or just measure at different server concurrency levels.
    # Here we submit the job and measure wall-clock time to completion.
    console.print(f"  [yellow]Running concurrency={concurrency}...[/yellow]")

    try:
        t0 = time.monotonic()
        resp = await client.post(
            "/process",
            json={"tickets": tickets},
            timeout=30.0,
        )
        resp.raise_for_status()
        job_data = resp.json()
        job_id = job_data["job_id"]

        # Poll until done or failed
        deadline = time.monotonic() + MAX_WAIT_S
        last_payload: dict[str, Any] = {}
        while time.monotonic() < deadline:
            await asyncio.sleep(POLL_INTERVAL)
            r = await client.get(f"/results/{job_id}", timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                last_payload = data
                st = data.get("status")
                if st == "done":
                    break
                if st == "failed":
                    reason = (data.get("stats") or {}).get("reason", "unknown_error")
                    result.error = f"job_failed: {reason}"
                    return result
        else:
            hint = ""
            if last_payload:
                st = last_payload.get("status")
                pr = last_payload.get("progress")
                hint = f" (last status={st!r}, progress={pr})"
            result.error = f"timeout after {MAX_WAIT_S}s{hint}"
            return result

        elapsed = time.monotonic() - t0

        # Gather stats from /results
        r = await client.get(f"/results/{job_id}", timeout=10.0)
        data = r.json()
        results_list: list[dict] = data.get("results") or []

        n_failed = sum(1 for r in results_list if r.get("error"))
        total = len(results_list) or 1
        total_tokens = sum(r.get("tokens", 0) for r in results_list)

        # Latency from /health (global percentiles)
        health_r = await client.get("/health", timeout=5.0)
        health = health_r.json()
        metrics = health.get("metrics", {})

        result.duration_s = round(elapsed, 2)
        result.throughput = round(total / elapsed, 2)
        result.p95_ms = metrics.get("p95_latency_ms", 0.0)
        result.total_tokens = total_tokens
        result.cost_usd = round((total_tokens / 1000) * COST_PER_1K_TOKENS, 6)
        result.failure_rate = round(n_failed / total * 100, 2)
        result.cache_hit_rate = round(health.get("cache_hit_rate", 0.0) * 100, 2)

    except Exception as exc:
        result.error = str(exc)

    return result


async def wait_for_circuit_closed(client: httpx.AsyncClient) -> bool:
    """Wait for the circuit breaker to close before starting a run."""
    deadline = time.monotonic() + CB_COOLDOWN_S
    while time.monotonic() < deadline:
        try:
            r = await client.get("/health", timeout=5.0)
            if r.status_code == 200:
                state = r.json().get("circuit_breaker")
                if state == "CLOSED":
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def warmup_probe(client: httpx.AsyncClient) -> bool:
    """Send a single ticket and ensure the result has no error."""
    resp = await client.post("/process", json={"tickets": ["Warmup ticket"]})
    if resp.status_code != 202:
        return False
    job_id = resp.json().get("job_id")
    if not job_id:
        return False

    deadline = time.monotonic() + MAX_WAIT_S
    while time.monotonic() < deadline:
        await asyncio.sleep(POLL_INTERVAL)
        r = await client.get(f"/results/{job_id}")
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == "failed":
                reason = (data.get("stats") or {}).get("reason", "")
                console.print(f"[red]Warmup job failed: {reason}[/red]")
                return False
            if data.get("status") == "done":
                results = data.get("results") or []
                return all(not item.get("error") for item in results)
    return False


async def main() -> None:
    console.rule("[bold blue]inference-forge Benchmark Runner")
    console.print(f"Target: [cyan]{BASE_URL}[/cyan]")
    console.print(f"Dataset: [cyan]{TOTAL_TICKETS}[/cyan] tickets (25% duplicates)")
    console.print(f"Concurrency sweep: {CONCURRENCY_LEVELS}\n")
    console.print(
        "[yellow]Note:[/yellow] Rows label target concurrency; for a valid semaphore sweep, "
        "restart the API each time with [bold]MAX_CONCURRENT_API_CALLS=<level>[/bold] "
        "(otherwise all rows measure the same server config).\n"
    )
    console.print(
        f"Circuit breaker cooldown: [cyan]{CB_COOLDOWN_S}s[/cyan] between runs\n"
    )

    tickets = generate_dataset()
    console.print(f"Generated [green]{len(tickets)}[/green] tickets\n")

    results: list[BenchmarkResult] = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60.0) as client:
        # Verify server is up
        try:
            r = await client.get("/health")
            r.raise_for_status()
            console.print("[green]Server is healthy. Starting benchmark...[/green]\n")
            if r.json().get("circuit_breaker") != "CLOSED":
                console.print("[red]Circuit breaker is open. Restart the server and retry.[/red]")
                sys.exit(1)
        except Exception as exc:
            console.print(f"[red]Server not reachable: {exc}[/red]")
            console.print("Start the server with: [bold]make dev[/bold] or [bold]make docker-up[/bold]")
            sys.exit(1)

        if WARMUP_ENABLED:
            ok = await warmup_probe(client)
            if not ok:
                console.print("[red]Warmup probe failed. Aborting benchmark.[/red]")
                sys.exit(1)

        for level in CONCURRENCY_LEVELS:
            ready = await wait_for_circuit_closed(client)
            if not ready:
                console.print(
                    "[red]Circuit breaker still open. Aborting benchmark.[/red]"
                )
                sys.exit(1)
            r = await run_single_benchmark(level, tickets, client)
            results.append(r)
            if r.error:
                console.print(f"  [red]Error at concurrency={level}: {r.error}[/red]")

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    table = Table(
        title="Benchmark Results — inference-forge",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Concurrency", justify="center")
    table.add_column("Duration (s)", justify="right")
    table.add_column("Throughput (t/s)", justify="right")
    table.add_column("P95 Latency (ms)", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Failure %", justify="right")
    table.add_column("Cache Hit %", justify="right")

    best = max(results, key=lambda r: r.throughput if not r.error else 0)

    for r in results:
        style = "bold green" if r == best else ""
        table.add_row(
            str(r.concurrency),
            f"{r.duration_s:.2f}",
            f"{r.throughput:.2f}" + (" *" if r == best else ""),
            f"{r.p95_ms:.0f}",
            f"{r.total_tokens:,}",
            f"${r.cost_usd:.4f}",
            f"{r.failure_rate:.1f}%",
            f"{r.cache_hit_rate:.1f}%",
            style=style,
        )

    console.print(table)

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    console.rule("[bold]Analysis")

    if best.throughput > 0:
        cost_per_1k = (best.cost_usd / TOTAL_TICKETS) * 1000
        console.print(
            f"[green]Optimal concurrency:[/green] [bold]{best.concurrency}[/bold] "
            f"— peak throughput {best.throughput:.2f} tickets/sec"
        )
        console.print(
            f"[green]Cost per 1000 tickets at optimal concurrency:[/green] "
            f"[bold]${cost_per_1k:.4f}[/bold]"
        )

    # Cache savings
    cache_result = best
    if cache_result.cache_hit_rate > 0:
        hits = int(TOTAL_TICKETS * cache_result.cache_hit_rate / 100)
        # Assume avg 50 tokens/ticket
        tokens_saved = hits * 50
        cost_saved = (tokens_saved / 1000) * COST_PER_1K_TOKENS
        console.print(
            f"[green]Cache deduplication savings:[/green] "
            f"~{hits} cache hits, "
            f"~{tokens_saved:,} tokens saved, "
            f"~${cost_saved:.4f} USD saved"
        )

    # ---------------------------------------------------------------------------
    # Markdown table for README
    # ---------------------------------------------------------------------------
    console.rule("[bold]Markdown Table (paste into README.md)")
    md = "| Concurrency | Duration (s) | Throughput (t/s) | P95 Latency (ms) | Total Tokens | Cost (USD) | Failure % | Cache Hit % |\n"
    md += "|-------------|-------------|-----------------|-----------------|-------------|-----------|----------|------------|\n"
    for r in results:
        star = " *" if r == best else ""
        md += (
            f"| {r.concurrency}{star} | {r.duration_s:.2f} | {r.throughput:.2f} | "
            f"{r.p95_ms:.0f} | {r.total_tokens:,} | ${r.cost_usd:.4f} | "
            f"{r.failure_rate:.1f}% | {r.cache_hit_rate:.1f}% |\n"
        )
    console.print(md)


if __name__ == "__main__":
    asyncio.run(main())
