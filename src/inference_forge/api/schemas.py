"""Pydantic v2 request and response models for the inference-forge API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from inference_forge.config import settings


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ProcessRequest(BaseModel):
    tickets: list[str] = Field(
        ...,
        min_length=1,
        max_length=settings.max_tickets_per_request,
        description="List of support ticket texts to classify",
    )

    @field_validator("tickets")
    @classmethod
    def validate_ticket_lengths(cls, tickets: list[str]) -> list[str]:
        for i, t in enumerate(tickets):
            if len(t) > settings.max_ticket_chars:
                msg = (
                    f"Ticket at index {i} exceeds {settings.max_ticket_chars} characters "
                    f"(got {len(t)})"
                )
                raise ValueError(msg)
            if not t.strip():
                msg = f"Ticket at index {i} is empty or whitespace-only"
                raise ValueError(msg)
        return tickets


# ---------------------------------------------------------------------------
# Response Models
# ---------------------------------------------------------------------------


class ComplexityBreakdownSchema(BaseModel):
    simple: int
    medium: int
    complex: int


class ProcessResponse(BaseModel):
    job_id: str
    total: int
    cached: int
    unique: int
    estimated_seconds: float
    complexity_breakdown: ComplexityBreakdownSchema


class TicketResult(BaseModel):
    ticket: str | None = None
    category: str | None = None
    priority: str | None = None
    summary: str | None = None
    error: str | None = None
    cache_hit: bool = False
    tokens: int = 0


class JobStats(BaseModel):
    total_tokens: int
    cost_usd: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    cache_hit_rate: float
    failure_rate: float
    duration_seconds: float


class StreamDoneEvent(BaseModel):
    status: str = "done"
    results: list[dict[str, Any]]
    stats: JobStats


class ResultsResponse(BaseModel):
    status: str
    progress: float | None = None
    results: list[dict[str, Any]] | None = None
    stats: dict[str, Any] | None = None


class HealthMetrics(BaseModel):
    total_api_calls: int
    total_tokens_used: int
    estimated_cost_usd: float
    p95_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    circuit_breaker: str
    redis: str
    cache_hit_rate: float
    queue_depth: int
    active_jobs: int
    metrics: HealthMetrics
