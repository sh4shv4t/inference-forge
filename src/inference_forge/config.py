from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Sarvam AI
    sarvam_api_key: str = Field(..., description="Sarvam AI API subscription key")
    sarvam_api_base: str = Field(
        default="https://api.sarvam.ai/v1", description="Sarvam AI base URL"
    )
    sarvam_model: str = Field(default="sarvam-m", description="Model to use for inference")

    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")

    # Pipeline concurrency
    max_concurrent_api_calls: int = Field(
        default=10, ge=1, le=50, description="Global semaphore limit for API calls"
    )

    # Circuit breaker
    cb_failure_threshold: int = Field(
        default=5, ge=1, description="Failures within window before opening circuit"
    )
    cb_window_seconds: int = Field(
        default=20, ge=5, description="Sliding window size for failure counting"
    )
    cb_recovery_timeout: int = Field(
        default=30, ge=5, description="Seconds before attempting HALF_OPEN probe"
    )

    # Retry
    max_retries: int = Field(default=3, ge=1, le=5, description="Max attempts per API call")
    retry_backoff_base: float = Field(default=1.0, description="Base backoff in seconds")

    # Cache / Job TTLs
    dedup_ttl_seconds: int = Field(default=86400, description="Cache entry TTL (24 hours)")
    job_ttl_seconds: int = Field(default=3600, description="Job state TTL (1 hour)")

    # Validation limits
    max_tickets_per_request: int = Field(default=1000, description="Max tickets per /process call")
    max_ticket_chars: int = Field(default=2000, description="Max characters per ticket")

    # Observability
    log_level: str = Field(default="INFO", description="Log level")
    latency_window_size: int = Field(
        default=1000, description="Rolling window size for latency percentile computation"
    )

    # Cost
    cost_per_1k_tokens: float = Field(
        default=0.0002, description="USD cost per 1000 tokens"
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"log_level must be one of {allowed}"
            raise ValueError(msg)
        return upper


settings = Settings()
