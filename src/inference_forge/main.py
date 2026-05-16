"""FastAPI application entry point with lifespan context manager."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference_forge import __version__
from inference_forge.api.routes import router
from inference_forge.config import settings
from inference_forge.observability.logger import configure_logging, get_logger
from inference_forge.pipeline.cache import DeduplicationCache
from inference_forge.pipeline.caller import SarvamCaller, build_http_client
from inference_forge.pipeline.job_store import JobStore

import asyncio

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application-wide resources:
      - Redis connections (main + dedicated pub/sub subscriber client)
      - httpx async client (shared connection pool)
      - Global asyncio semaphore
      - SarvamCaller, DeduplicationCache, JobStore singletons
    """
    configure_logging()
    logger.info("startup", version=__version__)

    # Redis — main client (commands + publish)
    redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=20,
    )

    # Redis — dedicated subscriber client (pub/sub cannot multiplex with commands)
    sub_redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=50,
    )

    # Verify connectivity
    try:
        await redis_client.ping()
        logger.info("redis_connected", url=settings.redis_url)
    except Exception as exc:
        logger.error("redis_connection_failed", error=str(exc))
        raise

    # Shared httpx client
    http_client = build_http_client()

    # Global semaphore (caps concurrent Sarvam API calls across ALL jobs)
    semaphore = asyncio.Semaphore(settings.max_concurrent_api_calls)

    # Domain objects
    cache = DeduplicationCache(redis_client)
    job_store = JobStore(redis_client)
    caller = SarvamCaller(http_client, cache, semaphore)

    # Attach to app.state so routes can access them via request.app.state
    app.state.redis = redis_client
    app.state.sub_redis = sub_redis_client
    app.state.http_client = http_client
    app.state.cache = cache
    app.state.job_store = job_store
    app.state.caller = caller
    app.state.semaphore = semaphore

    yield  # ← application runs here

    # Shutdown
    logger.info("shutdown")
    await http_client.aclose()
    await redis_client.aclose()
    await sub_redis_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="inference-forge",
        description="Production-grade async batch inference pipeline for Sarvam AI support ticket classification",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, tags=["inference"])

    return app


app = create_app()
