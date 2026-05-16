"""
Integration tests — full pipeline flow using the FastAPI test client.

These tests use fakeredis and a mocked httpx client, so they do NOT require
a live Sarvam AI API key or Redis instance. The test marked with
`@pytest.mark.live` will contact the real API and requires SARVAM_API_KEY.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest


def _sarvam_success_response(category: str = "billing") -> httpx.Response:
    body = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "category": category,
                            "priority": "medium",
                            "summary": "Test summary for integration test",
                        }
                    )
                }
            }
        ],
        "usage": {"total_tokens": 42},
    }
    return httpx.Response(
        status_code=200,
        content=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", "https://api.sarvam.ai/v1/chat/completions"),
    )


class TestProcessEndpoint:
    async def test_process_returns_job_id(self, test_app) -> None:
        client, mock_http = test_app
        mock_http.post = AsyncMock(return_value=_sarvam_success_response())

        response = await client.post(
            "/process", json={"tickets": ["My invoice is incorrect"]}
        )
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert data["total"] == 1
        assert data["unique"] >= 0

    async def test_process_validates_max_tickets(self, test_app) -> None:
        client, _ = test_app
        tickets = ["ticket"] * 1001
        response = await client.post("/process", json={"tickets": tickets})
        assert response.status_code == 422

    async def test_process_validates_empty_ticket(self, test_app) -> None:
        client, _ = test_app
        response = await client.post("/process", json={"tickets": ["   "]})
        assert response.status_code == 422

    async def test_process_validates_long_ticket(self, test_app) -> None:
        client, _ = test_app
        long_ticket = "a" * 2001
        response = await client.post("/process", json={"tickets": [long_ticket]})
        assert response.status_code == 422

    async def test_duplicate_tickets_counted_as_cached(self, test_app) -> None:
        client, mock_http = test_app
        mock_http.post = AsyncMock(return_value=_sarvam_success_response())

        ticket = "My account is blocked"
        response = await client.post("/process", json={"tickets": [ticket, ticket]})
        assert response.status_code == 202
        data = response.json()
        # One is unique, one is a duplicate
        assert data["cached"] >= 1

    async def test_complexity_breakdown_present(self, test_app) -> None:
        client, mock_http = test_app
        mock_http.post = AsyncMock(return_value=_sarvam_success_response())

        response = await client.post(
            "/process",
            json={
                "tickets": [
                    "Short",
                    " ".join(["word"] * 30),
                    " ".join(["word"] * 70),
                ]
            },
        )
        data = response.json()
        bd = data["complexity_breakdown"]
        assert "simple" in bd
        assert "medium" in bd
        assert "complex" in bd


class TestResultsEndpoint:
    async def test_results_404_for_unknown_job(self, test_app) -> None:
        client, _ = test_app
        response = await client.get("/results/nonexistent-job-id")
        assert response.status_code == 404

    async def test_results_pending_when_processing(self, test_app) -> None:
        client, mock_http = test_app
        mock_http.post = AsyncMock(return_value=_sarvam_success_response())

        post = await client.post("/process", json={"tickets": ["test"]})
        job_id = post.json()["job_id"]

        # Immediately check — may be pending or done
        response = await client.get(f"/results/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("pending", "processing", "done")


class TestHealthEndpoint:
    async def test_health_returns_ok(self, test_app) -> None:
        client, _ = test_app
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "circuit_breaker" in data
        assert "redis" in data
        assert "metrics" in data


class TestMetricsEndpoint:
    async def test_metrics_prometheus_format(self, test_app) -> None:
        client, _ = test_app
        response = await client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "inference_forge_cache_hits_total" in content
        assert "inference_forge_circuit_breaker_state" in content


class TestStreamEndpoint:
    async def test_stream_404_for_unknown_job(self, test_app) -> None:
        client, _ = test_app
        response = await client.get("/stream/nonexistent-job-id")
        assert response.status_code == 404


@pytest.mark.live
class TestLiveAPI:
    """
    These tests call the real Sarvam AI API.
    Run with:  poetry run pytest tests/integration -m live -v
    Requires SARVAM_API_KEY to be set in .env
    """

    TICKETS = [
        "My invoice shows incorrect charges for this month.",
        "I cannot log in to my account even after resetting password.",
        "The API is returning 500 errors intermittently when I POST to /orders.",
        "Please add dark mode support to the dashboard.",
        "My subscription was cancelled but I was still charged.",
        (
            "When I try to upload files larger than 10MB through the web interface, "
            "the upload appears to succeed but the file is not visible in my account. "
            "This started happening after the last deployment on Friday. I've tried "
            "Chrome and Firefox with the same result. My account ID is 12345."
        ),
        "How do I export my data?",
        "The mobile app crashes on startup on iOS 17.",
        "Can you add support for bulk CSV imports?",
        "I need to update my billing address.",
    ]

    async def test_live_pipeline_10_tickets(self) -> None:
        import os

        if not os.getenv("SARVAM_API_KEY"):
            pytest.skip("SARVAM_API_KEY not set")

        from httpx import AsyncClient, ASGITransport
        from inference_forge.main import app

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/process", json={"tickets": self.TICKETS})
            assert response.status_code == 202
            job_id = response.json()["job_id"]

            # Poll for completion
            for _ in range(30):
                await asyncio.sleep(2)
                result = await client.get(f"/results/{job_id}")
                if result.json().get("status") == "done":
                    break

            final = await client.get(f"/results/{job_id}")
            data = final.json()
            assert data["status"] == "done"
            assert data["results"] is not None
            assert len(data["results"]) == len(self.TICKETS)

            # Validate result schema
            for r in data["results"]:
                if r.get("error") is None:
                    assert r["category"] in {
                        "billing", "technical", "account", "feature_request", "other"
                    }
                    assert r["priority"] in {"low", "medium", "high", "critical"}
