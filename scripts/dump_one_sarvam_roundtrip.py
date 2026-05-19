"""One-shot Sarvam round-trip for debugging (prints redacted request + full response)."""
from __future__ import annotations

import asyncio
import json
import sys

import httpx

from inference_forge.config import settings
from inference_forge.pipeline.caller import SYSTEM_PROMPT


async def main() -> None:
    if settings.sarvam_mock_mode:
        print("SARVAM_MOCK_MODE=true — no HTTP call.", file=sys.stderr)
        sys.exit(2)

    ticket = "My invoice is wrong."
    payload = {
        "model": settings.sarvam_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
        "reasoning_effort": "low",
        "response_format": {"type": "json_object"},
    }
    base = settings.sarvam_api_base.rstrip("/")
    url = f"{base}/chat/completions"

    print("=== REQUEST (same shape as caller.py) ===")
    print("URL:", url)
    print("Headers:", json.dumps({"api-subscription-key": "***REDACTED***"}))
    print("JSON body:", json.dumps(payload, indent=2, ensure_ascii=False))

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0), http2=True) as client:
        r = await client.post(
            url,
            json=payload,
            headers={"api-subscription-key": settings.sarvam_api_key},
        )

    print()
    print("=== RESPONSE ===")
    print("HTTP status:", r.status_code)
    print("Content-Type:", r.headers.get("content-type", ""))
    try:
        body = r.json()
        print(json.dumps(body, indent=2, ensure_ascii=False))
        if r.status_code == 200 and body.get("choices"):
            content = body["choices"][0]["message"]["content"]
            print()
            print("=== message.content (inner JSON text) ===")
            print(content)
    except Exception:
        print(r.text[:8000])


if __name__ == "__main__":
    asyncio.run(main())
