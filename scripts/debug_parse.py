"""Debug script: test JSON parsing on a live Sarvam response."""
import asyncio, json, re, sys
import httpx
from inference_forge.config import settings
from inference_forge.pipeline.caller import SYSTEM_PROMPT, _strip_think_tags, _extract_json_object


async def main() -> None:
    ticket = "Warmup ticket."
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
    sys.stdout.write("Calling API...\n"); sys.stdout.flush()
    async with httpx.AsyncClient(timeout=60.0, http2=True) as c:
        r = await c.post(
            f"{settings.sarvam_api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers={"api-subscription-key": settings.sarvam_api_key},
        )
    sys.stdout.write(f"HTTP {r.status_code}\n"); sys.stdout.flush()
    content = r.json()["choices"][0]["message"]["content"]
    sys.stdout.write(f"CONTENT LEN: {len(content)}\n"); sys.stdout.flush()
    sys.stdout.write(f"FULL CONTENT:\n{content}\n===END===\n"); sys.stdout.flush()
    cleaned = _strip_think_tags(content)
    sys.stdout.write(f"CLEANED LEN: {len(cleaned)}\n"); sys.stdout.flush()
    sys.stdout.write(f"FULL CLEANED:\n{cleaned}\n===END===\n"); sys.stdout.flush()
    try:
        result = json.loads(cleaned)
        sys.stdout.write(f"json.loads OK: {result}\n"); sys.stdout.flush()
    except json.JSONDecodeError as e:
        sys.stdout.write(f"json.loads FAILED: {e}\n"); sys.stdout.flush()
        try:
            result = _extract_json_object(content)
            sys.stdout.write(f"_extract_json_object OK: {result}\n"); sys.stdout.flush()
        except json.JSONDecodeError as e2:
            sys.stdout.write(f"_extract_json_object FAILED: {e2}\n"); sys.stdout.flush()


asyncio.run(main())
