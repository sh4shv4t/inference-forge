"""Quick end-to-end test of _single_api_call with the think-tag fix."""
import asyncio
import sys

from inference_forge.pipeline.caller import (
    _single_api_call,
    _strip_think_tags,
    build_http_client,
)


async def main() -> None:
    ticket = "My invoice is wrong."
    sys.stderr.write("Testing _strip_think_tags import...\n")
    assert callable(_strip_think_tags), "strip_think_tags not found"
    sys.stderr.write("OK\n")

    sys.stderr.write("Calling _single_api_call...\n")
    async with build_http_client() as client:
        result, tokens = await _single_api_call(client, ticket, attempt=1)

    sys.stderr.write(f"result={result}  tokens={tokens}\n")
    assert result.get("category"), f"No category in result: {result}"
    sys.stderr.write("SUCCESS\n")


asyncio.run(main())
