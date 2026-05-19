"""Test a complex ticket known to fail parsing."""
import asyncio
import sys

from inference_forge.pipeline.caller import _call_with_retry, build_http_client

COMPLEX_TICKET = (
    "I am writing to report a critical production issue that started on 2026-04-23. "
    "When users upload files larger than 35MB at https://app.example.com/upload, "
    "the progress bar reaches 100% but the file doesn't appear in the dashboard. "
    "Tested on Chrome 122, Firefox 123, Safari 17. "
    "Server log: StorageException: multipart upload aborted. "
    "Tried rolling back deployment. Blocking 287 active users. SLA: 4h. "
    "Account: 54321. Contact: admin@enterprise.com. Priority: CRITICAL."
)


async def main() -> None:
    sys.stderr.write(f"Testing complex ticket (len={len(COMPLEX_TICKET)})...\n")
    async with build_http_client() as client:
        result = await _call_with_retry(client, COMPLEX_TICKET)
    sys.stderr.write(f"RESULT: {result}\n")
    if result.get("error"):
        sys.stderr.write(f"FAILURE: error={result['error']}\n")
        sys.exit(1)
    else:
        sys.stderr.write("SUCCESS\n")


asyncio.run(main())
