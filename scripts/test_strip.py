"""Test _strip_think_tags with inline debug."""
import sys
import json

# Test directly first (without importing the module)
def strip_think_new(text):
    """New rfind-based approach."""
    if "</think>" in text:
        after = text[text.rfind("</think>") + len("</think>"):].strip()
        return after if after else text.strip()
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned if cleaned else text.strip()

# Test case: complex think block where JSON is after </think>
test1 = '<think>\nReasoning here. </think>\n{"category": "software_issue", "priority": "high", "summary": "test"}'
result1 = strip_think_new(test1)
sys.stderr.write(f"Test1 result: {repr(result1)}\n")
assert result1 == '{"category": "software_issue", "priority": "high", "summary": "test"}', f"FAIL: {result1}"
sys.stderr.write("Test1 PASS\n")

# Test case: think block with braces inside
test2 = '<think>\nI need {something} here. Answer: {"category": "other"} is wrong. </think>\n{"category": "billing", "priority": "medium", "summary": "invoice issue"}'
result2 = strip_think_new(test2)
sys.stderr.write(f"Test2 result: {repr(result2)}\n")
assert result2 == '{"category": "billing", "priority": "medium", "summary": "invoice issue"}', f"FAIL: {result2}"
sys.stderr.write("Test2 PASS\n")

# Now test with actual imported function
from inference_forge.pipeline.caller import _strip_think_tags
sys.stderr.write(f"\nImported function source check...\n")
import inspect
src = inspect.getsource(_strip_think_tags)
sys.stderr.write(f"Source snippet: {src[:200]}\n")
has_rfind = "rfind" in src
sys.stderr.write(f"Has rfind: {has_rfind}\n")

result_imported = _strip_think_tags(test1)
sys.stderr.write(f"Imported result: {repr(result_imported)}\n")
sys.stderr.write(f"json.loads: {json.loads(result_imported)}\n")
sys.stderr.write("All tests PASSED\n")
