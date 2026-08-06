import os
import sys
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
os.environ.setdefault(
    "BCP_CAMPAIGN_CONFIG",
    str(PACKAGE_ROOT / "configs" / "campaign.example.json"),
)
os.environ.setdefault("BCP_RUN_ID", "offline-test")
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))

from query_worker import (
    ContextExhausted,
    DeepSeekV4ContextBudget,
    calculate_native_generation_budget,
)


class _FakeEncoding:
    def __init__(self):
        self.seen_messages = None

    def encode_messages(self, messages, **_kwargs):
        self.seen_messages = messages
        return "rendered"


class _FakeTokenizer:
    def encode(self, prompt, **_kwargs):
        assert prompt == "rendered"
        return list(range(37))


class DeepSeekGenerationBudgetTests(unittest.TestCase):
    def test_short_context_keeps_proven_4096_cap(self):
        evidence = calculate_native_generation_budget(100, 32768)
        self.assertEqual(evidence["effective_max_tokens"], 4096)
        self.assertEqual(evidence["input_tokens"], 100)
        self.assertEqual(evidence["context_limit"], 32768)

    def test_cumulative_long_context_reduces_budget(self):
        evidence = calculate_native_generation_budget(32000, 32768)
        self.assertEqual(evidence["effective_max_tokens"], 512)
        self.assertEqual(evidence["remaining_tokens"], 768)

    def test_tools_schema_is_counted_with_complete_wire_messages(self):
        counter = DeepSeekV4ContextBudget.__new__(DeepSeekV4ContextBudget)
        counter._tokenizer = _FakeTokenizer()
        counter._encoding = _FakeEncoding()
        messages = [{"role": "user", "content": "query"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search-server-search",
                    "description": "search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        count = counter.count(messages, tools, {"extra_body": {}})
        self.assertEqual(count, 37)
        self.assertEqual(counter._encoding.seen_messages[0]["role"], "system")
        self.assertEqual(counter._encoding.seen_messages[0]["tools"], tools)
        self.assertEqual(counter._encoding.seen_messages[1], messages[0])

    def test_budget_shortage_is_explicit_and_has_no_final_answer(self):
        with self.assertRaises(ContextExhausted) as raised:
            calculate_native_generation_budget(32700, 32768)
        evidence = raised.exception.evidence
        self.assertEqual(evidence["state"], "context_exhausted")
        self.assertEqual(evidence["effective_max_tokens"], 0)
        self.assertNotIn("final", evidence)

if __name__ == "__main__":
    unittest.main()
