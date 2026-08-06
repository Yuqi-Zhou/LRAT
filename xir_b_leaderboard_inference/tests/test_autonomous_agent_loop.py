"""Focused offline tests for the native BC-Plus autonomous loop.

The fake client only returns predeclared model decisions.  It never supplies a
synthetic tool call or final answer; the assertions therefore exercise the
runner's real request/history/tool-execution path without contacting a model or
MCP server.
"""

from __future__ import annotations

import copy
import os
import sys
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
os.environ.setdefault(
    "BCP_CAMPAIGN_CONFIG",
    str(PACKAGE_ROOT / "configs" / "campaign.example.json"),
)
os.environ.setdefault("BCP_RUN_ID", "offline-test")
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))

from query_worker import CampaignAssistant  # noqa: E402


SEARCH_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}
DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {"docid": {"type": "string"}},
    "required": ["docid"],
}


class FakeTool:
    def __init__(self, name: str, schema: dict):
        self.name = name
        self.function = {
            "name": name,
            "description": "test tool",
            "parameters": copy.deepcopy(schema),
        }


class FakeCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class FakeCompletion:
    def __init__(self, tool_calls=None, content="", finish_reason="stop"):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content="",
                    reasoning=None,
                    tool_calls=list(tool_calls or []),
                ),
                finish_reason=finish_reason,
            )
        ]


class FakeCompletions:
    def __init__(self, completions):
        self._completions = list(completions)
        self.requests = []

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if not self._completions:
            raise AssertionError("fake model was called after its declared output")
        return self._completions.pop(0)


def tool_call(call_id: str, tool_name: str, arguments: str) -> FakeCall:
    return FakeCall(call_id, tool_name, arguments)


def new_assistant(completions, tool_outputs=None):
    assistant = CampaignAssistant.__new__(CampaignAssistant)
    assistant._native_model = "test-model"
    assistant._native_generation = {}
    assistant._native_final_generation = {}
    assistant.native_api_mode = "chat_completions"
    assistant._native_context_budget_config = None
    assistant._native_context_budget_runtime = None
    assistant.native_requested_protocol_strategy = "configured"
    assistant.native_protocol_strategy = "autonomous_qwen_parity"
    assistant.native_generation_budget_evidence = []
    assistant.native_context_limit_evidence = {}
    assistant.native_tokenizer_evidence = {}
    assistant.native_forced_final_count = 0
    assistant.native_forced_final_reasons = []
    assistant.interaction_steps = 0
    assistant.llm_call_count = 0
    assistant.executed_tool_call_count = 0
    assistant.model_context_truncation_count = 0
    assistant.native_history_compaction_count = 0
    assistant.final_markup_repair_count = 0
    assistant.tool_protocol_retry_count = 0
    assistant.malformed_tool_call_retry_count = 0
    assistant.native_tool_rejection_count = 0
    assistant.tool_protocol_retry_reasons = []
    assistant.native_tool_call_bindings = []
    assistant.native_tool_name_mappings = []
    assistant._native_advertised_mcp_tools = []
    assistant.native_message_history = []
    assistant.native_response_evidence = []
    assistant.native_force_final_pending = False
    assistant.native_force_final_pending_reason = None

    assistant.function_map = OrderedDict(
        [
            (
                "search-server-search",
                FakeTool("search-server-search", SEARCH_SCHEMA),
            ),
            (
                "search-server-get_document",
                FakeTool("search-server-get_document", DOCUMENT_SCHEMA),
            ),
        ]
    )
    completions_client = FakeCompletions(completions)
    assistant._native_client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions_client)
    )
    assistant._live_advertised_mcp_tools = lambda selected: []
    assistant._prepend_knowledge_prompt = lambda **kwargs: kwargs["messages"]
    outputs = iter(tool_outputs or [])

    def call_tool(name, arguments, **kwargs):
        del arguments, kwargs
        try:
            return next(outputs)
        except StopIteration:
            return '{"ok":true}'

    assistant._call_tool = call_tool
    return assistant, completions_client


def run_loop(assistant):
    chunks = list(
        assistant._run(
            [{"role": "user", "content": "opaque test input"}],
            knowledge="provided",
        )
    )
    return chunks[-1] if chunks else []


class AutonomousAgentLoopTest(unittest.TestCase):
    def test_zero_tool_immediate_final(self):
        assistant, client = new_assistant(
            [FakeCompletion(content="model final")]
        )
        response = run_loop(assistant)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["tool_choice"], "auto")
        self.assertEqual(
            [item["function"]["name"] for item in client.requests[0]["tools"]],
            ["search-server-search", "search-server-get_document"],
        )
        self.assertEqual(assistant.llm_call_count, 1)
        self.assertEqual(assistant.executed_tool_call_count, 0)
        self.assertEqual(response[-1]["role"], "assistant")
        self.assertEqual(response[-1]["content"], "model final")

    def test_repeated_searches_return_to_model(self):
        outputs = [
            FakeCompletion(
                [tool_call("s1", "search-server-search", '{"query":"a"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(
                [tool_call("s2", "search-server-search", '{"query":"b"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(content="final after repeated search"),
        ]
        assistant, client = new_assistant(outputs, ['{"hits":[1]}', '{"hits":[2]}'])
        run_loop(assistant)
        self.assertEqual(assistant.llm_call_count, 3)
        self.assertEqual(assistant.executed_tool_call_count, 2)
        self.assertEqual(
            [b["tool_call_id"] for b in assistant.native_tool_call_bindings],
            ["s1", "s2"],
        )
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(
            [message["role"] for message in client.requests[1]["messages"]],
            ["user", "assistant", "tool"],
        )

    def test_repeated_get_document_is_model_selected(self):
        outputs = [
            FakeCompletion(
                [tool_call("d1", "search-server-get_document", '{"docid":"D1"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(
                [tool_call("d2", "search-server-get_document", '{"docid":"D2"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(content="final after repeated documents"),
        ]
        assistant, client = new_assistant(outputs, ['{"docid":"D1"}', '{"docid":"D2"}'])
        run_loop(assistant)
        self.assertEqual(assistant.llm_call_count, 3)
        self.assertEqual(assistant.executed_tool_call_count, 2)
        self.assertEqual(
            [b["tool_call_id"] for b in assistant.native_tool_call_bindings],
            ["d1", "d2"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in client.requests[2]["messages"] if message["role"] == "tool"],
            ["d1", "d2"],
        )

    def test_mixed_multi_call_turn_preserves_model_order(self):
        outputs = [
            FakeCompletion(
                [
                    tool_call("m1", "search-server-search", '{"query":"a"}'),
                    tool_call("m2", "search-server-get_document", '{"docid":"D1"}'),
                ],
                finish_reason="tool_calls",
            ),
            FakeCompletion(content="final after mixed turn"),
        ]
        assistant, client = new_assistant(outputs, ['{"hits":[1]}', '{"docid":"D1"}'])
        run_loop(assistant)
        self.assertEqual(
            [b["tool_call_id"] for b in assistant.native_tool_call_bindings],
            ["m1", "m2"],
        )
        second_request = client.requests[1]["messages"]
        self.assertEqual(
            [item["role"] for item in second_request],
            ["user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [item["tool_call_id"] for item in second_request if item["role"] == "tool"],
            ["m1", "m2"],
        )
        self.assertEqual(
            [call["id"] for call in second_request[1]["tool_calls"]], ["m1", "m2"]
        )

    def test_natural_final_after_arbitrary_tool_turns(self):
        outputs = [
            FakeCompletion(
                [tool_call("a1", "search-server-search", '{"query":"a"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(
                [tool_call("a2", "search-server-get_document", '{"docid":"D1"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(
                [tool_call("a3", "search-server-search", '{"query":"b"}')],
                finish_reason="tool_calls",
            ),
            FakeCompletion(content="natural final"),
        ]
        assistant, _ = new_assistant(outputs, ['{}', '{}', '{}'])
        response = run_loop(assistant)
        self.assertEqual(assistant.llm_call_count, 4)
        self.assertEqual(assistant.executed_tool_call_count, 3)
        self.assertEqual(response[-1]["role"], "assistant")
        self.assertEqual(response[-1]["content"], "natural final")
        self.assertFalse(any(item.get("function_call") for item in response[-1:]))

    def test_forced_final_omits_empty_tools_payload(self):
        assistant, client = new_assistant(
            [
                FakeCompletion(
                    content=(
                        "Explanation: evidence\n"
                        "Exact Answer: answer\n"
                        "Confidence: high"
                    )
                )
            ]
        )
        assistant.interaction_steps = 49
        assistant.llm_call_count = 49
        run_loop(assistant)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(client.requests[0]["tool_choice"], "none")
        self.assertNotIn("tools", client.requests[0])

    def test_length_only_turn_recovers_with_forced_final(self):
        assistant, client = new_assistant(
            [
                FakeCompletion(content="", finish_reason="length"),
                FakeCompletion(
                    content=(
                        "Explanation: evidence\n"
                        "Exact Answer: answer\n"
                        "Confidence: high"
                    )
                ),
            ]
        )
        response = run_loop(assistant)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[0]["tool_choice"], "auto")
        self.assertIn("tools", client.requests[0])
        self.assertEqual(client.requests[1]["tool_choice"], "none")
        self.assertNotIn("tools", client.requests[1])
        self.assertEqual(response[-1]["content"].splitlines()[1], "Exact Answer: answer")

    def test_fifty_turn_termination_has_no_synthetic_final(self):
        outputs = [
            FakeCompletion(
                [tool_call(f"c{i}", "search-server-search", '{"query":"x"}')],
                finish_reason="tool_calls",
            )
            for i in range(50)
        ]
        assistant, client = new_assistant(outputs, ["{}"] * 50)
        response = run_loop(assistant)
        self.assertEqual(len(client.requests), 50)
        self.assertEqual(assistant.interaction_steps, 50)
        self.assertEqual(assistant.llm_call_count, 50)
        # The 50th model turn is the reserved no-tool final-safety turn.  If
        # a non-compliant fake model still emits a tool call there, the
        # adapter suppresses it rather than exceeding the step cap.
        self.assertEqual(assistant.executed_tool_call_count, 49)
        self.assertNotIn("tools", client.requests[-1])
        self.assertEqual(client.requests[-1]["tool_choice"], "none")
        self.assertFalse(
            any(
                item.get("role") == "assistant"
                and item.get("content")
                and not item.get("function_call")
                for item in response
            )
        )
        # The non-compliant final-turn tool call is suppressed and recorded as
        # an empty assistant safety turn; it must not be executed.
        self.assertEqual(assistant.native_message_history[-1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
