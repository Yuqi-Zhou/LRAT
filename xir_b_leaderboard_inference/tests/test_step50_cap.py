"""Regression checks for the XIR-B 50-call upper bound."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PACKAGE_ROOT / "configs" / "campaign.example.json"
WORKER_PATH = PACKAGE_ROOT / "search_agent" / "query_worker.py"
RUNNER_PATH = PACKAGE_ROOT / "scripts" / "run_shard.py"
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
os.environ.setdefault("BCP_CAMPAIGN_CONFIG", str(CONFIG_PATH))
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))

from campaign_common import validate_trajectory  # noqa: E402


def test_config_worker_and_runner_are_exactly_step50() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["max_interaction_steps"] == 50

    worker = WORKER_PATH.read_text(encoding="utf-8")
    assert "EXPECTED_LIMIT = 50" in worker
    assert 'os.environ["BCP_MAX_TOOL_STEPS"] = "50"' in worker
    assert 'os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "50"' in worker
    assert 'os.environ["BCP_MAX_INTERACTION_STEPS"] = "50"' in worker
    assert "interaction_steps >= EXPECTED_LIMIT" in worker
    assert "LLM turn counter exceeded 50" in worker

    runner = RUNNER_PATH.read_text(encoding="utf-8")
    assert '"BCP_MAX_TOOL_STEPS": "50"' in runner
    assert '"BCP_MAX_INTERACTION_STEPS": "50"' in runner
    assert '"QWEN_AGENT_MAX_LLM_CALL_PER_RUN": "50"' in runner
    assert '"max_interaction_steps": 50' in runner


def test_no_legacy_25_step_contract_remains_in_package_code() -> None:
    for path in (
        PACKAGE_ROOT / "search_agent" / "query_worker.py",
        PACKAGE_ROOT / "search_agent" / "campaign_common.py",
        PACKAGE_ROOT / "scripts" / "run_shard.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert "step25" not in source.lower()
        assert "exactly_25" not in source
        assert "limit_is_25" not in source


def test_healthy_fifty_call_tool_trajectory_is_terminal_not_retryable(tmp_path) -> None:
    result = []
    raw_response = []
    native_history = [{"role": "user", "content": "test query"}]
    response_evidence = []
    for index in range(50):
        call_id = f"call-{index}"
        arguments = json.dumps({"query": f"search {index}"})
        output = json.dumps([{"docid": f"doc-{index}"}])
        result.append(
            {
                "type": "tool_call",
                "tool_name": "search",
                "arguments": arguments,
                "output": output,
            }
        )
        raw_response.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "function_call": {
                        "name": "search-server-search",
                        "arguments": arguments,
                    },
                },
                {
                    "role": "function",
                    "name": "search-server-search",
                    "content": output,
                },
            ]
        )
        native_history.extend(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": "search-server-search",
                                "arguments": arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": "search-server-search",
                    "content": output,
                },
            ]
        )
        response_evidence.append(
            {
                "finish_reason": "tool_calls",
                "tool_calls": [{"id_present": True, "name": "search-server-search"}],
            }
        )

    payload = {
        "metadata": {
            "profile": "gpt-oss-120b",
            "retriever_slug": "test-retriever",
            "tool_api_mode": "native_openai_real_mcp",
            "trial_id": "offline-test",
            "model_endpoint": "http://127.0.0.1:8001/v1",
            "mcp_endpoint": "http://127.0.0.1:38190/mcp",
            "final_answer_source": "model_assistant_natural",
            "synthetic_final": False,
            "max_interaction_steps": 50,
            "max_output_tokens": 4096,
            "native_response_evidence": response_evidence,
            "native_advertised_mcp_tools": [
                {
                    "name": "search",
                    "required": ["query"],
                    "property_names": ["query"],
                },
                {
                    "name": "get_document",
                    "required": ["docid"],
                    "property_names": ["docid"],
                },
            ],
            "complete_tool_returns_persisted": True,
            "mcp_init_gate": {"graceful_shutdown_error_class": None},
        },
        "query_id": "test-query",
        "status": "incomplete",
        "error_class": "IncompleteTrajectory",
        "tool_call_counts": {"search": 50},
        "result": result,
        "raw_response": raw_response,
        "native_message_history": native_history,
        "interaction_step_count": 50,
        "llm_call_count": 50,
        "executed_tool_call_count": 50,
        "final_answer": "",
        "timing": {"duration_seconds": 1.0},
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    verification = validate_trajectory(
        path,
        expected_query_id="test-query",
        expected_profile="gpt-oss-120b",
        expected_retriever_slug="test-retriever",
    )
    assert verification["passed"] is True
    assert verification["termination_reason"] == "max_steps_limit"
    assert verification["max_step_termination"]["passed"] is True
