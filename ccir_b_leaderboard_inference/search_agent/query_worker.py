#!/usr/bin/env python3
"""Run one isolated, atomically persisted CCIR-B BC-Plus trajectory."""

from __future__ import annotations

import argparse
import copy
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# These values must exist before the first qwen-agent import. Do not move any
# qwen_agent/model_agent_client import above this block.
os.environ["BCP_MAX_TOOL_STEPS"] = "50"
os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"] = "50"
os.environ["BCP_MAX_INTERACTION_STEPS"] = "50"

# Bypass the enterprise proxy only for loopback model/MCP endpoints. External
# retrieval/network traffic continues to use the inherited proxy settings.
_loopback_hosts = {"127.0.0.1", "localhost", "::1"}
for _proxy_name in ("NO_PROXY", "no_proxy"):
    _existing_proxy_bypass = {
        item.strip()
        for item in os.environ.get(_proxy_name, "").split(",")
        if item.strip()
    }
    os.environ[_proxy_name] = ",".join(
        sorted(_existing_proxy_bypass | _loopback_hosts)
    )

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = Path(
    os.environ.get("BCP_CAMPAIGN_ROOT", str(PACKAGE_ROOT))
).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))

from campaign_common import (  # noqa: E402
    atomic_write_json,
    load_config,
    now,
    query_rows,
    read_json,
    validate_trajectory,
)

CONFIG = load_config()
RUN_ID = (
    os.environ.get("BCP_RUN_ID", "").strip()
    or os.environ.get("BCP_TRIAL_ID", "").strip()
)
if not RUN_ID:
    raise RuntimeError("BCP_RUN_ID must be explicit in every live query worker")
# Keep the historical metadata field populated for trajectory readers that
# predate the public CCIR-B package.
os.environ.setdefault("BCP_TRIAL_ID", RUN_ID)
SOURCE = Path(CONFIG["browsecomp_source"])
QUERY_SOURCE_PATH = Path(CONFIG["query_path"])
QUERY_SOURCE_SHA256 = hashlib.sha256(QUERY_SOURCE_PATH.read_bytes()).hexdigest()
QUERY_ROWS = query_rows(CONFIG)
QUERY_INDEX = {query_id: index for index, (query_id, _query) in enumerate(QUERY_ROWS)}
sys.path.insert(0, str(SOURCE / "search_agent"))
sys.path.insert(0, str(SOURCE))

from openai import OpenAI  # noqa: E402
from prompts import format_query  # noqa: E402
from qwen_agent.agents import Assistant  # noqa: E402
from qwen_agent.agents import fncall_agent as fncall_agent  # noqa: E402
from qwen_agent.llm.schema import FUNCTION, USER, FunctionCall, Message  # noqa: E402
import qwen_agent.settings as qwen_settings  # noqa: E402
from utils import extract_retrieved_docids_from_result  # noqa: E402


EXPECTED_LIMIT = 50
MAX_STEP_POLICY = "CCIR_B_HEALTHY_MAX_STEPS_V1"
MCP_INIT_GATE_ENV = "BCP_MCP_INIT_GATE"
MCP_INIT_GATE_TIMEOUT_ENV = "BCP_MCP_INIT_GATE_TIMEOUT_SECONDS"
MCP_INIT_RAMP_LEVELS = (
    (0.0, 4),
    (10.0, 8),
    (30.0, 16),
)
MCP_INIT_LEASE_STALE_SECONDS = 180.0
MCP_INIT_LEASE_HEARTBEAT_SECONDS = 5.0
# Normal native turns preserve the accumulated transcript.  Oversized
# get_document results are bounded at the model-facing adapter boundary so a
# single document cannot consume the entire context window.
if int(os.environ["BCP_MAX_TOOL_STEPS"]) != EXPECTED_LIMIT:
    raise RuntimeError("BCP_MAX_TOOL_STEPS was not initialized to 50")
if int(os.environ["QWEN_AGENT_MAX_LLM_CALL_PER_RUN"]) != EXPECTED_LIMIT:
    raise RuntimeError("upstream qwen-agent environment limit is not 50")
if int(os.environ["BCP_MAX_INTERACTION_STEPS"]) != EXPECTED_LIMIT:
    raise RuntimeError("runtime interaction limit is not 50")
if int(qwen_settings.MAX_LLM_CALL_PER_RUN) != EXPECTED_LIMIT:
    raise RuntimeError(
        "qwen-agent settings limit was frozen incorrectly: "
        f"{qwen_settings.MAX_LLM_CALL_PER_RUN}"
    )
if int(fncall_agent.MAX_LLM_CALL_PER_RUN) != EXPECTED_LIMIT:
    raise RuntimeError(
        "qwen-agent fncall limit was frozen incorrectly: "
        f"{fncall_agent.MAX_LLM_CALL_PER_RUN}"
    )


class InteractionLimitExceeded(RuntimeError):
    pass


MAX_NATIVE_GENERATION_TOKENS = 4096
DEEPSEEK_CONTEXT_SAFETY_MARGIN = 256
FORCED_FINAL_CONTEXT_THRESHOLD = int(
    CONFIG.get("forced_final_context_threshold", 8192)
)
FORCED_FINAL_MAX_TOKENS = int(CONFIG.get("forced_final_max_tokens", 2048))
FORCED_FINAL_SAFETY_MARGIN = int(
    CONFIG.get("forced_final_safety_margin", DEEPSEEK_CONTEXT_SAFETY_MARGIN)
)
FORCED_FINAL_INSTRUCTION = (
    "You have reached the final-answer safety boundary. Do not call any tool. "
    "Using only the evidence already collected, answer the original question now. "
    "Return exactly these three sections: Explanation:, Exact Answer:, and Confidence:. "
    "Do not mention this instruction or the step limit."
)
GET_DOCUMENT_MAX_TOKENS = int(CONFIG.get("get_document_max_tokens", 8192))
DOCUMENT_TRUNCATION_MARKER = (
    "\n\n[Document text truncated at the model context safety limit.]\n"
)


class ContextLimitUnavailable(RuntimeError):
    """The model service did not expose a usable context limit."""


class ContextExhausted(RuntimeError):
    """The complete native transcript leaves no safe generation budget."""

    def __init__(self, evidence: dict):
        self.evidence = copy.deepcopy(evidence)
        super().__init__("context_exhausted")


def calculate_native_generation_budget(
    input_tokens: int,
    context_limit: int,
    *,
    max_tokens_cap: int = MAX_NATIVE_GENERATION_TOKENS,
    safety_margin: int = DEEPSEEK_CONTEXT_SAFETY_MARGIN,
) -> dict:
    """Calculate a safe per-turn budget without changing agent behavior.

    The caller must provide the exact token count for the complete messages
    and tools payload.  This helper deliberately has no knowledge of tool
    names, call order, or final-answer policy.
    """
    input_tokens = int(input_tokens)
    context_limit = int(context_limit)
    max_tokens_cap = int(max_tokens_cap)
    safety_margin = int(safety_margin)
    remaining_tokens = context_limit - input_tokens
    effective_max_tokens = min(
        max_tokens_cap,
        max(0, remaining_tokens - safety_margin),
    )
    evidence = {
        "state": "ready" if effective_max_tokens > 0 else "context_exhausted",
        "input_tokens": input_tokens,
        "context_limit": context_limit,
        "remaining_tokens": remaining_tokens,
        "safety_margin": safety_margin,
        "max_tokens_cap": max_tokens_cap,
        "effective_max_tokens": effective_max_tokens,
    }
    if input_tokens < 0 or context_limit <= 0 or max_tokens_cap <= 0:
        evidence["state"] = "context_exhausted"
        evidence["reason"] = "invalid_token_budget_inputs"
        raise ContextExhausted(evidence)
    if effective_max_tokens <= 0:
        evidence["reason"] = "remaining_budget_below_safety_margin"
        raise ContextExhausted(evidence)
    return evidence


class DeepSeekV4ContextBudget:
    """Exact local mirror of vLLM's official DeepSeek-V4 tokenizer wrapper.

    The wrapper inserts the OpenAI tools list into a system message, renders
    with the exact official vLLM ``deepseek_v4_encoding.encode_messages``
    installed beside the running service, and counts the resulting string
    with the same tokenizer.json.  It does not trim history or impose any
    tool/final-answer behavior.
    """

    def __init__(
        self,
        *,
        native_server: str,
        native_model: str,
        fallback_model_path: str,
        encoding_path: str | None = None,
        max_tokens_cap: int = MAX_NATIVE_GENERATION_TOKENS,
        safety_margin: int = DEEPSEEK_CONTEXT_SAFETY_MARGIN,
    ) -> None:
        self.native_server = native_server.rstrip("/")
        self.native_model = native_model
        self.fallback_model_path = Path(fallback_model_path).resolve()
        self.encoding_path_override = (
            Path(encoding_path).resolve() if encoding_path else None
        )
        self.max_tokens_cap = int(max_tokens_cap)
        self.safety_margin = int(safety_margin)
        self.service_evidence: dict = {}
        self.tokenizer_evidence: dict = {}
        self.context_limit: int | None = None
        self._tokenizer = None
        self._encoding = None
        self._load_service_metadata()

    @staticmethod
    def _sha256_file(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _read_local_json(self, path: str) -> dict:
        request = urllib.request.Request(
            f"{self.native_server}/models",
            headers={"Accept": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=5.0) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise ContextLimitUnavailable(
                f"failed to read DeepSeek service metadata from {path}: {type(exc).__name__}"
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ContextLimitUnavailable("DeepSeek service /v1/models returned invalid metadata")
        return payload

    def _load_service_metadata(self) -> None:
        endpoint = f"{self.native_server}/models"
        payload = self._read_local_json(endpoint)
        entries = [item for item in payload["data"] if isinstance(item, dict)]
        selected = next(
            (item for item in entries if str(item.get("id")) == self.native_model),
            entries[0] if len(entries) == 1 else None,
        )
        if not isinstance(selected, dict):
            raise ContextLimitUnavailable("DeepSeek service model id was not uniquely identified")
        try:
            context_limit = int(selected["max_model_len"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContextLimitUnavailable("DeepSeek service omitted max_model_len") from exc
        if context_limit <= 0:
            raise ContextLimitUnavailable("DeepSeek service returned non-positive max_model_len")
        service_root = selected.get("root")
        model_path = Path(str(service_root)).resolve() if service_root else self.fallback_model_path
        if not model_path.is_dir():
            model_path = self.fallback_model_path
        self.context_limit = context_limit
        self.service_evidence = {
            "source": endpoint,
            "served_model_id": selected.get("id"),
            "context_limit": context_limit,
            "service_root": str(service_root) if service_root else None,
            "tokenizer_model_path_selected": str(model_path),
        }
        self._load_official_tokenizer(model_path)

    def _load_official_tokenizer(self, model_path: Path) -> None:
        encoding_path = self.encoding_path_override or (
            model_path / "encoding" / "encoding_dsv4.py"
        )
        tokenizer_path = model_path / "tokenizer.json"
        if not encoding_path.is_file() or not tokenizer_path.is_file():
            raise ContextLimitUnavailable(
                "DeepSeek official encoding_dsv4.py or tokenizer.json is unavailable"
            )
        try:
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_path))
            module_name = f"bcplus_deepseek_v4_encoding_{os.getpid()}"
            spec = importlib.util.spec_from_file_location(module_name, str(encoding_path))
            if spec is None or spec.loader is None:
                raise ImportError("unable to load encoding_dsv4 module")
            encoding = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(encoding)
        except Exception as exc:
            raise ContextLimitUnavailable(
                f"failed to load official DeepSeek tokenizer: {type(exc).__name__}"
            ) from exc
        self._tokenizer = tokenizer
        self._encoding = encoding
        self.tokenizer_evidence = {
            "implementation": "official_vllm_deepseek_v4_encoding_plus_PreTrainedTokenizerFast",
            "model_path": str(model_path),
            "encoding_path": str(encoding_path),
            "encoding_sha256": self._sha256_file(encoding_path),
            "tokenizer_json_path": str(tokenizer_path),
            "tokenizer_json_sha256": self._sha256_file(tokenizer_path),
            "tokenizer_mode": "deepseek_v4",
            "jinja_chat_template_used": False,
            "service_encoding_override": self.encoding_path_override is not None,
        }

    @staticmethod
    def _chat_template_kwargs(generation: dict) -> dict:
        extra_body = generation.get("extra_body") or {}
        return dict(extra_body.get("chat_template_kwargs") or {})

    def count(self, messages: list[dict], tools: list[dict], generation: dict) -> int:
        if self._tokenizer is None or self._encoding is None:
            raise ContextLimitUnavailable("DeepSeek tokenizer was not initialized")
        conversation = copy.deepcopy(messages)
        if tools:
            conversation.insert(
                0,
                {"role": "system", "tools": copy.deepcopy(tools)},
            )
        template_kwargs = self._chat_template_kwargs(generation)
        thinking = bool(
            template_kwargs.get("thinking") or template_kwargs.get("enable_thinking")
        )
        thinking_mode = "thinking" if thinking else "chat"
        reasoning_effort = template_kwargs.get("reasoning_effort")
        if reasoning_effort == "none":
            thinking_mode = "chat"
            reasoning_effort = None
        elif reasoning_effort in {"max", "xhigh"}:
            reasoning_effort = "max"
        elif isinstance(reasoning_effort, str):
            reasoning_effort = "high"
        else:
            reasoning_effort = None
        prompt = self._encoding.encode_messages(
            conversation,
            thinking_mode=thinking_mode,
            drop_thinking=True,
            reasoning_effort=reasoning_effort,
        )
        return len(self._tokenizer.encode(prompt, add_special_tokens=False))

    def budget(
        self,
        messages: list[dict],
        tools: list[dict],
        generation: dict,
        *,
        max_tokens_cap: int | None = None,
        safety_margin: int | None = None,
        forced_final: bool = False,
        forced_final_reason: str | None = None,
    ) -> dict:
        if self.context_limit is None:
            raise ContextLimitUnavailable("DeepSeek service context limit was not loaded")
        input_tokens = self.count(messages, tools, generation)
        evidence = calculate_native_generation_budget(
            input_tokens,
            self.context_limit,
            max_tokens_cap=(
                self.max_tokens_cap if max_tokens_cap is None else int(max_tokens_cap)
            ),
            safety_margin=(
                self.safety_margin if safety_margin is None else int(safety_margin)
            ),
        )
        evidence.update(
            {
                "service": copy.deepcopy(self.service_evidence),
                "tokenizer": copy.deepcopy(self.tokenizer_evidence),
                "forced_final": bool(forced_final),
                "forced_final_reason": forced_final_reason,
            }
        )
        return evidence


class MCPInitLease:
    """Cross-process gate for MCP SSE initialization only."""

    def __init__(self, state_path: str | None):
        self.state_path = Path(state_path) if state_path else None
        self.lock_dir = (
            self.state_path.with_name(self.state_path.name + ".lock.d")
            if self.state_path
            else None
        )
        self.pid = os.getpid()
        self.token = f"{self.pid}-{time.time_ns()}"
        self.acquired = False
        self.wait_seconds = 0.0
        self.hold_seconds = 0.0
        self.limit = None
        self._hold_started = None
        self.lease_dir = (
            self.state_path.with_name(
                f"{self.state_path.name}.lease.{self.pid}.{self.token}.d"
            )
            if self.state_path
            else None
        )
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.shutdown_timeout = False
        self.shutdown_error_class: str | None = None

    @staticmethod
    def _pid_live(pid: int) -> bool:
        return pid > 1 and Path(f"/proc/{pid}").exists()

    @staticmethod
    def _default_state() -> dict:
        return {
            "schema_version": 1,
            "started_at": time.time(),
            "ramp_levels": [
                {"after_seconds": after, "limit": limit}
                for after, limit in MCP_INIT_RAMP_LEVELS
            ],
            "active": [],
            "acquired_total": 0,
            "last_update_at": time.time(),
        }

    def _read_state(self) -> dict:
        try:
            payload = json.loads(self.state_path.read_text())
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            payload = self._default_state()
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            payload = self._default_state()
        if not isinstance(payload.get("active"), list):
            payload["active"] = []
        active = []
        for item in payload["active"]:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item.get("pid", 0))
            except (TypeError, ValueError):
                continue
            if self._pid_live(pid):
                active.append(item)
        payload["active"] = active
        return payload

    @staticmethod
    def _limit(state: dict) -> int:
        try:
            elapsed = max(0.0, time.time() - float(state.get("started_at", time.time())))
        except (TypeError, ValueError):
            elapsed = 0.0
        selected = 4
        for item in state.get("ramp_levels") or []:
            try:
                if elapsed >= float(item["after_seconds"]):
                    selected = max(1, int(item["limit"]))
            except (KeyError, TypeError, ValueError):
                continue
        return selected

    def _reap_stale_lock(self) -> None:
        if self.lock_dir is None or not self.lock_dir.exists():
            return
        owner_path = self.lock_dir / "owner.json"
        owner = {}
        try:
            owner = json.loads(owner_path.read_text())
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            owner = {}
        try:
            owner_pid = int(owner.get("pid", 0))
        except (TypeError, ValueError):
            owner_pid = 0
        try:
            age = max(0.0, time.time() - self.lock_dir.stat().st_mtime)
        except OSError:
            age = 0.0
        # A live owner is never reaped.  A missing owner gets a bounded grace
        # period for publishing metadata, after which the directory is moved
        # aside atomically before cleanup.
        if owner_pid and self._pid_live(owner_pid):
            return
        if not owner_pid and age < 30.0:
            return
        stale_path = self.lock_dir.with_name(
            f".{self.lock_dir.name}.stale.{os.getpid()}.{time.time_ns()}.d"
        )
        try:
            os.rename(self.lock_dir, stale_path)
        except OSError:
            return
        shutil.rmtree(stale_path, ignore_errors=True)

    def _try_take_lock(self) -> bool:
        temporary = self.lock_dir.with_name(
            f".{self.lock_dir.name}.tmp.{self.pid}.{time.time_ns()}.d"
        )
        try:
            temporary.mkdir()
        except FileExistsError:
            return False
        try:
            atomic_write_json(
                temporary / "owner.json",
                {
                    "pid": self.pid,
                    "token": self.token,
                    "created_at": time.time(),
                },
            )
            os.rename(temporary, self.lock_dir)
        except OSError as exc:
            # Concurrent initializers can race on the final directory rename.
            # Linux reports replacement of a non-empty lock directory as
            # ENOTEMPTY (not FileExistsError); it is a normal contention
            # result, not a trajectory failure.
            if exc.errno in (errno.EEXIST, errno.ENOTEMPTY):
                shutil.rmtree(temporary, ignore_errors=True)
                self._reap_stale_lock()
                return False
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return True

    def _drop_lock(self) -> None:
        if self.lock_dir is None:
            return
        owner = {}
        try:
            owner = json.loads((self.lock_dir / "owner.json").read_text())
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            return
        if int(owner.get("pid", 0)) != self.pid or owner.get("token") != self.token:
            return
        try:
            (self.lock_dir / "owner.json").unlink()
        except OSError:
            return
        try:
            self.lock_dir.rmdir()
        except OSError:
            return

    def _create_lease(self) -> None:
        if self.lease_dir is None:
            return
        temporary = self.lease_dir.with_name(
            f".{self.lease_dir.name}.tmp.{self.pid}.{time.time_ns()}.d"
        )
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            atomic_write_json(
                temporary / "owner.json",
                {
                    "pid": self.pid,
                    "token": self.token,
                    "trial_id": os.environ.get("BCP_TRIAL_ID"),
                    "campaign_root": os.environ.get("BCP_CAMPAIGN_ROOT"),
                    "started_at": time.time(),
                    "heartbeat_at": time.time(),
                },
            )
            os.rename(temporary, self.lease_dir)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _heartbeat_loop(self) -> None:
        while not self._heartbeat_stop.wait(MCP_INIT_LEASE_HEARTBEAT_SECONDS):
            if self.lease_dir is None:
                return
            owner_path = self.lease_dir / "owner.json"
            try:
                owner = json.loads(owner_path.read_text())
                if int(owner.get("pid", 0)) != self.pid or owner.get("token") != self.token:
                    return
                owner["heartbeat_at"] = time.time()
                atomic_write_json(owner_path, owner)
            except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
                return

    def _start_heartbeat(self) -> None:
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            name="mcp-init-lease-heartbeat",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _drop_lease(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        self._heartbeat_thread = None
        if self.lease_dir is None:
            return
        try:
            owner = json.loads((self.lease_dir / "owner.json").read_text())
        except (FileNotFoundError, PermissionError, json.JSONDecodeError, OSError):
            return
        if int(owner.get("pid", 0)) != self.pid or owner.get("token") != self.token:
            return
        shutil.rmtree(self.lease_dir, ignore_errors=True)

    def acquire(self) -> "MCPInitLease":
        if self.state_path is None:
            return self
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + float(
            os.environ.get(MCP_INIT_GATE_TIMEOUT_ENV, "900")
        )
        started_wait = time.monotonic()
        delay = 0.1
        while time.monotonic() < deadline:
            if self._try_take_lock():
                try:
                    state = self._read_state()
                    limit = self._limit(state)
                    if len(state["active"]) < limit:
                        self._create_lease()
                        state["active"].append(
                            {
                                "pid": self.pid,
                                "token": self.token,
                                "acquired_at": time.time(),
                            }
                        )
                        state["acquired_total"] = int(state.get("acquired_total", 0)) + 1
                        state["last_update_at"] = time.time()
                        atomic_write_json(self.state_path, state)
                        self.limit = limit
                        self.wait_seconds = time.monotonic() - started_wait
                        self._hold_started = time.monotonic()
                        self.acquired = True
                        self._drop_lock()
                        self._start_heartbeat()
                        return self
                finally:
                    self._drop_lock()
            time.sleep(delay)
            delay = min(1.0, delay * 1.5)
        raise RuntimeError("MCPInitGateTimeout")

    def release(self) -> None:
        if not self.acquired or self.state_path is None:
            return
        self.shutdown_timeout = False
        self.shutdown_error_class = None
        self._heartbeat_stop.set()
        try:
            deadline = time.monotonic() + 30.0
            got_lock = False
            while time.monotonic() < deadline:
                if self._try_take_lock():
                    got_lock = True
                    break
                time.sleep(0.1)
            if got_lock:
                state = self._read_state()
                state["active"] = [
                    item
                    for item in state["active"]
                    if not (
                        int(item.get("pid", 0)) == self.pid
                        and item.get("token") == self.token
                    )
                ]
                state["last_update_at"] = time.time()
                atomic_write_json(self.state_path, state)
                self._drop_lock()
            else:
                self.shutdown_timeout = True
                self.shutdown_error_class = "MCPGracefulShutdownTimeout"
        finally:
            self.hold_seconds = (
                max(0.0, time.monotonic() - self._hold_started)
                if self._hold_started is not None
                else 0.0
            )
            self.acquired = False
            self._drop_lease()

    def __enter__(self) -> "MCPInitLease":
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


class CampaignAssistant(Assistant):
    """Native BC-Plus tool-calling assistant with a 50-turn safety stop.

    The upstream qwen-agent contract counts LLM turns, not individual tool
    calls.  A completion may contain zero, one, or many model-selected tool
    calls; valid calls are executed in the returned order and their exact
    results are appended before the next LLM turn.  No client-side action
    sequence is imposed.
    """

    def __init__(
        self,
        *,
        native_model: str,
        native_server: str,
        native_generation: dict,
        native_final_generation: dict,
        native_protocol_strategy: str | None = None,
        native_api_mode: str = "chat_completions",
        native_max_output_tokens: int | None = None,
        native_context_budget: dict | None = None,
        native_document_tokenizer_path: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._native_model = native_model
        self._native_server = native_server
        self._native_client = OpenAI(base_url=native_server, api_key="EMPTY")
        self._native_generation = copy.deepcopy(native_generation)
        self._native_final_generation = copy.deepcopy(native_final_generation)
        self.native_api_mode = native_api_mode
        self._native_max_output_tokens = native_max_output_tokens
        if self.native_api_mode not in {"chat_completions", "responses_harmony"}:
            raise ValueError(f"unknown native API mode: {self.native_api_mode}")
        requested_protocol_strategy = native_protocol_strategy or "configured"
        if requested_protocol_strategy not in {"configured"}:
            raise ValueError(
                "unknown native protocol strategy: "
                f"{requested_protocol_strategy}"
            )
        self.native_requested_protocol_strategy = requested_protocol_strategy
        self.native_protocol_strategy = "autonomous_qwen_parity"
        self._native_context_budget_config = copy.deepcopy(native_context_budget)
        self._native_context_budget_runtime = None
        self._native_document_tokenizer_path = native_document_tokenizer_path
        self._native_document_tokenizer = None
        self.native_generation_budget_evidence: list[dict] = []
        self.native_context_limit_evidence: dict = {}
        self.native_tokenizer_evidence: dict = {}
        self.native_forced_final_count = 0
        self.native_forced_final_reasons: list[str] = []
        self.interaction_steps = 0
        self.llm_call_count = 0
        self.executed_tool_call_count = 0
        self.model_context_truncation_count = 0
        self.native_tool_context_truncations: list[dict] = []
        self.native_force_final_pending = False
        self.native_force_final_pending_reason: str | None = None
        self.native_history_compaction_count = 0
        self.final_markup_repair_count = 0
        self.tool_protocol_retry_count = 0
        self.malformed_tool_call_retry_count = 0
        self.native_tool_rejection_count = 0
        self.tool_protocol_retry_reasons: list[str] = []
        self.native_tool_call_bindings: list[dict] = []
        # Keep model-emitted names separate from the names registered by the
        # Qwen MCP adapter.  A direct MCP name may be canonicalized only when
        # the live advertised schema and exactly one namespaced function
        # schema are byte-for-byte equivalent after JSON normalization.
        self.native_tool_name_mappings: list[dict] = []
        self._native_advertised_mcp_tools: list[dict] = []
        # Exact provider-facing history, including tool_call_id bindings.  The
        # qwen-agent Message projection below is retained for compatibility,
        # while this history is the authoritative protocol audit transcript.
        self.native_message_history: list[dict] = []
        # Keep the native OpenAI/vLLM evidence alongside qwen-agent's legacy
        # function_call projection.  DeepSeek-V4 correctness depends on
        # preserving reasoning plus structured tool_calls across turns.
        self.native_response_evidence = []

    def _ensure_native_context_budget_runtime(self):
        if self._native_context_budget_config is None:
            return None
        if self._native_context_budget_runtime is None:
            config = self._native_context_budget_config
            self._native_context_budget_runtime = DeepSeekV4ContextBudget(
                native_server=self._native_server,
                native_model=self._native_model,
                fallback_model_path=str(config["model_path"]),
                encoding_path=config.get("encoding_path"),
                max_tokens_cap=int(
                    config.get("max_tokens_cap", MAX_NATIVE_GENERATION_TOKENS)
                ),
                safety_margin=int(
                    config.get("safety_margin", DEEPSEEK_CONTEXT_SAFETY_MARGIN)
                ),
            )
            self.native_context_limit_evidence = copy.deepcopy(
                self._native_context_budget_runtime.service_evidence
            )
            self.native_tokenizer_evidence = copy.deepcopy(
                self._native_context_budget_runtime.tokenizer_evidence
            )
        return self._native_context_budget_runtime

    def _native_context_probe(
        self,
        native_messages: list[dict],
        tools: list[dict],
        generate_cfg: dict,
    ) -> dict | None:
        runtime = self._ensure_native_context_budget_runtime()
        if runtime is None or runtime.context_limit is None:
            return None
        input_tokens = runtime.count(native_messages, tools, generate_cfg)
        return {
            "input_tokens": input_tokens,
            "context_limit": runtime.context_limit,
            "remaining_tokens": runtime.context_limit - input_tokens,
        }

    def _native_generation_budget(
        self,
        native_messages: list[dict],
        tools: list[dict],
        generate_cfg: dict,
        *,
        forced_final: bool = False,
        forced_final_reason: str | None = None,
    ) -> dict | None:
        runtime = self._ensure_native_context_budget_runtime()
        if runtime is None:
            return None
        budget_config = self._native_context_budget_config
        try:
            evidence = runtime.budget(
                native_messages,
                tools,
                generate_cfg,
                max_tokens_cap=(
                    FORCED_FINAL_MAX_TOKENS
                    if forced_final
                    else int(
                        budget_config.get(
                            "max_tokens_cap", MAX_NATIVE_GENERATION_TOKENS
                        )
                    )
                ),
                safety_margin=(
                    FORCED_FINAL_SAFETY_MARGIN
                    if forced_final
                    else int(
                        budget_config.get(
                            "safety_margin", DEEPSEEK_CONTEXT_SAFETY_MARGIN
                        )
                    )
                ),
                forced_final=forced_final,
                forced_final_reason=forced_final_reason,
            )
        except ContextExhausted as exc:
            self.native_generation_budget_evidence.append(copy.deepcopy(exc.evidence))
            raise
        self.native_generation_budget_evidence.append(copy.deepcopy(evidence))
        return evidence

    @staticmethod
    def _forced_final_generation(generation: dict, final_generation: dict) -> dict:
        """Build a short, non-tool generation config for the final safety turn."""
        result = copy.deepcopy(generation)
        for key, value in (final_generation or {}).items():
            if key != "extra_body":
                result[key] = copy.deepcopy(value)
        extra_body = copy.deepcopy(result.get("extra_body") or {})
        final_extra_body = copy.deepcopy((final_generation or {}).get("extra_body") or {})
        for key, value in final_extra_body.items():
            if key == "chat_template_kwargs":
                template_kwargs = dict(extra_body.get(key) or {})
                template_kwargs.update(copy.deepcopy(value or {}))
                extra_body[key] = template_kwargs
            else:
                extra_body[key] = copy.deepcopy(value)
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        template_kwargs.update({"thinking": False, "reasoning_effort": "none"})
        extra_body["chat_template_kwargs"] = template_kwargs
        result["extra_body"] = extra_body
        result.pop("max_output_tokens", None)
        result["max_tokens"] = FORCED_FINAL_MAX_TOKENS
        return result

    @staticmethod
    def _native_response_record(completion) -> dict:
        choice = completion.choices[0]
        message = choice.message
        reasoning_content = (
            getattr(message, "reasoning_content", None)
            or getattr(message, "reasoning", None)
            or ""
        )
        tool_calls = []
        for call in list(getattr(message, "tool_calls", None) or []):
            if hasattr(call, "model_dump"):
                try:
                    call = call.model_dump(mode="json")
                except TypeError:
                    call = call.model_dump()
            elif hasattr(call, "dict"):
                call = call.dict()
            tool_calls.append(call)
        if hasattr(completion, "model_dump"):
            try:
                raw_completion = completion.model_dump(mode="json")
            except TypeError:
                raw_completion = completion.model_dump()
        elif hasattr(completion, "dict"):
            raw_completion = completion.dict()
        else:
            raw_completion = {}
        usage = raw_completion.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return {
            "content": message.content or "",
            "reasoning_content": reasoning_content,
            "reasoning_field_present": bool(getattr(message, "reasoning", None)),
            "tool_calls": tool_calls,
            "finish_reason": getattr(choice, "finish_reason", None),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": completion_details.get("reasoning_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
        }

    @staticmethod
    def _harmony_text(item) -> str:
        """Extract text without exposing it in terminal diagnostics."""
        parts = []
        for part in list(getattr(item, "content", None) or []):
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        for part in list(getattr(item, "summary", None) or []):
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _harmony_response_record(response) -> dict:
        """Persist structure/token evidence for one official Responses turn."""
        items = []
        for item in list(getattr(response, "output", None) or []):
            item_type = str(getattr(item, "type", ""))
            entry = {"type": item_type}
            if item_type == "function_call":
                arguments = str(getattr(item, "arguments", "") or "")
                entry.update(
                    {
                        "call_id_present": bool(getattr(item, "call_id", None)),
                        "name": str(getattr(item, "name", "")),
                        "arguments_len": len(arguments),
                        "arguments_sha256": hashlib.sha256(
                            arguments.encode("utf-8")
                        ).hexdigest(),
                        "status": getattr(item, "status", None),
                    }
                )
            else:
                text = CampaignAssistant._harmony_text(item)
                entry.update(
                    {
                        "text_len": len(text),
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if text
                        else None,
                    }
                )
            items.append(entry)
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "model_dump"):
            try:
                usage = usage.model_dump(mode="json")
            except TypeError:
                usage = usage.model_dump()
        usage = usage if isinstance(usage, dict) else {}
        return {
            "protocol": "openai_responses_harmony",
            "status": getattr(response, "status", None),
            "incomplete_reason": getattr(
                getattr(response, "incomplete_details", None), "reason", None
            ),
            "output_items": items,
            "prompt_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "output_tokens": usage.get(
                "output_tokens", usage.get("completion_tokens")
            ),
            "reasoning_tokens": (
                (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
                or (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                )
            ),
        }

    @staticmethod
    def _harmony_input_message(message: dict) -> dict:
        role = str(message.get("role") or "user")
        if role not in {"user", "system", "developer", "assistant"}:
            role = "user"
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        return {"type": "message", "role": role, "content": content}

    @staticmethod
    def _harmony_tool(func) -> dict:
        """Convert a Chat Completions function schema to Responses shape."""
        schema = dict(getattr(func, "function", {}) or {})
        return {
            "type": "function",
            "name": str(schema.get("name") or func.name),
            "description": schema.get("description"),
            "parameters": schema.get("parameters") or {},
            "strict": schema.get("strict"),
        }

    def _harmony_generation(self, generate_cfg: dict) -> dict:
        """Map campaign sampling config to the official Responses request."""
        request = {}
        for key in ("temperature", "top_p"):
            if generate_cfg.get(key) is not None:
                request[key] = generate_cfg[key]
        if self._native_max_output_tokens is not None:
            request["max_output_tokens"] = int(self._native_max_output_tokens)
        elif generate_cfg.get("max_output_tokens") is not None:
            request["max_output_tokens"] = int(generate_cfg["max_output_tokens"])
        template_kwargs = (
            (generate_cfg.get("extra_body") or {}).get("chat_template_kwargs") or {}
        )
        effort = template_kwargs.get("reasoning_effort")
        if effort in {"low", "medium", "high"}:
            request["reasoning"] = {"effort": effort}
        return request

    def _run_harmony(self, messages, lang="en", knowledge="", **kwargs):
        """Run GPT-OSS through vLLM's official Harmony Responses endpoint.

        Tool execution remains the real qwen-agent MCP path.  Only the model
        wire format changes: Responses function_call/function_call_output items
        are kept as input history and never converted from text markup.
        """
        messages = self._prepend_knowledge_prompt(
            messages=messages, lang=lang, knowledge=knowledge, **kwargs
        )
        messages = copy.deepcopy(messages)
        harmony_input = [
            self._harmony_input_message(self._native_message(message))
            for message in messages
        ]
        self.native_message_history = [
            {"role": item["role"], "content": item["content"]}
            for item in harmony_input
        ]
        response = []
        tool_protocol_retries = 0
        # Expose the complete tool set registered by the real MCP adapter on
        # every turn.  The model, rather than this wrapper, chooses whether to
        # call none, one, or many of these tools.
        selected_tools = list(self.function_map.values())
        if not selected_tools:
            raise RuntimeError("BCPlusMCPNoAdvertisedTools")
        offered_tool_specs = [
            {"name": tool.name, "function": copy.deepcopy(tool.function)}
            for tool in selected_tools
        ]
        allowed_names = {tool.name for tool in selected_tools}
        self._native_advertised_mcp_tools = self._live_advertised_mcp_tools(
            selected_tools
        )

        while self.interaction_steps < EXPECTED_LIMIT:
            # BC-Plus's autonomous agent exposes the complete real MCP tool
            # set on every turn and leaves the model's choice unconstrained.
            generate_cfg = copy.deepcopy(self._native_generation)
            if kwargs.get("seed") is not None:
                # vLLM's ResponsesRequest intentionally has no seed field;
                # the seed remains recorded by the campaign metadata instead.
                pass
            request = {
                "model": self._native_model,
                "input": harmony_input,
                "tools": [self._harmony_tool(func) for func in selected_tools],
                "tool_choice": "auto",
                "store": False,
                **self._harmony_generation(generate_cfg),
            }

            self._charge_llm()
            harmony_response = self._native_client.responses.create(**request)
            evidence = self._harmony_response_record(harmony_response)
            evidence.update(
                {
                    "seed": kwargs.get("seed"),
                    "attempt": len(self.native_response_evidence) + 1,
                    "tool_choice_policy": "auto_all_tools",
                }
            )
            self.native_response_evidence.append(evidence)
            output_items = list(getattr(harmony_response, "output", None) or [])
            tool_calls = [
                item for item in output_items if getattr(item, "type", None) == "function_call"
            ]
            content = "".join(
                self._harmony_text(item)
                for item in output_items
                if getattr(item, "type", None) == "message"
            )
            reasoning = "".join(
                self._harmony_text(item)
                for item in output_items
                if getattr(item, "type", None) == "reasoning"
            )
            finish_reason = getattr(
                getattr(harmony_response, "incomplete_details", None), "reason", None
            )
            if getattr(harmony_response, "status", None) != "completed" and not finish_reason:
                finish_reason = "incomplete"
            evidence.update(
                {
                    "finish_reason": "tool_calls" if tool_calls else (finish_reason or "stop"),
                    "tool_calls": [
                        {
                            "id_present": bool(
                                getattr(call, "call_id", None)
                                or getattr(call, "id", None)
                            ),
                            "name": str(getattr(call, "name", "")),
                        }
                        for call in tool_calls
                    ],
                }
            )

            if not tool_calls:
                self.native_message_history.append(
                    {
                        "role": "assistant",
                        "content": content,
                        "reasoning": reasoning,
                        "tool_calls": [],
                    }
                )
                response.append(
                    Message(role="assistant", content=content, reasoning_content=reasoning)
                )
                yield response
                break

            self.native_message_history.append(
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning": reasoning,
                    "tool_calls": [
                        {
                            "id": str(
                                getattr(call, "call_id", None)
                                or getattr(call, "id", "")
                            ),
                            "type": "function",
                            "function": {
                                "name": str(getattr(call, "name", "")),
                                "arguments": str(
                                    getattr(call, "arguments", "") or ""
                                ),
                            },
                        }
                        for call in tool_calls
                    ],
                }
            )

            # Harmony may return more than one model-selected call in a
            # response. Execute every valid call and pair each with its own
            # function_call_output below; there is no client-side tool count.
            call = tool_calls[0]
            tool_name = str(getattr(call, "name", ""))
            tool_args = str(getattr(call, "arguments", "") or "")
            call_id = str(
                getattr(call, "call_id", None) or getattr(call, "id", "")
            )
            allowed_names = {tool.name for tool in selected_tools}
            call_is_valid = bool(call_id and tool_name in allowed_names)
            call_reason = "tool_call_id_or_name_invalid"
            try:
                parsed_args = json.loads(tool_args)
            except (TypeError, json.JSONDecodeError):
                parsed_args = None
                call_is_valid = False
                call_reason = "tool_arguments_not_json"
            if not isinstance(parsed_args, dict):
                call_is_valid = False
                call_reason = "tool_arguments_not_object"
            if finish_reason in {"max_output_tokens", "length"}:
                call_is_valid = False
                call_reason = "finish_reason_length"
            if not call_is_valid:
                self.native_tool_rejection_count += 1
                self.tool_protocol_retry_reasons.append(call_reason)
                raise RuntimeError(f"InvalidHarmonyToolCall:{call_reason}")

            tool_protocol_retries = 0
            self._charge_tool()
            tool_result = self._call_tool(
                tool_name, tool_args, messages=messages, **kwargs
            )
            self.native_tool_call_bindings.append(
                {
                    "tool_call_id": str(
                        getattr(call, "call_id", None)
                        or getattr(call, "id", "")
                    ),
                    "tool_name": tool_name,
                    "result_nonempty": bool(str(tool_result).strip()),
                    "document_id_present": bool(
                        not tool_name.endswith("-get_document")
                        or str(parsed_args.get("docid", "")).strip()
                    ),
                    "arguments_sha256": hashlib.sha256(tool_args.encode("utf-8")).hexdigest(),
                }
            )
            model_tool_result = tool_result
            assistant_msg = Message(
                role="assistant",
                content=content,
                reasoning_content=reasoning,
                function_call=FunctionCall(name=tool_name, arguments=tool_args),
            )
            function_msg = Message(role=FUNCTION, name=tool_name, content=tool_result)
            response.extend([assistant_msg, function_msg])
            messages.extend([assistant_msg, function_msg])
            self.native_message_history.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": tool_result,
                }
            )
            # Preserve the official Responses history item-by-item.  In
            # particular, the assistant function_call and its call_id must be
            # present before the matching function_call_output.  This is the
            # same backfill used by the BC-Plus OSS client.
            for item in output_items:
                if hasattr(item, "model_dump"):
                    try:
                        serialized_item = item.model_dump(mode="python")
                    except TypeError:
                        serialized_item = item.model_dump()
                elif hasattr(item, "dict"):
                    serialized_item = item.dict()
                else:
                    serialized_item = dict(getattr(item, "__dict__", {}) or {})
                if isinstance(serialized_item, dict):
                    serialized_item.pop("status", None)
                    harmony_input.append(serialized_item)
            harmony_input.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": model_tool_result,
                }
            )
            for call in tool_calls[1:]:
                call_id = str(
                    getattr(call, "call_id", None) or getattr(call, "id", "")
                )
                tool_name = str(getattr(call, "name", ""))
                tool_args = str(getattr(call, "arguments", "") or "")
                call_is_valid = bool(call_id and tool_name in allowed_names)
                call_reason = "tool_call_id_or_name_invalid"
                try:
                    parsed_args = json.loads(tool_args)
                except (TypeError, json.JSONDecodeError):
                    parsed_args = None
                    call_is_valid = False
                    call_reason = "tool_arguments_not_json"
                if not isinstance(parsed_args, dict):
                    call_is_valid = False
                    call_reason = "tool_arguments_not_object"
                if finish_reason in {"max_output_tokens", "length"}:
                    call_is_valid = False
                    call_reason = "finish_reason_length"
                if not call_is_valid:
                    self.native_tool_rejection_count += 1
                    self.tool_protocol_retry_reasons.append(call_reason)
                    raise RuntimeError(f"InvalidHarmonyToolCall:{call_reason}")
                self._charge_tool()
                tool_result = self._call_tool(
                    tool_name, tool_args, messages=messages, **kwargs
                )
                self.native_tool_call_bindings.append(
                    {
                        "tool_call_id": call_id,
                        "tool_name": tool_name,
                        "result_nonempty": bool(str(tool_result).strip()),
                        "document_id_present": bool(
                            not tool_name.endswith("-get_document")
                            or str(parsed_args.get("docid", "")).strip()
                        ),
                        "arguments_sha256": hashlib.sha256(tool_args.encode("utf-8")).hexdigest(),
                    }
                )
                model_tool_result = tool_result
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                    function_call=FunctionCall(name=tool_name, arguments=tool_args),
                )
                function_msg = Message(role=FUNCTION, name=tool_name, content=tool_result)
                response.extend([assistant_msg, function_msg])
                messages.extend([assistant_msg, function_msg])
                self.native_message_history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": tool_result,
                    }
                )
                harmony_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": model_tool_result,
                    }
                )
            yield response
        yield response

    @staticmethod
    def _is_get_document_tool(tool_name: str) -> bool:
        return str(tool_name or "") in {"get_document"} or str(
            tool_name or ""
        ).endswith("-get_document")

    def _truncate_document_result(
        self,
        tool_name: str,
        tool_result: str,
        token_limit: int,
    ) -> tuple[str, dict]:
        """Bound one get_document result using the active model tokenizer.

        The MCP server still performs the real retrieval.  The full response
        remains in the legacy trajectory projection for BC-Plus audit and
        validation, but only the bounded projection is copied into the
        model-facing native transcript.  JSON document payloads keep their
        structure while their ``text`` field is reduced; non-JSON payloads use
        a token-safe prefix.
        """
        original = "" if tool_result is None else str(tool_result)
        evidence = {
            "tool_name": str(tool_name),
            "max_tokens": int(token_limit),
            "truncated": False,
            "original_chars": len(original),
            "original_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
        }
        if not self._is_get_document_tool(tool_name):
            return original, evidence

        runtime = self._ensure_native_context_budget_runtime()
        tokenizer = getattr(runtime, "_tokenizer", None) if runtime else None
        tokenizer_path = getattr(self, "_native_document_tokenizer_path", None)
        if tokenizer is None and tokenizer_path:
            if self._native_document_tokenizer is None:
                try:
                    from transformers import AutoTokenizer

                    self._native_document_tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_path,
                        trust_remote_code=True,
                        use_fast=True,
                        local_files_only=True,
                    )
                except Exception as exc:
                    raise ContextLimitUnavailable(
                        "failed to load the model tokenizer for get_document "
                        f"bounding: {type(exc).__name__}"
                    ) from exc
            tokenizer = self._native_document_tokenizer
        if tokenizer is None:
            evidence["state"] = "tokenizer_unavailable"
            return original, evidence

        limit = max(1, int(token_limit))

        def encode(value: str) -> list[int]:
            return list(tokenizer.encode(value, add_special_tokens=False))

        def decode(tokens: list[int]) -> str:
            return tokenizer.decode(tokens, skip_special_tokens=True)

        original_tokens = encode(original)
        evidence["original_tokens"] = len(original_tokens)
        if len(original_tokens) <= limit:
            evidence.update(
                {
                    "state": "within_limit",
                    "model_chars": len(original),
                    "model_tokens": len(original_tokens),
                }
            )
            return original, evidence

        marker = DOCUMENT_TRUNCATION_MARKER
        truncated: str | None = None
        try:
            payload = json.loads(original)
        except (TypeError, json.JSONDecodeError):
            payload = None

        if isinstance(payload, dict) and isinstance(payload.get("text"), str):
            source_tokens = encode(payload["text"])

            def json_candidate(keep: int) -> str:
                candidate = copy.deepcopy(payload)
                candidate_text = decode(source_tokens[:keep]).rstrip()
                candidate["text"] = candidate_text + marker
                candidate["text_truncated"] = True
                candidate["original_text_tokens"] = len(source_tokens)
                candidate["text_token_limit"] = limit
                return json.dumps(candidate, ensure_ascii=False, indent=2)

            low, high = 0, min(len(source_tokens), limit)
            best: str | None = None
            while low <= high:
                middle = (low + high) // 2
                candidate = json_candidate(middle)
                if len(encode(candidate)) <= limit:
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            truncated = best

        if truncated is None:
            marker_tokens = encode(marker)

            def text_candidate(keep: int) -> str:
                prefix = decode(original_tokens[:keep])
                return prefix + marker

            low, high = 0, min(len(original_tokens), limit)
            best = ""
            while low <= high:
                middle = (low + high) // 2
                candidate = text_candidate(middle)
                if len(encode(candidate)) <= limit:
                    best = candidate
                    low = middle + 1
                else:
                    high = middle - 1
            truncated = best or decode(marker_tokens[:limit])

        model_tokens = len(encode(truncated))
        evidence.update(
            {
                "state": "truncated",
                "truncated": True,
                "model_chars": len(truncated),
                "model_tokens": model_tokens,
                "truncation_marker": True,
            }
        )
        return truncated, evidence

    def _model_context_tool_result(
        self,
        tool_name: str,
        tool_result: str,
        *,
        native_messages: list[dict] | None = None,
        tools: list[dict] | None = None,
        generation: dict | None = None,
        tool_call_id: str | None = None,
    ) -> tuple[str, dict]:
        """Return the bounded model-facing MCP result and safety evidence."""
        model_result, evidence = self._truncate_document_result(
            tool_name,
            tool_result,
            GET_DOCUMENT_MAX_TOKENS,
        )
        runtime = self._ensure_native_context_budget_runtime()
        if (
            self._is_get_document_tool(tool_name)
            and runtime is not None
            and runtime.context_limit
            and native_messages is not None
        ):
            normal_generation = copy.deepcopy(generation or self._native_generation)
            candidate_messages = copy.deepcopy(native_messages)
            candidate_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call_id or ""),
                    "content": model_result,
                }
            )
            try:
                normal_input_tokens = runtime.count(
                    candidate_messages,
                    copy.deepcopy(tools or []),
                    normal_generation,
                )
                final_messages = copy.deepcopy(candidate_messages)
                final_messages.append(
                    {"role": "user", "content": FORCED_FINAL_INSTRUCTION}
                )
                final_generation = self._forced_final_generation(
                    normal_generation,
                    self._native_final_generation,
                )
                final_input_tokens = runtime.count(
                    final_messages,
                    [],
                    final_generation,
                )
                evidence.update(
                    {
                        "next_turn_input_tokens": normal_input_tokens,
                        "forced_final_input_tokens": final_input_tokens,
                        "context_limit": runtime.context_limit,
                    }
                )
                normal_remaining = runtime.context_limit - normal_input_tokens
                final_target = (
                    runtime.context_limit
                    - FORCED_FINAL_MAX_TOKENS
                    - FORCED_FINAL_SAFETY_MARGIN
                )
                if normal_remaining <= FORCED_FINAL_CONTEXT_THRESHOLD:
                    self.native_force_final_pending = True
                    self.native_force_final_pending_reason = (
                        "document_context_remaining_below_threshold"
                    )
                if final_input_tokens > final_target:
                    # A nearly-full history can still overflow after an
                    # 8192-token document. Reduce only this document further
                    # so the reserved no-tool final turn remains possible.
                    low, high = 1, GET_DOCUMENT_MAX_TOKENS
                    best_result: str | None = None
                    best_evidence: dict | None = None
                    while low <= high:
                        middle = (low + high) // 2
                        candidate_result, candidate_evidence = (
                            self._truncate_document_result(
                                tool_name, tool_result, middle
                            )
                        )
                        candidate_messages[-1]["content"] = candidate_result
                        candidate_final_messages = copy.deepcopy(candidate_messages)
                        candidate_final_messages.append(
                            {"role": "user", "content": FORCED_FINAL_INSTRUCTION}
                        )
                        candidate_final_tokens = runtime.count(
                            candidate_final_messages,
                            [],
                            final_generation,
                        )
                        if candidate_final_tokens <= final_target:
                            best_result = candidate_result
                            best_evidence = candidate_evidence
                            low = middle + 1
                        else:
                            high = middle - 1
                    if best_result is not None:
                        model_result = best_result
                        candidate_messages[-1]["content"] = model_result
                        evidence = best_evidence or evidence
                        evidence.update(
                            {
                                "next_turn_input_tokens": runtime.count(
                                    candidate_messages,
                                    copy.deepcopy(tools or []),
                                    normal_generation,
                                ),
                                "forced_final_input_tokens": runtime.count(
                                    candidate_messages
                                    + [
                                        {
                                            "role": "user",
                                            "content": FORCED_FINAL_INSTRUCTION,
                                        }
                                    ],
                                    [],
                                    final_generation,
                                ),
                                "context_limit": runtime.context_limit,
                                "context_reserve_truncation": True,
                            }
                        )
                    self.native_force_final_pending = True
                    self.native_force_final_pending_reason = (
                        "document_context_reserve"
                    )
            except (ContextLimitUnavailable, ValueError, TypeError):
                evidence["context_preflight"] = "unavailable"

        if evidence.get("truncated"):
            self.model_context_truncation_count += 1
            self.native_tool_context_truncations.append(copy.deepcopy(evidence))
        return model_result, evidence
    @staticmethod
    def _native_message(message: Message) -> dict:
        result = {"role": message.get("role"), "content": message.get("content", "")}
        if message.get("name"):
            result["name"] = message["name"]
        # vLLM's bundled DeepSeek-V4 encoding uses `reasoning`; the public
        # model encoding uses `reasoning_content`. Normalize only the wire
        # adapter field; the persisted BC-Plus projection keeps its original
        # vocabulary.
        reasoning = message.get("reasoning") or message.get("reasoning_content")
        if reasoning:
            result["reasoning"] = reasoning
        if message.get("tool_calls"):
            result["tool_calls"] = copy.deepcopy(message["tool_calls"])
        if message.get("tool_call_id"):
            result["tool_call_id"] = message["tool_call_id"]
        return result

    def _compact_native_history(self, native_messages: list[dict]) -> list[dict]:
        """Bound model-facing history without fabricating tool results."""
        def cost(message: dict) -> int:
            return len(json.dumps(message, ensure_ascii=False, separators=(",", ":")))

        total = sum(cost(message) for message in native_messages)
        if total <= MODEL_CONTEXT_MAX_HISTORY_CHARS:
            return native_messages

        prefix: list[dict] = []
        blocks: list[list[dict]] = []
        current: list[dict] | None = None
        for message in native_messages:
            is_tool_turn = message.get("role") == "assistant" and message.get("tool_calls")
            if is_tool_turn:
                if current is not None:
                    blocks.append(current)
                current = [message]
            elif current is not None:
                current.append(message)
            else:
                prefix.append(message)
        if current is not None:
            blocks.append(current)

        kept_reversed: list[list[dict]] = []
        kept_cost = sum(cost(message) for message in prefix)
        for block in reversed(blocks):
            block_cost = sum(cost(message) for message in block)
            if not kept_reversed or kept_cost + block_cost <= MODEL_CONTEXT_MAX_HISTORY_CHARS:
                kept_reversed.append(block)
                kept_cost += block_cost
        dropped = len(blocks) - len(kept_reversed)
        if dropped:
            self.native_history_compaction_count += dropped
        return prefix + [message for block in reversed(kept_reversed) for message in block]

    def _charge_llm(self) -> None:
        if self.interaction_steps >= EXPECTED_LIMIT:
            raise InteractionLimitExceeded("LLM turn limit reached")
        self.interaction_steps += 1
        self.llm_call_count += 1

    def _charge_tool(self) -> None:
        # qwen-agent's MAX_LLM_CALL_PER_RUN limits model turns.  Tool calls
        # returned in one model turn are all executed before the next turn and
        # do not consume that model-turn budget.
        self.executed_tool_call_count += 1

    @staticmethod
    def _tool_schema_for_mapping(tool) -> dict:
        """Return the JSON schema used for a tool-name mapping comparison."""
        if isinstance(tool, dict):
            function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            schema = function.get("parameters") if isinstance(function, dict) else None
            return copy.deepcopy(schema) if isinstance(schema, dict) else {}
        function = getattr(tool, "function", {})
        if isinstance(function, dict):
            schema = function.get("parameters")
        else:
            schema = None
        return copy.deepcopy(schema) if isinstance(schema, dict) else {}

    @staticmethod
    def _mcp_schema_hash(schema: dict) -> str:
        normalized = json.dumps(
            schema if isinstance(schema, dict) else {},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @classmethod
    def _canonicalize_mcp_tool_name(
        cls,
        model_tool_name: str,
        offered_tools: list[dict],
        advertised_tools: list[dict],
    ) -> tuple[str | None, str]:
        """Canonicalize only a deterministic MCP namespace difference.

        The model's direct MCP name is accepted only if it is advertised by
        the live server and exactly one offered name has that direct name as
        its final namespace segment with an identical JSON schema.  This is a
        protocol adapter, not a fuzzy name repair: unknown, ambiguous, or
        schema-mismatched names remain rejected.
        """
        model_tool_name = str(model_tool_name or "")
        offered = [
            item
            for item in offered_tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        offered_names = {item["name"] for item in offered}
        if model_tool_name in offered_names:
            return model_tool_name, "already_canonical"

        advertised = {
            str(item.get("name")): item
            for item in advertised_tools
            if isinstance(item, dict) and str(item.get("name") or "")
        }
        direct = advertised.get(model_tool_name)
        if direct is None:
            return None, "unknown_mcp_tool_name"

        direct_schema = direct.get("inputSchema")
        direct_hash = cls._mcp_schema_hash(direct_schema if isinstance(direct_schema, dict) else {})
        suffix_matches = [
            item
            for item in offered
            if item["name"].rsplit("-", 1)[-1] == model_tool_name
            and cls._mcp_schema_hash(cls._tool_schema_for_mapping(item)) == direct_hash
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]["name"], "mcp_namespace_suffix_exact_schema"
        if not suffix_matches:
            return None, "mcp_namespace_schema_mismatch"
        return None, "mcp_namespace_ambiguous"

    @staticmethod
    def _tool_call_validation(
        call,
        allowed_names: set[str],
        canonical_tool_name: str | None = None,
        offered_tools: list[dict] | None = None,
    ) -> tuple[bool, str]:
        """Validate native call structure before any MCP execution."""
        function = getattr(call, "function", None)
        raw_name = (
            getattr(function, "name", None)
            if function is not None
            else getattr(call, "name", None)
        )
        raw_arguments = (
            getattr(function, "arguments", None)
            if function is not None
            else getattr(call, "arguments", None)
        )
        tool_name = canonical_tool_name or str(raw_name or "")
        if tool_name not in allowed_names:
            return False, "tool_name_not_offered_by_schema"
        arguments_text = str(raw_arguments or "")
        try:
            arguments = json.loads(arguments_text)
        except (TypeError, json.JSONDecodeError):
            return False, "tool_arguments_not_json"
        if not isinstance(arguments, dict):
            return False, "tool_arguments_not_object"
        schema = None
        for item in offered_tools or []:
            if isinstance(item, dict) and item.get("name") == tool_name:
                schema = CampaignAssistant._tool_schema_for_mapping(item)
                break
        if isinstance(schema, dict):
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or [])
            missing = sorted(required - set(arguments))
            if missing:
                return False, "tool_arguments_missing_required"
            if schema.get("additionalProperties") is False:
                unknown = sorted(set(arguments) - set(properties))
                if unknown:
                    return False, "tool_arguments_unknown_field"
            for key, value in arguments.items():
                expected = properties.get(key) if isinstance(properties, dict) else None
                expected_type = expected.get("type") if isinstance(expected, dict) else None
                if expected_type == "string" and not isinstance(value, str):
                    return False, "tool_arguments_wrong_type"
        return True, "valid_native_tool_call"

    def _live_advertised_mcp_tools(self, selected_tools) -> list[dict]:
        """Read the exact MCP tool metadata used to create ``selected_tools``."""
        try:
            from qwen_agent.tools.mcp_manager import MCPManager

            manager = MCPManager()
        except (ImportError, RuntimeError, TypeError):
            return []

        advertised: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for registered_tool in selected_tools:
            client_id = getattr(registered_tool, "client_id", None)
            client = manager.clients.get(client_id)
            for mcp_tool in list(getattr(client, "tools", None) or []):
                name = str(getattr(mcp_tool, "name", "") or "")
                schema = getattr(mcp_tool, "inputSchema", None)
                if not name or not isinstance(schema, dict):
                    continue
                key = (name, self._mcp_schema_hash(schema))
                if key in seen:
                    continue
                seen.add(key)
                advertised.append(
                    {"name": name, "inputSchema": copy.deepcopy(schema)}
                )
        return advertised

    def _run(self, messages, lang="en", knowledge="", **kwargs):
        """Run the original autonomous BC-Plus tool loop.

        The model receives the real MCP schemas on every normal turn and
        retains the provider's automatic tool-choice semantics. This method
        does protocol validation and native message backfill; it never turns
        a search result into a required next call or filters tools by a phase.
        It does reserve the final allowed turn and enters a no-tool final-answer
        mode when the native context budget is nearly exhausted.
        """
        if self.native_api_mode == "responses_harmony":
            yield from self._run_harmony(messages, lang=lang, knowledge=knowledge, **kwargs)
            return
        messages = self._prepend_knowledge_prompt(
            messages=messages, lang=lang, knowledge=knowledge, **kwargs
        )
        messages = copy.deepcopy(messages)
        native_messages = [self._native_message(message) for message in messages]
        self.native_message_history = copy.deepcopy(native_messages)
        response = []
        tool_protocol_retries = 0
        # Keep all MCP-advertised tools visible on every normal turn.  The
        # explicit final-safety turn below intentionally removes them.
        selected_tools = list(self.function_map.values())
        if not selected_tools:
            raise RuntimeError("BCPlusMCPNoAdvertisedTools")
        allowed_names = {tool.name for tool in selected_tools}
        offered_tool_specs = [
            {"name": tool.name, "function": copy.deepcopy(tool.function)}
            for tool in selected_tools
        ]
        native_tool_payload = [
            {"type": "function", "function": copy.deepcopy(func.function)}
            for func in selected_tools
        ]
        self._native_advertised_mcp_tools = self._live_advertised_mcp_tools(
            selected_tools
        )

        while self.interaction_steps < EXPECTED_LIMIT:
            generate_cfg = copy.deepcopy(self._native_generation)
            if kwargs.get("seed") is not None:
                generate_cfg["seed"] = kwargs["seed"]
            pending_force_final = self.native_force_final_pending
            pending_force_reason = self.native_force_final_pending_reason
            forced_final = pending_force_final or (
                self.interaction_steps >= EXPECTED_LIMIT - 1
            )
            forced_final_reason = (
                pending_force_reason
                if pending_force_final
                else ("last_allowed_step" if forced_final else None)
            )
            self.native_force_final_pending = False
            self.native_force_final_pending_reason = None
            if not forced_final:
                context_probe = self._native_context_probe(
                    native_messages,
                    native_tool_payload,
                    generate_cfg,
                )
                if (
                    context_probe is not None
                    and int(context_probe["remaining_tokens"])
                    <= FORCED_FINAL_CONTEXT_THRESHOLD
                ):
                    forced_final = True
                    forced_final_reason = "context_remaining_below_threshold"
            request_messages = copy.deepcopy(native_messages)
            request_tools = copy.deepcopy(native_tool_payload)
            if forced_final:
                request_messages.append(
                    {"role": "user", "content": FORCED_FINAL_INSTRUCTION}
                )
                request_tools = []
                generate_cfg = self._forced_final_generation(
                    generate_cfg, self._native_final_generation
                )
            generation_budget = self._native_generation_budget(
                request_messages,
                request_tools,
                generate_cfg,
                forced_final=forced_final,
                forced_final_reason=forced_final_reason,
            )
            if forced_final:
                self.native_forced_final_count += 1
                self.native_forced_final_reasons.append(str(forced_final_reason))
            if generation_budget is not None:
                # DeepSeek is served through Chat Completions.  Keep the
                # proven Qwen-parity cap as max_tokens and only lower it to
                # the exact remaining context budget for this turn.
                generate_cfg.pop("max_output_tokens", None)
                generate_cfg["max_tokens"] = int(
                    generation_budget["effective_max_tokens"]
                )
            request = {
                "model": self._native_model,
                "messages": request_messages,
                "tool_choice": "none" if forced_final else "auto",
                **generate_cfg,
            }
            # vLLM rejects ``tools=[]`` with HTTP 400.  A forced final turn is
            # deliberately tool-free, so omit the field entirely instead of
            # serializing an empty array.  Normal turns always carry the
            # complete MCP tool payload.
            if request_tools:
                request["tools"] = request_tools

            self._charge_llm()
            completion = self._native_client.chat.completions.create(**request)
            response_evidence = self._native_response_record(completion)
            response_evidence.update(
                {
                    "seed": generate_cfg.get("seed"),
                    "attempt": len(self.native_response_evidence) + 1,
                    "tool_choice_policy": (
                        "forced_final_no_tools" if forced_final else "auto_all_tools"
                    ),
                    "forced_final": forced_final,
                    "forced_final_reason": forced_final_reason,
                    "offered_tool_names": sorted(allowed_names),
                }
            )
            if generation_budget is not None:
                response_evidence["generation_budget"] = copy.deepcopy(
                    generation_budget
                )
            self.native_response_evidence.append(response_evidence)
            native_output = completion.choices[0].message
            content = native_output.content or ""
            reasoning = (
                getattr(native_output, "reasoning_content", None)
                or getattr(native_output, "reasoning", None)
                or ""
            )
            tool_calls = list(getattr(native_output, "tool_calls", None) or [])

            if forced_final and tool_calls:
                self.native_response_evidence[-1][
                    "forced_final_tool_calls_suppressed"
                ] = len(tool_calls)
                tool_calls = []

            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            if (
                not tool_calls
                and not forced_final
                and (
                    finish_reason in {"length", "max_tokens", "max_output_tokens"}
                    or not content.strip()
                )
            ):
                # DeepSeek can spend the ordinary 4096-token budget entirely
                # on reasoning and return no tool call or final content.  Do
                # not persist that truncated thought as the trajectory's
                # final answer.  Spend the next allowed model turn on the
                # existing no-tool final-safety path instead.
                self.native_response_evidence[-1].update(
                    {
                        "recovered_by_forced_final": True,
                        "recovery_reason": (
                            "finish_reason_length_without_tool_call"
                            if finish_reason in {"length", "max_tokens", "max_output_tokens"}
                            else "empty_assistant_turn"
                        ),
                    }
                )
                self.native_force_final_pending = True
                self.native_force_final_pending_reason = (
                    "previous_turn_finish_reason_length"
                    if finish_reason in {"length", "max_tokens", "max_output_tokens"}
                    else "previous_turn_empty_assistant"
                )
                continue

            if not tool_calls:
                # A plain assistant turn is the model's final decision.  It
                # may be rejected by the post-hoc strict validator if it did
                # not satisfy the BC-Plus evidence requirements.
                final_native_message = {
                    "role": "assistant",
                    "content": content,
                }
                if reasoning:
                    final_native_message["reasoning"] = reasoning
                if forced_final:
                    native_messages.append(
                        {"role": "user", "content": FORCED_FINAL_INSTRUCTION}
                    )
                native_messages.append(final_native_message)
                self.native_message_history = copy.deepcopy(native_messages)
                response.append(
                    Message(
                        role="assistant", content=content, reasoning_content=reasoning
                    )
                )
                yield response
                break

            native_assistant = {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [],
            }
            if reasoning:
                native_assistant["reasoning"] = reasoning
            invalid_reason = None
            canonical_tool_names: list[str] = []
            completion_mappings: list[dict] = []
            for call in tool_calls:
                call_id = str(getattr(call, "id", "") or "")
                function = getattr(call, "function", None)
                model_tool_name = str(getattr(function, "name", "") or "")
                function_arguments = str(getattr(function, "arguments", "") or "")
                executed_tool_name, mapping_reason = self._canonicalize_mcp_tool_name(
                    model_tool_name,
                    offered_tool_specs,
                    self._native_advertised_mcp_tools,
                )
                completion_mappings.append(
                    {
                        "model_tool_name": model_tool_name,
                        "executed_tool_name": executed_tool_name,
                        "mapping_reason": mapping_reason,
                        "schema_hash_checked": bool(
                            mapping_reason == "mcp_namespace_suffix_exact_schema"
                        ),
                    }
                )
                function_name = executed_tool_name or model_tool_name
                native_assistant["tool_calls"].append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": function_arguments,
                        },
                    }
                )
                if not call_id:
                    invalid_reason = "tool_call_id_missing"
                    break
                try:
                    call_is_valid, call_reason = self._tool_call_validation(
                        call,
                        allowed_names,
                        canonical_tool_name=executed_tool_name,
                        offered_tools=offered_tool_specs,
                    )
                except (AttributeError, TypeError, ValueError) as exc:
                    call_is_valid = False
                    call_reason = f"tool_call_structure_error_{type(exc).__name__}"
                if not call_is_valid:
                    invalid_reason = call_reason
                    break
                canonical_tool_names.append(function_name)
            self.native_response_evidence[-1]["tool_name_mappings"] = completion_mappings
            self.native_response_evidence[-1][
                "mcp_advertised_tool_names"
            ] = sorted(
                str(item.get("name"))
                for item in self._native_advertised_mcp_tools
            )
            for call, mapping in zip(tool_calls, completion_mappings):
                mapping_record = dict(mapping)
                mapping_record.update(
                    {
                        "attempt": len(self.native_response_evidence),
                        "tool_call_id_sha256": hashlib.sha256(
                            str(getattr(call, "id", "") or "").encode("utf-8")
                        ).hexdigest(),
                    }
                )
                self.native_tool_name_mappings.append(mapping_record)
            native_messages.append(native_assistant)

            if getattr(completion.choices[0], "finish_reason", None) == "length":
                invalid_reason = invalid_reason or "finish_reason_length"
            if invalid_reason is not None:
                self.native_tool_rejection_count += 1
                self.tool_protocol_retry_reasons.append(invalid_reason)
                raise RuntimeError(f"InvalidNativeToolCall:{invalid_reason}")

            tool_protocol_retries = 0
            for call, tool_name, mapping in zip(
                tool_calls, canonical_tool_names, completion_mappings
            ):
                model_tool_name = mapping["model_tool_name"]
                tool_args = str(call.function.arguments)
                self._charge_tool()
                tool_result = self._call_tool(
                    tool_name, tool_args, messages=messages, **kwargs
                )
                model_tool_result, model_tool_evidence = (
                    self._model_context_tool_result(
                        tool_name,
                        tool_result,
                        native_messages=native_messages,
                        tools=native_tool_payload,
                        generation=generate_cfg,
                        tool_call_id=str(call.id),
                    )
                )
                self.native_tool_call_bindings.append(
                    {
                        "tool_call_id": str(call.id),
                        "tool_name": tool_name,
                        "model_tool_name": model_tool_name,
                        "executed_tool_name": tool_name,
                        "mapping_reason": mapping["mapping_reason"],
                        "result_nonempty": bool(str(tool_result).strip()),
                        "model_context_result_truncated": bool(
                            model_tool_evidence.get("truncated")
                        ),
                        "model_context_result_original_chars": model_tool_evidence.get(
                            "original_chars"
                        ),
                        "model_context_result_original_tokens": model_tool_evidence.get(
                            "original_tokens"
                        ),
                        "model_context_result_tokens": model_tool_evidence.get(
                            "model_tokens"
                        ),
                        "arguments_sha256": hashlib.sha256(
                            tool_args.encode("utf-8")
                        ).hexdigest(),
                    }
                )
                assistant_msg = Message(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                    function_call=FunctionCall(name=tool_name, arguments=tool_args),
                )
                function_msg = Message(
                    role=FUNCTION, name=tool_name, content=tool_result
                )
                response.extend([assistant_msg, function_msg])
                messages.extend([assistant_msg, function_msg])
                native_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.id),
                        "content": model_tool_result,
                    }
                )
            self.native_message_history = copy.deepcopy(native_messages)
            yield response

        yield response


def build_messages(query: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": format_query(query, "QUERY_TEMPLATE"),
        }
    ]


def plain_messages(response) -> list[dict]:
    return json.loads(json.dumps(response, ensure_ascii=False, default=str))


def validation_reason(verification: dict) -> str | None:
    """Return a stable, non-content reason for a rejected trajectory."""
    if verification.get("passed"):
        return None
    checks = verification.get("checks") or {}
    failed = sorted(str(key) for key, value in checks.items() if not value)
    return ";".join(failed) or str(verification.get("error_class") or "validation_failed")


def result_records(response: list[dict]) -> tuple[list[dict], dict[str, int]]:
    results: list[dict] = []
    counts: dict[str, int] = {}
    index = 0
    while index < len(response):
        message = response[index]
        role = message.get("role")
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        if role == "assistant" and isinstance(message.get("function_call"), dict):
            function_call = message["function_call"]
            full_name = str(function_call.get("name", ""))
            tool_name = re.sub(r"^search-server-", "", full_name)
            output = None
            if index + 1 < len(response):
                following = response[index + 1]
                if following.get("role") == "function" and following.get("name") == full_name:
                    output = following.get("content", "")
                    index += 1
            results.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "arguments": function_call.get("arguments"),
                    "output": output,
                }
            )
            counts[tool_name] = counts.get(tool_name, 0) + 1
        elif role == "assistant":
            # Native reasoning and visible answer text may coexist in the
            # same assistant message.  Preserve both in order so the strict
            # gate sees the visible final answer instead of treating a
            # reasoning-only projection as the last result.
            if reasoning and isinstance(reasoning, str) and reasoning.strip():
                results.append(
                    {
                        "type": "reasoning",
                        "tool_name": None,
                        "arguments": None,
                        "output": [reasoning.strip()],
                    }
                )
            if isinstance(content, str) and content.strip():
                results.append(
                    {
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": content.strip(),
                    }
                )
        index += 1
    return results, counts


def token_counts_from_evidence(evidence: list[dict] | None) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "reported_turns": 0}
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        totals["reported_turns"] += 1
        for source, target in (
            ("prompt_tokens", "prompt_tokens"),
            ("output_tokens", "output_tokens"),
            ("reasoning_tokens", "reasoning_tokens"),
        ):
            value = item.get(source)
            if isinstance(value, (int, float)):
                totals[target] += int(value)
    return totals


def write_structure_audit(
    combination_root: Path,
    record: dict,
    verification: dict,
) -> None:
    """Persist evidence metadata without copying prompt/document/answer bodies."""
    trajectory_path = combination_root / "trajectories" / f"{record.get('query_id')}.json"
    raw_response = record.get("raw_response") or []
    structure = []
    for message in raw_response:
        if not isinstance(message, dict):
            continue
        function_call = message.get("function_call")
        structure.append(
            {
                "role": message.get("role"),
                "name": message.get("name"),
                "function_call_name": function_call.get("name") if isinstance(function_call, dict) else None,
                "has_function_call": isinstance(function_call, dict),
                "has_content": bool(str(message.get("content", "")).strip()),
                "content_chars": len(str(message.get("content", ""))),
                "has_reasoning": bool(str(message.get("reasoning_content", "")).strip()),
                "has_tool_result": message.get("role") == "function" and bool(str(message.get("content", "")).strip()),
            }
        )
    final_answer = str(record.get("final_answer", ""))
    audit = {
        "schema_version": 1,
        "audit_type": "structure_only",
        "trial_id": os.environ.get("BCP_TRIAL_ID"),
        "query_id": record.get("query_id"),
        "profile": (record.get("metadata") or {}).get("profile"),
        "retriever_slug": (record.get("metadata") or {}).get("retriever_slug"),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest() if trajectory_path.is_file() else None,
        "trajectory_bytes": trajectory_path.stat().st_size if trajectory_path.is_file() else None,
        "status": record.get("status"),
        "terminated_by": record.get("terminated_by"),
        "tool_call_counts": record.get("tool_call_counts") or {},
        "llm_call_count": record.get("llm_call_count", 0),
        "executed_tool_call_count": record.get("executed_tool_call_count", 0),
        "interaction_step_count": record.get("interaction_step_count", 0),
        "timing": record.get("timing") or {},
        "token_counts": token_counts_from_evidence((record.get("metadata") or {}).get("native_response_evidence")),
        "final_answer_present": bool(final_answer.strip()),
        "final_answer_sha256": hashlib.sha256(final_answer.encode("utf-8")).hexdigest() if final_answer else None,
        "raw_message_count": len(structure),
        "raw_message_structure": structure,
        "validation_passed": bool(verification.get("passed")),
        "max_step_termination": verification.get("max_step_termination"),
        "validation_reason": validation_reason(verification),
        "validation_checks": verification.get("checks") or {},
        "classification_history": record.get("classification_history") or [],
        "generated_at": now(),
    }
    atomic_write_json(combination_root / "audits" / f"{record.get('query_id')}.json", audit)


def find_config_entry(key: str, value: str, entries: list[dict]) -> dict:
    matches = [entry for entry in entries if entry.get(key) == value]
    if len(matches) != 1:
        raise RuntimeError(f"expected one config entry for {key}={value}")
    return matches[0]


def _resolve_deepseek_mode(profile: str, generation: dict) -> tuple[str | None, dict]:
    """Apply the installed official DeepSeek-V4 mode switch only.

    The autonomous loop, tool exposure, and ``tool_choice=auto`` semantics
    are intentionally untouched.  The local vLLM DeepSeek-V4 tokenizer
    wrapper maps the recorded kwargs to its internal ``thinking_mode`` value.
    """
    if profile != "deepseek-v4-flash":
        return None, {"source": "not_applicable"}
    requested = os.environ.get("BCP_DEEPSEEK_MODE", "thinking").strip().lower()
    requested = {"non-think": "chat", "nonthink": "chat"}.get(requested, requested)
    if requested not in {"thinking", "chat"}:
        raise ValueError(
            "BCP_DEEPSEEK_MODE must be one of thinking, chat, nonthink"
        )
    if requested == "chat":
        extra_body = copy.deepcopy(generation.get("extra_body") or {})
        template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
        # Exact kwargs consumed by the installed official
        # vllm/tokenizers/deepseek_v4.py wrapper: reasoning_effort=none
        # selects chat mode and thinking=False makes the request explicit.
        template_kwargs.update(
            {
                "thinking": False,
                "reasoning_effort": "none",
            }
        )
        extra_body["chat_template_kwargs"] = template_kwargs
        generation["extra_body"] = extra_body
    return requested, {
        "source": "BCP_DEEPSEEK_MODE",
        "requested": requested,
        "official_wrapper_mapping": {
            "thinking": requested == "thinking",
            "reasoning_effort": "high" if requested == "thinking" else "none",
            "vllm_internal_thinking_mode": (
                "thinking" if requested == "thinking" else "chat"
            ),
        },
        "chat_template": "not_used",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--retriever", required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--model-server", required=True)
    parser.add_argument("--mcp-url", required=True)
    parser.add_argument("--combination-root", required=True)
    parser.add_argument(
        "--protocol-strategy",
        choices=["configured"],
        default=None,
    )
    args = parser.parse_args()

    combination_root = Path(args.combination_root).resolve()
    expected_parent = Path(CONFIG["output_root"]).resolve()
    if expected_parent not in combination_root.parents:
        raise RuntimeError("combination output is outside campaign root")
    trajectory_path = combination_root / "trajectories" / f"{args.query_id}.json"
    query_status_path = combination_root / "status" / "queries" / f"{args.query_id}.json"

    existing = validate_trajectory(
        trajectory_path,
        expected_query_id=args.query_id,
        expected_profile=args.profile,
        expected_retriever_slug=args.retriever,
    )
    if existing.get("passed"):
        return 0

    model = find_config_entry("profile", args.profile, CONFIG["models"])
    retriever = find_config_entry("slug", args.retriever, CONFIG["retrievers"])
    queries = dict(query_rows(CONFIG))
    if args.query_id not in queries:
        raise RuntimeError("query ID is absent from the exact 830-query source")
    fixed_query_index = QUERY_INDEX[args.query_id]
    previous = read_json(query_status_path, {}) or {}
    attempt = int(previous.get("attempt", 0)) + 1
    started_at = now()
    started_monotonic = time.monotonic()
    atomic_write_json(
        query_status_path,
        {
            "query_id": args.query_id,
            "state": "running",
            "attempt": attempt,
            "started_at": started_at,
            "profile": args.profile,
            "retriever_slug": args.retriever,
            "max_interaction_steps": EXPECTED_LIMIT,
        },
    )

    response = []
    assistant: CampaignAssistant | None = None
    deepseek_mode: str | None = None
    deepseek_mode_evidence: dict = {"source": "not_initialized"}
    mcp_init_lease = MCPInitLease(os.environ.get(MCP_INIT_GATE_ENV))
    try:
        generation = {
            "max_tokens": int(CONFIG["max_output_tokens"]),
            **copy.deepcopy(model["generation"]),
        }
        deepseek_mode, deepseek_mode_evidence = _resolve_deepseek_mode(
            args.profile, generation
        )
        atomic_write_json(
            query_status_path,
            {
                "query_id": args.query_id,
                "state": "running",
                "attempt": attempt,
                "started_at": started_at,
                "profile": args.profile,
                "retriever_slug": args.retriever,
                "max_interaction_steps": EXPECTED_LIMIT,
                "deepseek_mode": deepseek_mode,
                "deepseek_mode_evidence": deepseek_mode_evidence,
            },
        )
        llm_config = {
            "model": model["served_model_name"],
            "model_server": args.model_server,
            "api_key": "EMPTY",
            "generate_cfg": generation,
        }
        # Gate only the MCP/SSE handshake. It is released before the first
        # model request, so all configured workers can feed the model queue.
        with mcp_init_lease:
            assistant = CampaignAssistant(
                llm=llm_config,
                function_list=[{"mcpServers": {"search-server": {"url": args.mcp_url}}}],
                native_model=model["served_model_name"],
                native_server=args.model_server,
                native_generation=generation,
                native_final_generation=copy.deepcopy(model.get("final_generation", {})),
                native_protocol_strategy=(
                    args.protocol_strategy
                    if args.protocol_strategy and args.protocol_strategy != "configured"
                    else model.get("protocol_strategy")
                ),
                native_api_mode=(
                    "responses_harmony"
                    if model.get("official_backend", {}).get("harmony_required")
                    else "chat_completions"
                ),
                native_max_output_tokens=int(CONFIG["max_output_tokens"]),
                native_document_tokenizer_path=model["model_path"],
                native_context_budget=(
                    {
                        "model_path": model["model_path"],
                        "max_tokens_cap": MAX_NATIVE_GENERATION_TOKENS,
                        "safety_margin": DEEPSEEK_CONTEXT_SAFETY_MARGIN,
                        "tokenizer_mode": "deepseek_v4",
                        "encoding_path": str(
                            Path(model["environment_path"])
                            / "lib"
                            / f"python{sys.version_info.major}.{sys.version_info.minor}"
                            / "site-packages"
                            / "vllm"
                            / "tokenizers"
                            / "deepseek_v4_encoding.py"
                        ),
                    }
                    if args.profile == "deepseek-v4-flash"
                    else None
                ),
            )
        messages = build_messages(queries[args.query_id])
        for chunk in assistant.run(messages=messages):
            response = chunk
            if assistant.interaction_steps > EXPECTED_LIMIT:
                raise InteractionLimitExceeded("LLM turn counter exceeded 50")

        raw_response = plain_messages(response)
        results, counts = result_records(raw_response)
        last_message = raw_response[-1] if raw_response else {}
        final_answer = (
            str(last_message.get("content", "")).strip()
            if last_message.get("role") == "assistant"
            and not last_message.get("function_call")
            else ""
        )
        completed = bool(
            final_answer
            and all(
                item.get("output") is not None
                for item in results
                if item.get("type") == "tool_call"
            )
        )
        finished_at = now()
        duration = time.monotonic() - started_monotonic
        record = {
            "metadata": {
                "campaign_id": CONFIG["campaign_id"],
                "batch_id": os.environ.get("BCP_BATCH_ID"),
                "profile": args.profile,
                "fixed_query_count": len(QUERY_ROWS),
                "fixed_query_index": fixed_query_index,
                "query_source_path": str(QUERY_SOURCE_PATH),
                "query_source_sha256": QUERY_SOURCE_SHA256,
                "model_repo_id": model["repo_id"],
                "served_model_name": model["served_model_name"],
                "retriever_repo_id": retriever["repo_id"],
                "retriever_slug": retriever["slug"],
                "index_path": retriever["index_path"],
                "retriever_model_path": retriever["model_path"],
                "corpus_path": retriever["corpus_path"],
                "tool_api_mode": "native_openai_real_mcp",
                "model_endpoint": args.model_server,
                "mcp_endpoint": args.mcp_url,
                "deepseek_mode": deepseek_mode,
                "deepseek_mode_evidence": deepseek_mode_evidence,
                "max_interaction_steps": EXPECTED_LIMIT,
                "qwen_agent_settings_limit": int(qwen_settings.MAX_LLM_CALL_PER_RUN),
                "qwen_agent_fncall_limit": int(fncall_agent.MAX_LLM_CALL_PER_RUN),
                "max_output_tokens": int(CONFIG["max_output_tokens"]),
                "native_generation_budget_cap": MAX_NATIVE_GENERATION_TOKENS,
                "native_context_safety_margin": DEEPSEEK_CONTEXT_SAFETY_MARGIN,
                "forced_final_context_threshold": FORCED_FINAL_CONTEXT_THRESHOLD,
                "forced_final_max_tokens": FORCED_FINAL_MAX_TOKENS,
                "forced_final_safety_margin": FORCED_FINAL_SAFETY_MARGIN,
                "get_document_max_tokens": GET_DOCUMENT_MAX_TOKENS,
                "native_forced_final_count": assistant.native_forced_final_count,
                "native_forced_final_reasons": assistant.native_forced_final_reasons,
                "native_context_limit_evidence": assistant.native_context_limit_evidence,
                "native_tokenizer_evidence": assistant.native_tokenizer_evidence,
                "native_generation_budget_evidence": assistant.native_generation_budget_evidence,
                "native_tool_context_max_chars": None,
                "native_tool_context_truncation_count": assistant.model_context_truncation_count,
                "native_tool_context_truncations": assistant.native_tool_context_truncations,
                "native_history_compaction_count": assistant.native_history_compaction_count,
                "native_protocol_retry_count": assistant.tool_protocol_retry_count,
                "malformed_tool_call_retry_count": assistant.malformed_tool_call_retry_count,
                "native_tool_rejection_count": assistant.native_tool_rejection_count,
                "tool_protocol_retry_reasons": assistant.tool_protocol_retry_reasons,
                "native_tool_call_bindings": assistant.native_tool_call_bindings,
                "native_tool_name_mappings": assistant.native_tool_name_mappings,
                "native_advertised_mcp_tools": [
                    {
                        "name": str(item.get("name")),
                        "schema_sha256": CampaignAssistant._mcp_schema_hash(
                            item.get("inputSchema") or {}
                        ),
                        "required": sorted(
                            (item.get("inputSchema") or {}).get("required") or []
                        ),
                        "property_names": sorted(
                            ((item.get("inputSchema") or {}).get("properties") or {}).keys()
                        ),
                    }
                    for item in assistant._native_advertised_mcp_tools
                ],
                "native_protocol_strategy": assistant.native_protocol_strategy,
                "native_requested_protocol_strategy": assistant.native_requested_protocol_strategy,
                "native_api_mode": assistant.native_api_mode,
                "native_tool_choice_policy": "auto_all_tools",
                "mcp_init_gate": {
                    "path": os.environ.get(MCP_INIT_GATE_ENV),
                    "limit": mcp_init_lease.limit,
                    "wait_seconds": mcp_init_lease.wait_seconds,
                    "hold_seconds": mcp_init_lease.hold_seconds,
                    "graceful_shutdown_timeout": mcp_init_lease.shutdown_timeout,
                    "graceful_shutdown_error_class": mcp_init_lease.shutdown_error_class,
                },
                "trial_id": os.environ.get("BCP_TRIAL_ID"),
                "final_answer_source": "model_assistant_natural",
                "synthetic_final": False,
                "localhost_proxy_bypass": os.environ.get("NO_PROXY") == "127.0.0.1,localhost,::1" and os.environ.get("no_proxy") == "127.0.0.1,localhost,::1",
                "native_response_evidence": assistant.native_response_evidence,
                "final_markup_repair_count": assistant.final_markup_repair_count,
                "complete_tool_returns_persisted": True,
                "model_transcript_tool_results_bounded": True,
                "generation": generation,
                "final_generation": model.get("final_generation", {}),
                "attempt": attempt,
            },
            "query_id": args.query_id,
            "status": "completed" if completed else "incomplete",
            "tool_call_counts": counts,
            "llm_call_count": assistant.llm_call_count,
            "executed_tool_call_count": assistant.executed_tool_call_count,
            "interaction_step_count": assistant.interaction_steps,
            "final_answer_present": bool(final_answer),
            "final_answer": final_answer,
            "retrieved_docids": extract_retrieved_docids_from_result(results),
            "timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration,
            },
            "error_class": None if completed else "IncompleteTrajectory",
            "result": results,
            "raw_response": raw_response,
            "native_message_history": copy.deepcopy(
                assistant.native_message_history
            ),
        }
        atomic_write_json(trajectory_path, record)
        verification = validate_trajectory(
            trajectory_path,
            expected_query_id=args.query_id,
            expected_profile=args.profile,
            expected_retriever_slug=args.retriever,
        )
        max_step_accepted = bool(
            (verification.get("max_step_termination") or {}).get("passed")
            and verification.get("termination_reason") == "max_steps_limit"
        )
        previous_reason = ";".join(
            sorted(
                str(key)
                for key, value in (verification.get("checks") or {}).items()
                if not value
            )
        ) or None
        if max_step_accepted:
            previous_classification = {
                "policy": MAX_STEP_POLICY,
                "classified_at": now(),
                "attempt": attempt,
                "prior_state": {
                    "status": record.get("status"),
                    "error_class": record.get("error_class"),
                    "validation_reason": previous_reason,
                    "strict_validation": copy.deepcopy(verification),
                },
                "new_state": {
                    "status": "completed",
                    "terminated_by": "max_steps_limit",
                    "error_class": None,
                },
            }
            record["classification_history"] = [
                *(record.get("classification_history") or []),
                previous_classification,
            ]
            record["status"] = "completed"
            record["terminated_by"] = "max_steps_limit"
            record["error_class"] = None
            # The original incomplete/error classification is retained above;
            # no synthetic final answer is added to the raw trajectory.
            atomic_write_json(trajectory_path, record)
            verification = validate_trajectory(
                trajectory_path,
                expected_query_id=args.query_id,
                expected_profile=args.profile,
                expected_retriever_slug=args.retriever,
            )
        reason = validation_reason(verification)
        record["strict_validation"] = verification
        record["validation_reason"] = reason
        atomic_write_json(trajectory_path, record)
        write_structure_audit(combination_root, record, verification)
        state = "completed" if verification.get("passed") else "failed_retryable"
        query_status = {
            "query_id": args.query_id,
            "state": state,
            "attempt": attempt,
            "profile": args.profile,
            "retriever_slug": args.retriever,
            "trajectory_path": str(trajectory_path),
            "tool_call_counts": counts,
            "final_answer_present": bool(final_answer),
            "llm_call_count": assistant.llm_call_count,
            "executed_tool_call_count": assistant.executed_tool_call_count,
            "interaction_step_count": assistant.interaction_steps,
            "duration_seconds": duration,
            "error_class": None if state == "completed" else "IncompleteTrajectory",
            "verification_passed": bool(verification.get("passed")),
            "validation_reason": reason,
            "strict_validation": verification,
        }
        if max_step_accepted:
            query_status.update(
                {
                    "terminated_by": "max_steps_limit",
                    "prior_state": "incomplete",
                    "prior_error_class": "IncompleteTrajectory",
                    "prior_validation_reason": previous_reason,
                    "classification_history": record.get("classification_history") or [],
                }
            )
        atomic_write_json(query_status_path, query_status)
        return 0 if state == "completed" else 2
    except Exception as exc:
        finished_at = now()
        duration = time.monotonic() - started_monotonic
        raw_response = plain_messages(response) if response else []
        results, counts = result_records(raw_response)
        error_class = "context_exhausted" if isinstance(exc, ContextExhausted) else type(exc).__name__
        reason = "context_exhausted" if isinstance(exc, ContextExhausted) else f"service_or_worker_error:{error_class}"
        failure_record = {
            "metadata": {
                "campaign_id": CONFIG["campaign_id"],
                "batch_id": os.environ.get("BCP_BATCH_ID"),
                "profile": args.profile,
                "fixed_query_count": len(QUERY_ROWS),
                "fixed_query_index": fixed_query_index,
                "query_source_path": str(QUERY_SOURCE_PATH),
                "query_source_sha256": QUERY_SOURCE_SHA256,
                "model_repo_id": model["repo_id"],
                "retriever_repo_id": retriever["repo_id"],
                "retriever_slug": retriever["slug"],
                "index_path": retriever["index_path"],
                "retriever_model_path": retriever["model_path"],
                "corpus_path": retriever["corpus_path"],
                "tool_api_mode": "native_openai_real_mcp",
                "model_endpoint": args.model_server,
                "mcp_endpoint": args.mcp_url,
                "deepseek_mode": deepseek_mode,
                "deepseek_mode_evidence": deepseek_mode_evidence,
                "max_interaction_steps": EXPECTED_LIMIT,
                "qwen_agent_settings_limit": int(qwen_settings.MAX_LLM_CALL_PER_RUN),
                "qwen_agent_fncall_limit": int(fncall_agent.MAX_LLM_CALL_PER_RUN),
                "max_output_tokens": int(CONFIG["max_output_tokens"]),
                "native_generation_budget_cap": MAX_NATIVE_GENERATION_TOKENS,
                "native_context_safety_margin": DEEPSEEK_CONTEXT_SAFETY_MARGIN,
                "forced_final_context_threshold": FORCED_FINAL_CONTEXT_THRESHOLD,
                "forced_final_max_tokens": FORCED_FINAL_MAX_TOKENS,
                "forced_final_safety_margin": FORCED_FINAL_SAFETY_MARGIN,
                "get_document_max_tokens": GET_DOCUMENT_MAX_TOKENS,
                "native_forced_final_count": assistant.native_forced_final_count if assistant else 0,
                "native_forced_final_reasons": assistant.native_forced_final_reasons if assistant else [],
                "native_context_limit_evidence": assistant.native_context_limit_evidence if assistant else {},
                "native_tokenizer_evidence": assistant.native_tokenizer_evidence if assistant else {},
                "native_generation_budget_evidence": assistant.native_generation_budget_evidence if assistant else [],
                "native_tool_context_max_chars": None,
                "native_tool_context_truncation_count": assistant.model_context_truncation_count if assistant else 0,
                "native_tool_context_truncations": assistant.native_tool_context_truncations if assistant else [],
                "native_history_compaction_count": assistant.native_history_compaction_count if assistant else 0,
                "native_protocol_retry_count": assistant.tool_protocol_retry_count if assistant else 0,
                "malformed_tool_call_retry_count": assistant.malformed_tool_call_retry_count if assistant else 0,
                "native_tool_rejection_count": assistant.native_tool_rejection_count if assistant else 0,
                "tool_protocol_retry_reasons": assistant.tool_protocol_retry_reasons if assistant else [],
                "native_tool_call_bindings": assistant.native_tool_call_bindings if assistant else [],
                "native_tool_name_mappings": assistant.native_tool_name_mappings if assistant else [],
                "native_advertised_mcp_tools": [
                    {
                        "name": str(item.get("name")),
                        "schema_sha256": CampaignAssistant._mcp_schema_hash(
                            item.get("inputSchema") or {}
                        ),
                        "required": sorted(
                            (item.get("inputSchema") or {}).get("required") or []
                        ),
                        "property_names": sorted(
                            ((item.get("inputSchema") or {}).get("properties") or {}).keys()
                        ),
                    }
                    for item in (assistant._native_advertised_mcp_tools if assistant else [])
                ],
                "native_protocol_strategy": assistant.native_protocol_strategy if assistant else None,
                "native_requested_protocol_strategy": assistant.native_requested_protocol_strategy if assistant else None,
                "native_api_mode": assistant.native_api_mode if assistant else None,
                "native_tool_choice_policy": "auto_all_tools",
                "mcp_init_gate": {
                    "path": os.environ.get(MCP_INIT_GATE_ENV),
                    "limit": mcp_init_lease.limit,
                    "wait_seconds": mcp_init_lease.wait_seconds,
                    "hold_seconds": mcp_init_lease.hold_seconds,
                    "graceful_shutdown_timeout": mcp_init_lease.shutdown_timeout,
                    "graceful_shutdown_error_class": mcp_init_lease.shutdown_error_class,
                },
                "trial_id": os.environ.get("BCP_TRIAL_ID"),
                "localhost_proxy_bypass": os.environ.get("NO_PROXY") == "127.0.0.1,localhost,::1" and os.environ.get("no_proxy") == "127.0.0.1,localhost,::1",
                "native_response_evidence": assistant.native_response_evidence if assistant else [],
                "final_markup_repair_count": assistant.final_markup_repair_count if assistant else 0,
                "complete_tool_returns_persisted": True,
                "model_transcript_tool_results_bounded": True,
                "attempt": attempt,
            },
            "query_id": args.query_id,
            "status": "failed_retryable",
            "tool_call_counts": counts,
            "llm_call_count": assistant.llm_call_count if assistant else 0,
            "executed_tool_call_count": assistant.executed_tool_call_count if assistant else 0,
            "interaction_step_count": assistant.interaction_steps if assistant else 0,
            "final_answer_present": False,
            "final_answer": "",
            "retrieved_docids": extract_retrieved_docids_from_result(results),
            "timing": {
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_seconds": duration,
            },
            "error_class": error_class,
            "error_message_sha256": hashlib.sha256(
                str(exc).encode("utf-8")
            ).hexdigest(),
            "error_message_len": len(str(exc)),
            "error_errno": getattr(exc, "errno", None),
            "error_strerror": getattr(exc, "strerror", None),
            "error_message_sanitized": str(exc)[:512],
            "validation_reason": reason,
            "result": results,
            "raw_response": raw_response,
            "native_message_history": copy.deepcopy(
                assistant.native_message_history if assistant else []
            ),
        }
        atomic_write_json(trajectory_path, failure_record)
        write_structure_audit(
            combination_root,
            failure_record,
            {
                "passed": False,
                "error_class": error_class,
                "checks": {"service_or_worker_error": False},
            },
        )
        atomic_write_json(
            query_status_path,
            {
                "query_id": args.query_id,
                "state": "failed_retryable",
                "attempt": attempt,
                "profile": args.profile,
                "retriever_slug": args.retriever,
                "trajectory_path": str(trajectory_path),
                "tool_call_counts": counts,
                "final_answer_present": False,
                "duration_seconds": duration,
                "started_at": started_at,
                "finished_at": finished_at,
                "error_class": error_class,
                "error_errno": getattr(exc, "errno", None),
                "error_strerror": getattr(exc, "strerror", None),
                "error_message_sanitized": str(exc)[:512],
                "validation_reason": reason,
                "traceback_fingerprint": str(
                    abs(hash("".join(traceback.format_exception_only(type(exc), exc))))
                ),
            },
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
