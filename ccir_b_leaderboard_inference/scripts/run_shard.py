#!/usr/bin/env python3
"""Run one deterministic shard of the CCIR-B BC-Plus campaign.

Sharding follows the row order in ``queries.tsv`` rather than numeric query
IDs.  Multiple hosts may use the same output root because every shard owns a
disjoint row interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "campaign.example.json"

# ``run_shard.py`` is launched from ``batch-runs/<run-doc>``.  The validator
# and other campaign helpers live one level above that directory, so make the
# task-owned campaign root importable before any worker result is verified.
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_queries(path: Path) -> list[tuple[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [(row[0].strip(), row[1].strip()) for row in csv.reader(handle, delimiter="\t")]
    if len(rows) != 830 or len({query_id for query_id, _ in rows}) != 830:
        raise RuntimeError(f"expected 830 unique query rows, got {len(rows)}")
    if any(not query_id or not query for query_id, query in rows):
        raise RuntimeError("queries.tsv contains an empty query id or query text")
    return rows


def local_http_json(url: str, timeout: float = 5.0) -> dict[str, Any]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read(4 * 1024 * 1024).decode("utf-8"))


def model_profile(config: dict[str, Any], profile: str) -> dict[str, Any]:
    matches = [item for item in config["models"] if str(item["profile"]) == profile]
    if len(matches) != 1:
        raise RuntimeError(f"expected one model profile {profile!r}, got {len(matches)}")
    return matches[0]


def check_prerequisites(config: dict[str, Any], profile: str) -> None:
    model = model_profile(config, profile)
    endpoint = str(model["model_endpoint"])
    served = str(model["served_model_name"])
    models = local_http_json(endpoint.rstrip("/") + "/models")
    ids = {str(item.get("id")) for item in models.get("data", []) if isinstance(item, dict)}
    if served not in ids:
        raise RuntimeError(f"local model endpoint does not serve {served}: {sorted(ids)}")
    for slug, port in (config.get("mcp_ports") or {}).items():
        sock = socket.create_connection(("127.0.0.1", int(port)), timeout=3)
        sock.close()
        print(f"MCP_READY {slug} port={port}", flush=True)
    print(f"MODEL_READY {served} endpoint={endpoint}", flush=True)


def verify_trajectory(
    path: Path, query_id: str, slug: str, profile: str
) -> tuple[bool, str | None]:
    if not path.is_file():
        return False, "missing_trajectory"
    try:
        payload = load_json(path, {}) or {}
        if payload.get("error_class") == "context_exhausted":
            return False, "context_exhausted"
        # Import after the environment/config path is set by main().
        from campaign_common import validate_trajectory

        verification = validate_trajectory(
            path,
            expected_query_id=query_id,
            expected_profile=profile,
            expected_retriever_slug=slug,
        )
        if verification.get("passed"):
            return True, None
        return False, str(payload.get("error_class") or payload.get("validation_reason") or "invalid_trajectory")
    except Exception as exc:
        return False, type(exc).__name__


def run_one(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_id: str,
    profile: str,
    query_id: str,
    slug: str,
    attempt_limit: int,
    worker_timeout: int,
    shard_status_root: Path,
) -> dict[str, Any]:
    model = model_profile(config, profile)
    output_root = Path(config["output_root"])
    group_root = output_root / run_id / profile / slug
    trajectory_path = group_root / "trajectories" / f"{query_id}.json"
    status_path = group_root / "status" / "queries" / f"{query_id}.json"
    log_path = shard_status_root / "logs" / slug / f"query-{query_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    valid, reason = verify_trajectory(trajectory_path, query_id, slug, profile)
    if valid:
        return {"query_id": query_id, "retriever_slug": slug, "state": "skipped_valid", "attempt": 0}
    if reason == "context_exhausted":
        return {
            "query_id": query_id,
            "retriever_slug": slug,
            "state": "permanent_failure",
            "attempt": 0,
            "reason": reason,
        }

    gate_path = Path("/tmp") / f"bcplus-{run_id}-shard-mcp-gate.json"
    last_reason = reason
    for attempt in range(1, attempt_limit + 1):
        environment = os.environ.copy()
        environment.update(
            {
                "BCP_RUN_ID": run_id,
                "BCP_TRIAL_ID": run_id,
                "BCP_BATCH_ID": run_id,
                "BCP_CAMPAIGN_CONFIG": str(config_path),
                "BCP_MAX_TOOL_STEPS": "50",
                "BCP_MAX_INTERACTION_STEPS": "50",
                "QWEN_AGENT_MAX_LLM_CALL_PER_RUN": "50",
                "BCP_MCP_INIT_GATE": str(gate_path),
                "BCP_MCP_INIT_GATE_TIMEOUT_SECONDS": "180",
                "NO_PROXY": "127.0.0.1,localhost,::1",
                "no_proxy": "127.0.0.1,localhost,::1",
            }
        )
        command = [
            str(config.get("search_python") or sys.executable),
            str(SEARCH_AGENT_ROOT / "query_worker.py"),
            "--profile",
            profile,
            "--retriever",
            slug,
            "--query-id",
            query_id,
            "--model-server",
            str(model["model_endpoint"]),
            "--mcp-url",
            f"http://127.0.0.1:{int(config['mcp_ports'][slug])}/mcp",
            "--combination-root",
            str(group_root),
        ]
        started = time.monotonic()
        return_code: int | None = None
        timed_out = False
        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(
                command,
                cwd=PACKAGE_ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=worker_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    return_code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    return_code = process.wait(timeout=30)
        valid, reason = verify_trajectory(trajectory_path, query_id, slug, profile)
        elapsed = time.monotonic() - started
        if valid:
            return {
                "query_id": query_id,
                "retriever_slug": slug,
                "state": "completed",
                "attempt": attempt,
                "duration_seconds": round(elapsed, 3),
                "return_code": return_code,
            }
        last_reason = reason
        # The campaign policy is to stop retrying a context-exhausted turn;
        # it needs a forced-final policy change, not another identical prompt.
        if reason == "context_exhausted":
            break
    return {
        "query_id": query_id,
        "retriever_slug": slug,
        "state": "permanent_failure" if last_reason == "context_exhausted" else "failed",
        "attempt": attempt,
        "reason": last_reason,
        "duration_seconds": round(time.monotonic() - started, 3),
        "return_code": return_code,
        "timed_out": timed_out,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--attempt-limit", type=int, default=5)
    parser.add_argument("--worker-timeout", type=int, default=900)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if (
        args.concurrency < 1
        or args.attempt_limit < 1
        or args.worker_timeout < 1
        or args.shard_count < 1
        or not 0 <= args.shard_index < args.shard_count
    ):
        raise SystemExit("invalid shard, concurrency, attempt-limit, or worker-timeout")

    config_path = Path(args.config).resolve()
    os.environ["BCP_CAMPAIGN_CONFIG"] = str(config_path)
    from campaign_common import load_config

    config = load_config()
    check_prerequisites(config, args.profile)
    rows = read_queries(Path(config["query_path"]))
    split_size = (len(rows) + args.shard_count - 1) // args.shard_count
    start = args.shard_index * split_size
    end = min(len(rows), start + split_size)
    if start >= end:
        raise RuntimeError(
            f"shard {args.shard_index}/{args.shard_count} owns no query rows"
        )
    owned_queries = [query_id for query_id, _ in rows[start:end]]
    retrievers = [str(item["slug"]) for item in config["retrievers"]]
    tasks = [(query_id, slug) for query_id in owned_queries for slug in retrievers]
    output_root = Path(config["output_root"])
    status_root = output_root / args.run_id / "status" / f"shard-{args.shard_index}-of-{args.shard_count}"
    status_path = output_root / args.run_id / "status" / f"shard-{args.shard_index}-of-{args.shard_count}.json"
    lock = threading.Lock()
    results: list[dict[str, Any]] = []

    def write_status(active: int = 0) -> None:
        counts = {}
        for item in results:
            counts[item["state"]] = counts.get(item["state"], 0) + 1
        payload = {
            "run_id": args.run_id,
            "profile": args.profile,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "query_index_start_inclusive": start,
            "query_index_end_exclusive": end,
            "query_count": len(owned_queries),
            "retriever_count": len(retrievers),
            "target_trajectories": len(tasks),
            "concurrency": args.concurrency,
            "max_interaction_steps": 50,
            "max_output_tokens": 4096,
            "counts": counts,
            "active": active,
            "updated_at": time.time(),
        }
        atomic_json(status_path, payload)

    print(
        f"SHARD {args.shard_index}/{args.shard_count}: query indexes {start}:{end}, "
        f"ids {owned_queries[0]}..{owned_queries[-1]}, tasks={len(tasks)}, "
        f"concurrency={args.concurrency}",
        flush=True,
    )
    write_status(active=0)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(
                run_one,
                config=config,
                config_path=config_path,
                run_id=args.run_id,
                profile=args.profile,
                query_id=query_id,
                slug=slug,
                attempt_limit=args.attempt_limit,
                worker_timeout=args.worker_timeout,
                shard_status_root=status_root,
            ): (query_id, slug)
            for query_id, slug in tasks
        }
        while futures:
            done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                query_id, slug = futures.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "query_id": query_id,
                        "retriever_slug": slug,
                        "state": "launcher_error",
                        "reason": type(exc).__name__,
                    }
                with lock:
                    results.append(result)
                    write_status(active=min(args.concurrency, len(futures)))
                print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    write_status(active=0)
    failed = sum(item["state"] in {"failed", "permanent_failure", "launcher_error"} for item in results)
    print(f"SHARD_DONE total={len(results)} failed={failed} status={status_path}", flush=True)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
