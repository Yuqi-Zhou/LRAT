#!/usr/bin/env python3
"""Resume BrowseComp-Plus A-leaderboard inference with existing Qwen3.5 vLLM servers.

This runner is intended for the fast evaluation phase after the retriever model
has already been downloaded and indexed. It assumes Qwen3.5-4B OpenAI-compatible
vLLM servers are already listening on ports 8001-8004.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


GROUPS = [("0,1", 8001), ("2,3", 8002), ("4,5", 8003), ("6,7", 8004)]


def safe_name(repo: str) -> str:
    return repo.strip().replace("/", "__")


def log_line(log_file: Path, msg: str) -> None:
    line = time.strftime("[%Y-%m-%d %H:%M:%S] ") + msg
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_http(url: str, timeout_s: int) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if 200 <= response.status < 500:
                    return True
        except Exception:
            time.sleep(3)
    return False


def split_queries(query_file: Path, out_dir: Path, workers: int) -> list[Path]:
    rows: list[tuple[str, str]] = []
    with query_file.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) == 2:
                rows.append((row[0].strip(), row[1].strip()))
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [[] for _ in range(workers)]
    for i, row in enumerate(rows):
        chunks[i % workers].append(row)
    paths: list[Path] = []
    for i, chunk in enumerate(chunks):
        path = out_dir / f"queries.worker{i:02d}.tsv"
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f, delimiter="\t").writerows(chunk)
        paths.append(path)
    return paths


def preseed_results(runs_root: Path, resume_dir: Path) -> int:
    """Copy best previous result for each query_id into resume_dir."""
    by_qid: dict[str, tuple[tuple[int, float], Path]] = {}
    if not runs_root.exists():
        return 0
    for directory in sorted(runs_root.iterdir(), key=lambda p: p.stat().st_mtime):
        if not directory.is_dir():
            continue
        for result in directory.glob("run_*.json"):
            try:
                obj = json.loads(result.read_text())
            except Exception:
                continue
            qid = obj.get("query_id")
            if qid is None:
                continue
            score = (1 if obj.get("status") == "completed" else 0, result.stat().st_mtime)
            old = by_qid.get(str(qid))
            if old is None or score > old[0]:
                by_qid[str(qid)] = (score, result)
    resume_dir.mkdir(parents=True, exist_ok=True)
    for qid, (_, result) in by_qid.items():
        shutil.copy2(result, resume_dir / f"run_preseed_{qid}.json")
    return len(by_qid)


def terminate(processes: list[tuple[str, subprocess.Popen]], log_file: Path) -> None:
    for name, process in reversed(processes):
        if process.poll() is None:
            log_line(log_file, f"TERM {name} pid={process.pid}")
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception as exc:
                log_line(log_file, f"TERM_ERR {name}: {exc}")
    time.sleep(5)
    for name, process in reversed(processes):
        if process.poll() is None:
            log_line(log_file, f"KILL {name} pid={process.pid}")
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception as exc:
                log_line(log_file, f"KILL_ERR {name}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/bn/search-tiktok-nas-au/yuqizhou/competition"))
    parser.add_argument("--retriever-repo", required=True)
    parser.add_argument("--search-gpus", default="0,1,2,3")
    parser.add_argument("--workers-per-server", type=int, default=16)
    parser.add_argument("--mcp-port", type=int, default=8090)
    parser.add_argument("--max-tokens", type=str, default="4096")
    parser.add_argument("--no-preseed", action="store_true")
    args = parser.parse_args()

    root = args.root
    safe = safe_name(args.retriever_repo)
    src = root / "src/BrowseComp-Plus"
    python = root / "envs/browsecomp-plus/bin/python"
    model_dir = root / "models" / safe
    index_glob = root / "indexes" / safe / "corpus.shard*.pkl"
    corpus_path = root / "data/corpus_sorted" / f"{safe}.full.shards8.len_sorted.max4096.jsonl"

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"__{safe}_qwen35_existing_vllm_w64_multigpu_search"
    log_dir = root / "logs" / run_id
    run_dir = root / "runs" / safe / run_id
    chunk_dir = log_dir / "query_chunks"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "orchestrator.log"

    preseeded = 0
    if not args.no_preseed:
        preseeded = preseed_results(root / "runs" / safe, run_dir)

    manifest = {
        "run_id": run_id,
        "retriever_repo": args.retriever_repo,
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "existing_vllm_ports": [port for _, port in GROUPS],
        "search_gpus": args.search_gpus,
        "workers_per_server": args.workers_per_server,
        "preseeded_results": preseeded,
    }
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "logs/e2e830_latest_resume_multigpu_search.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_line(log_file, "MANIFEST " + json.dumps(manifest))

    for _, port in GROUPS:
        if not wait_http(f"http://127.0.0.1:{port}/v1/models", 120):
            log_line(log_file, f"vLLM port {port} not ready")
            return 2

    processes: list[tuple[str, subprocess.Popen]] = []

    def start(name: str, command: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.Popen:
        log_line(log_file, f"START {name}: {' '.join(command)}")
        output = (log_dir / f"{name}.log").open("ab")
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, cwd=str(cwd or root), env=env, start_new_session=True)
        processes.append((name, process))
        log_line(log_file, f"PID {name} {process.pid}")
        return process

    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.search_gpus,
            "BCP_ATTN_IMPLEMENTATION": "eager",
            "PYTHONPATH": str(src) + ":" + env.get("PYTHONPATH", ""),
            "OTEL_SDK_DISABLED": "true",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    mcp_cmd = [
        str(python),
        "searcher/mcp_server.py",
        "--searcher-type",
        "faiss",
        "--index-path",
        str(index_glob),
        "--model-name",
        str(model_dir),
        "--dataset-name",
        "json",
        "--dataset-path",
        str(corpus_path),
        "--normalize",
        "--get-document",
        "--snippet-max-tokens",
        "512",
        "--k",
        "5",
        "--transport",
        "sse",
        "--port",
        str(args.mcp_port),
    ]
    start("mcp_search_multigpu", mcp_cmd, env=env, cwd=src)
    if not wait_http(f"http://127.0.0.1:{args.mcp_port}/mcp", 900):
        log_line(log_file, "MCP search server not ready")
        terminate(processes, log_file)
        return 3

    workers = args.workers_per_server * len(GROUPS)
    chunks = split_queries(src / "topics-qrels/queries.tsv", chunk_dir, workers)
    for i, chunk in enumerate(chunks):
        _, port = GROUPS[i % len(GROUPS)]
        worker_env = os.environ.copy()
        worker_env.update(
            {
                "BCP_MAX_TOOL_STEPS": "50",
                "PYTHONPATH": str(src) + ":" + worker_env.get("PYTHONPATH", ""),
                "OTEL_SDK_DISABLED": "true",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        command = [
            str(python),
            "search_agent/qwen_client.py",
            "--query",
            str(chunk),
            "--model",
            "Qwen/Qwen3.5-4B",
            "--model-server",
            f"http://127.0.0.1:{port}/v1",
            "--mcp-url",
            f"http://127.0.0.1:{args.mcp_port}/mcp",
            "--output-dir",
            str(run_dir),
            "--query-template",
            "QUERY_TEMPLATE",
            "--max_tokens",
            args.max_tokens,
        ]
        start(f"qwen_agent_worker{i:02d}_port{port}", command, env=worker_env, cwd=src)

    code = 0
    for name, process in processes[1:]:
        rc = process.wait()
        log_line(log_file, f"client {name} pid={process.pid} exited rc={rc}")
        if rc:
            code = rc
    log_line(log_file, f"DONE result_files={len(list(run_dir.glob('run_*.json')))}")
    terminate(processes, log_file)
    return code


if __name__ == "__main__":
    sys.exit(main())
