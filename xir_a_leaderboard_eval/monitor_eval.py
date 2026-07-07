#!/usr/bin/env python3
"""Small status monitor for the XIR A-leaderboard evaluation run."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/mnt/bn/search-tiktok-nas-au/yuqizhou/competition"))
    parser.add_argument("--run-glob", default="*qwen35*")
    parser.add_argument("--total", type=int, default=830)
    args = parser.parse_args()

    logs = sorted((args.root / "logs").glob(args.run_glob), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit(f"No logs matched {args.run_glob!r}")
    log_dir = logs[-1]
    manifest = json.loads((log_dir / "manifest.json").read_text())
    run_dir = Path(manifest["run_dir"])
    files = sorted(run_dir.glob("run_*.json"), key=lambda p: p.stat().st_mtime)
    now = time.time()

    print(f"run={log_dir}")
    print(f"run_dir={run_dir}")
    print(f"results={len(files)} remaining={args.total - len(files)}")
    if files:
        last = max(p.stat().st_mtime for p in files)
        print(f"last_result_sec_ago={now - last:.1f}")
        for window in (60, 180, 300, 600):
            n = sum(1 for p in files if now - p.stat().st_mtime <= window)
            rate = n / window * 60
            eta = (args.total - len(files)) / rate if rate else None
            print(f"last_{window}s={n} rate_q_min={rate:.2f} eta_min={eta if eta is not None else 'NA'}")

    ps = subprocess.run(["ps", "-eo", "pid,etime,cmd"], text=True, stdout=subprocess.PIPE, check=False).stdout
    print("alive_clients=", sum("qwen_client.py" in line for line in ps.splitlines()))
    print("alive_vllm=", sum("vllm.entrypoints.openai.api_server" in line for line in ps.splitlines()))
    print("alive_mcp=", sum("mcp_server.py" in line for line in ps.splitlines()))

    for vllm_log in sorted(log_dir.glob("vllm*.log")):
        metrics = [line for line in vllm_log.read_text(errors="replace").splitlines() if "Avg prompt throughput" in line]
        if not metrics:
            continue
        m = re.search(
            r"Avg prompt throughput: ([0-9.]+).*Avg generation throughput: ([0-9.]+).*Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([0-9.]+)%",
            metrics[-1],
        )
        print(vllm_log.name, m.groups() if m else metrics[-1][-180:])

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    ).stdout.strip()
    print(smi)


if __name__ == "__main__":
    main()
