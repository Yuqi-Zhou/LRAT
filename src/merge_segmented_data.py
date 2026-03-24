#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
from typing import Dict, List


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[dict]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(description="Merge multiple segmented training-data JSONL files.")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    ap.add_argument("--output-path", required=True, help="Output JSONL path")
    ap.add_argument(
        "--sample-mode",
        choices=["concat", "equal"],
        default="equal",
        help="concat keeps all samples; equal downsamples each input to the smallest size",
    )
    ap.add_argument("--seed", type=int, default=2026, help="Random seed for sampling/shuffling")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle the merged output")
    ap.add_argument("--summary-path", default="", help="Optional path to save merge summary JSON")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    grouped: Dict[str, List[dict]] = {}
    for input_path in args.inputs:
        grouped[input_path] = read_jsonl(input_path)

    counts = {path: len(rows) for path, rows in grouped.items()}

    if args.sample_mode == "equal":
        target = min(counts.values()) if counts else 0
        merged: List[dict] = []
        for path, rows in grouped.items():
            picked = rows[:]
            rng.shuffle(picked)
            merged.extend(picked[:target])
    else:
        merged = []
        for rows in grouped.values():
            merged.extend(rows)

    if args.shuffle:
        rng.shuffle(merged)

    summary = {
        "sample_mode": args.sample_mode,
        "seed": args.seed,
        "shuffle": bool(args.shuffle),
        "inputs": counts,
        "output_count": len(merged),
    }

    logger.info("Merge summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))
    write_jsonl(args.output_path, merged)

    if args.summary_path:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
