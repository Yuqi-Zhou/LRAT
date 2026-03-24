#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import statistics
from typing import Any, Dict, List


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compute quick stats for a training-data JSONL file.")
    parser.add_argument("--input-path", required=True, help="Input JSONL file")
    parser.add_argument("--output-path", default="", help="Optional JSON path to save stats")
    args = parser.parse_args()

    rows = read_jsonl(args.input_path)
    neg_counts = [len(row.get("neg", [])) for row in rows]
    reasoning_lens = [
        float(row["reasoning_len"])
        for row in rows
        if isinstance(row.get("reasoning_len"), (int, float)) and row["reasoning_len"] > 0
    ]
    segment_counts: Dict[str, int] = {}
    for row in rows:
        mode = row.get("segment_mode", "unsegmented")
        segment_counts[mode] = segment_counts.get(mode, 0) + 1

    empty_samples = sum(1 for row in rows if not row.get("pos") or not row.get("neg"))
    satisfied_true = sum(1 for row in rows if bool(row.get("satisfied")))

    if reasoning_lens:
        reasoning_p90 = statistics.quantiles(reasoning_lens, n=10)[8] if len(reasoning_lens) >= 10 else max(reasoning_lens)
    else:
        reasoning_p90 = 0.0

    stats = {
        "input_path": args.input_path,
        "sample_count": len(rows),
        "unique_queries": len({row.get("query", "") for row in rows}),
        "avg_negatives": statistics.mean(neg_counts) if neg_counts else 0.0,
        "satisfied_true": satisfied_true,
        "satisfied_false": len(rows) - satisfied_true,
        "satisfied_true_ratio": (satisfied_true / len(rows)) if rows else 0.0,
        "mean_reasoning_len": statistics.mean(reasoning_lens) if reasoning_lens else 0.0,
        "median_reasoning_len": statistics.median(reasoning_lens) if reasoning_lens else 0.0,
        "reasoning_len_p90": reasoning_p90,
        "empty_sample_count": empty_samples,
        "empty_sample_ratio": (empty_samples / len(rows)) if rows else 0.0,
        "segment_counts": segment_counts,
    }

    logger.info("Training-data stats:\n%s", json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output_path:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
