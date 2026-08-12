#!/usr/bin/env python3
"""Strictly validate every expected trajectory in a CCIR-B run/shard."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SEARCH_AGENT_ROOT = PACKAGE_ROOT / "search_agent"
DEFAULT_CONFIG = PACKAGE_ROOT / "configs" / "campaign.example.json"
if str(SEARCH_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(SEARCH_AGENT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard")

    config_path = Path(args.config).resolve()
    os.environ["BCP_CAMPAIGN_CONFIG"] = str(config_path)
    from campaign_common import load_config, query_rows, validate_trajectory

    config = load_config()
    rows = query_rows(config)
    split_size = (len(rows) + args.shard_count - 1) // args.shard_count
    start = args.shard_index * split_size
    end = min(len(rows), start + split_size)
    selected = rows[start:end]
    output_root = Path(config["output_root"]) / args.run_id / args.profile
    reasons: Counter[str] = Counter()
    valid = 0
    for query_id, _query in selected:
        for retriever in config["retrievers"]:
            slug = str(retriever["slug"])
            path = output_root / slug / "trajectories" / f"{query_id}.json"
            result = validate_trajectory(
                path,
                expected_query_id=query_id,
                expected_profile=args.profile,
                expected_retriever_slug=slug,
            )
            if result.get("passed"):
                valid += 1
            else:
                reason = result.get("reason") or result.get("termination_reason") or "invalid"
                reasons[str(reason)] += 1

    expected = len(selected) * len(config["retrievers"])
    summary = {
        "profile": args.profile,
        "run_id": args.run_id,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "query_index_start_inclusive": start,
        "query_index_end_exclusive": end,
        "expected": expected,
        "valid": valid,
        "invalid_or_missing": expected - valid,
        "reasons": dict(reasons),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if valid == expected else 2


if __name__ == "__main__":
    raise SystemExit(main())
