#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build segmented training samples from agent trajectories.

This script reuses the same judge-and-extract pipeline as src/data_builder.py,
but filters browse actions within each search run by segment position.
"""

import argparse
import json
import logging
import os
import statistics
from concurrent.futures import ProcessPoolExecutor, TimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from transformers import AutoTokenizer

from data_builder import (
    BROWSE_TOOLS,
    _collect_unbrowsed_docs,
    _extract_reasoning_text_from_next_step,
    _get_docid_from_browse_step,
    _parse_search,
    _unique_preserve_order,
    add_reweight_rate,
    judge_relevance,
    load_corpus_jsonl,
)


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SEGMENT_BOUNDARIES = {
    "front30": (0.0, 0.30),
    "middle30": (0.35, 0.65),
    "back30": (0.70, 1.0),
    "full100": (0.0, 1.0),
}

_GLOBALS: Dict[str, Any] = {}


def load_named_trajectories(dir_path: str, verbose: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
    trajectories: List[Tuple[str, Dict[str, Any]]] = []
    total = loaded = failed = 0

    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        total += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                traj = json.load(f)
            if isinstance(traj, dict) and "result" in traj:
                trajectories.append((fname, traj))
                loaded += 1
            else:
                failed += 1
        except Exception as exc:
            failed += 1
            if verbose:
                logger.warning("Failed to read %s: %s", fname, exc)

    if verbose:
        logger.info(
            "Trajectory loading summary | total=%s loaded=%s skipped_or_failed=%s",
            total,
            loaded,
            failed,
        )

    return trajectories


def _token_len(tokenizer, text: str) -> Optional[int]:
    if tokenizer is None:
        return None
    try:
        out = tokenizer(text, add_special_tokens=False)
        if isinstance(out, dict) and "input_ids" in out:
            return len(out["input_ids"])
    except Exception:
        pass
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def _normalize_position(idx: int, count: int) -> float:
    if count <= 1:
        return 0.0
    return idx / float(count - 1)


def _parse_segment_boundaries(raw: str) -> Dict[str, Tuple[float, float]]:
    boundaries = dict(DEFAULT_SEGMENT_BOUNDARIES)
    if not raw.strip():
        return boundaries

    loaded = json.loads(raw)
    alias = {"front": "front30", "middle": "middle30", "back": "back30", "full": "full100"}
    for key, value in loaded.items():
        norm_key = alias.get(key, key)
        if norm_key not in boundaries:
            raise ValueError(f"Unknown segment key in --segment-boundary: {key}")
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"Segment boundary for {key} must be a JSON list of length 2")
        start, end = float(value[0]), float(value[1])
        if start < 0 or end > 1 or start > end:
            raise ValueError(f"Invalid boundary for {key}: {value}")
        boundaries[norm_key] = (start, end)
    return boundaries


def _in_segment(idx: int, count: int, mode: str, boundaries: Dict[str, Tuple[float, float]]) -> bool:
    if mode == "full100":
        return True
    pos = _normalize_position(idx, count)
    start, end = boundaries[mode]
    if mode == "back30":
        return start <= pos <= end
    return start <= pos < end


def extract_segmented_pairs(
    traj: Dict[str, Any],
    traj_id: str,
    tokenizer,
    corpus_data: Dict[str, str],
    judge_fn,
    *,
    segment_mode: str,
    segment_tag: str,
    min_browse_per_search: int,
    boundaries: Dict[str, Tuple[float, float]],
) -> List[Dict[str, Any]]:
    steps = traj["result"]
    samples: List[Dict[str, Any]] = []

    current_query = None
    history_search_results: List[List[str]] = []
    history_browsed_unsat: List[str] = []
    search_idx = -1

    i = 0
    while i < len(steps):
        step = steps[i]

        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            current_query, current_search_docs = _parse_search(step)
            history_search_results.append(current_search_docs)
            search_idx += 1
            i += 1
            continue

        if step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS:
            if not current_query:
                current_query = ""

            run: List[Dict[str, Any]] = []
            j = i
            browse_count_in_search = 0

            while j < len(steps):
                st = steps[j]
                if not (st.get("type") == "tool_call" and st.get("tool_name") in BROWSE_TOOLS):
                    break

                docid = _get_docid_from_browse_step(st)
                if docid is None:
                    if j + 1 < len(steps) and steps[j + 1].get("type") == "reasoning":
                        j += 2
                    else:
                        j += 1
                    continue

                reasoning_text = _extract_reasoning_text_from_next_step(steps, j)
                reasoning_len = _token_len(tokenizer, reasoning_text)
                satisfied = bool(judge_fn(reasoning_text))

                run.append(
                    {
                        "docid": docid,
                        "reasoning_text": reasoning_text,
                        "reasoning_len": reasoning_len,
                        "satisfied": satisfied,
                        "browse_idx_in_search": browse_count_in_search,
                    }
                )
                browse_count_in_search += 1

                if j + 1 < len(steps) and steps[j + 1].get("type") == "reasoning":
                    j += 2
                else:
                    j += 1

            if browse_count_in_search < min_browse_per_search:
                i = j
                continue

            filtered_run = [
                item
                for item in run
                if _in_segment(
                    item["browse_idx_in_search"],
                    browse_count_in_search,
                    segment_mode,
                    boundaries,
                )
            ]

            if not filtered_run:
                i = j
                continue

            run_sat_docs = [x["docid"] for x in filtered_run if x["satisfied"]]
            run_unsat_docs = [x["docid"] for x in filtered_run if not x["satisfied"]]

            browsed_set = set(history_browsed_unsat) | set(run_unsat_docs) | set(run_sat_docs)
            unbrowsed_docs = _collect_unbrowsed_docs(history_search_results, browsed_set)

            prev_unsat_near_first = list(reversed(history_browsed_unsat))
            run_unsat_near_first = list(reversed(run_unsat_docs))
            sat_set = set(run_sat_docs)

            def make_neg_for_unsat_doc(pos_doc: str) -> List[str]:
                part1 = [d for d in run_unsat_near_first if d != pos_doc]
                part2 = prev_unsat_near_first
                part3 = unbrowsed_docs
                neg = _unique_preserve_order(part1 + part2 + part3)
                return [d for d in neg if d not in sat_set and d != pos_doc]

            def make_neg_for_sat_sample() -> List[str]:
                neg = _unique_preserve_order(run_unsat_near_first + prev_unsat_near_first + unbrowsed_docs)
                return [d for d in neg if d not in sat_set]

            all_searched_docids: List[str] = []
            for docs in history_search_results:
                all_searched_docids.extend(docs)

            for item in filtered_run:
                if item["docid"] not in all_searched_docids:
                    continue
                if item["docid"] not in corpus_data:
                    continue

                if item["satisfied"]:
                    neg_ids = make_neg_for_sat_sample()
                else:
                    neg_ids = make_neg_for_unsat_doc(item["docid"])

                neg_ids = [d for d in neg_ids if d in corpus_data]
                neg_texts = [corpus_data[d] for d in neg_ids]

                samples.append(
                    {
                        "query": current_query,
                        "pos": [corpus_data[item["docid"]]],
                        "neg": neg_texts,
                        "pos_id": [item["docid"]],
                        "neg_id": neg_ids,
                        "reasoning_len": item["reasoning_len"],
                        "satisfied": bool(item["satisfied"]),
                        "segment_mode": segment_mode,
                        "segment_tag": segment_tag,
                        "traj_id": traj_id,
                        "search_idx": search_idx,
                        "browse_idx_in_search": item["browse_idx_in_search"],
                        "browse_count_in_search": browse_count_in_search,
                    }
                )

            history_browsed_unsat.extend(run_unsat_docs)

            if run_sat_docs:
                history_search_results = []
                history_browsed_unsat = []
                current_query = None

            i = j
            continue

        i += 1

    return samples


def _init_worker(
    tokenizer_path: str,
    corpus_data: Dict[str, str],
    judge_api_url: str,
    judge_model: str,
    headers: Dict[str, str],
    segment_mode: str,
    segment_tag: str,
    min_browse_per_search: int,
    boundaries: Dict[str, Tuple[float, float]],
) -> None:
    _GLOBALS["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _GLOBALS["corpus_data"] = corpus_data
    _GLOBALS["judge_api_url"] = judge_api_url
    _GLOBALS["judge_model"] = judge_model
    _GLOBALS["headers"] = headers
    _GLOBALS["segment_mode"] = segment_mode
    _GLOBALS["segment_tag"] = segment_tag
    _GLOBALS["min_browse_per_search"] = min_browse_per_search
    _GLOBALS["boundaries"] = boundaries


def _worker(item: Tuple[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    traj_id, traj = item

    def _judge_fn(text: str) -> bool:
        return judge_relevance(
            text,
            judge_api_url=_GLOBALS["judge_api_url"],
            judge_model=_GLOBALS["judge_model"],
            headers=_GLOBALS["headers"],
        )

    return extract_segmented_pairs(
        traj,
        traj_id,
        _GLOBALS["tokenizer"],
        _GLOBALS["corpus_data"],
        _judge_fn,
        segment_mode=_GLOBALS["segment_mode"],
        segment_tag=_GLOBALS["segment_tag"],
        min_browse_per_search=_GLOBALS["min_browse_per_search"],
        boundaries=_GLOBALS["boundaries"],
    )


def build_summary(samples: List[Dict[str, Any]], segment_mode: str, segment_tag: str) -> Dict[str, Any]:
    reasoning_lens = [
        float(s["reasoning_len"])
        for s in samples
        if isinstance(s.get("reasoning_len"), (int, float)) and s["reasoning_len"] > 0
    ]
    neg_counts = [len(s.get("neg", [])) for s in samples]
    satisfied_true = sum(1 for s in samples if bool(s.get("satisfied")))
    empty_samples = sum(1 for s in samples if not s.get("pos") or not s.get("neg"))

    if reasoning_lens:
        mean_reasoning_len = statistics.mean(reasoning_lens)
        median_reasoning_len = statistics.median(reasoning_lens)
        reasoning_p90 = statistics.quantiles(reasoning_lens, n=10)[8] if len(reasoning_lens) >= 10 else max(reasoning_lens)
    else:
        mean_reasoning_len = 0.0
        median_reasoning_len = 0.0
        reasoning_p90 = 0.0

    search_windows = {(s["traj_id"], s["search_idx"]) for s in samples}

    return {
        "segment_mode": segment_mode,
        "segment_tag": segment_tag,
        "sample_count": len(samples),
        "unique_queries": len({s.get("query", "") for s in samples}),
        "unique_trajectories": len({s.get("traj_id", "") for s in samples}),
        "covered_search_windows": len(search_windows),
        "avg_negatives": statistics.mean(neg_counts) if neg_counts else 0.0,
        "satisfied_true": satisfied_true,
        "satisfied_false": len(samples) - satisfied_true,
        "satisfied_true_ratio": (satisfied_true / len(samples)) if samples else 0.0,
        "mean_reasoning_len": mean_reasoning_len,
        "median_reasoning_len": median_reasoning_len,
        "reasoning_len_p90": reasoning_p90,
        "empty_sample_count": empty_samples,
        "empty_sample_ratio": (empty_samples / len(samples)) if samples else 0.0,
    }


def parse_args():
    ap = argparse.ArgumentParser(description="Build segmented training samples from trajectories.")
    ap.add_argument("--corpus-path", type=str, required=True, help="Path to corpus jsonl: {docid, text} per line")
    ap.add_argument("--traj-dir", type=str, required=True, help="Directory containing trajectory JSON files")
    ap.add_argument("--output-path", type=str, required=True, help="Output jsonl path")
    ap.add_argument("--summary-path", type=str, default="", help="Optional path to save JSON summary")
    ap.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer path for counting reasoning tokens")
    ap.add_argument("--judge-api-url", type=str, required=True, help="OpenAI-compatible chat completions endpoint")
    ap.add_argument("--judge-model", type=str, default="auto", help="Judge model name passed to API")
    ap.add_argument("--judge-headers", type=str, default="", help='Optional JSON string for HTTP headers')
    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--future-timeout", type=int, default=15, help="Per-trajectory timeout (seconds)")
    ap.add_argument(
        "--segment-mode",
        choices=["front30", "middle30", "back30", "full100"],
        required=True,
        help="Which browse segment to keep within each search run",
    )
    ap.add_argument("--segment-save-tag", type=str, default="", help="Optional tag recorded in output metadata")
    ap.add_argument("--min-browse-per-search", type=int, default=1, help="Skip search runs with fewer browse actions than this threshold")
    ap.add_argument(
        "--segment-boundary",
        type=str,
        default="",
        help='Optional JSON override, e.g. {"front":[0,0.3],"middle":[0.35,0.65],"back":[0.7,1.0]}',
    )
    return ap.parse_args()


def main():
    args = parse_args()
    headers = json.loads(args.judge_headers) if args.judge_headers.strip() else {"Content-Type": "application/json"}
    boundaries = _parse_segment_boundaries(args.segment_boundary)
    segment_tag = args.segment_save_tag or args.segment_mode

    corpus_data = load_corpus_jsonl(args.corpus_path)
    trajectories = load_named_trajectories(args.traj_dir, verbose=True)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if args.summary_path:
        summary_dir = os.path.dirname(args.summary_path)
        if summary_dir:
            os.makedirs(summary_dir, exist_ok=True)

    all_samples: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(
            args.tokenizer_path,
            corpus_data,
            args.judge_api_url,
            args.judge_model,
            headers,
            args.segment_mode,
            segment_tag,
            args.min_browse_per_search,
            boundaries,
        ),
    ) as ex:
        futures = {ex.submit(_worker, item): item[0] for item in trajectories}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            traj_id = futures[future]
            try:
                samples = future.result(timeout=args.future_timeout)
            except TimeoutError:
                logger.warning("Timeout on trajectory: %s", traj_id)
                continue
            except Exception as exc:
                logger.warning("Error on trajectory %s: %r", traj_id, exc)
                continue
            if samples:
                all_samples.extend(samples)

    half_life, mean_w = add_reweight_rate(all_samples)
    summary = build_summary(all_samples, args.segment_mode, segment_tag)
    summary["reweight_half_life"] = half_life
    summary["reweight_mean_raw_weight"] = mean_w

    logger.info("Segment summary:\n%s", json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.output_path, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    if args.summary_path:
        with open(args.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
