#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build training samples from Tongyi trajectories.

Inputs:
- A corpus jsonl file: each line contains {"docid": ..., "text": ...}
- A directory of trajectory json files: each file is a dict containing key "result" (a list of steps)
- An OpenAI-compatible chat completion endpoint for LLM-judge (relevance classifier)

Outputs:
- jsonl training data with fields:
  query, pos, neg, pos_id, neg_id, reasoning_len, satisfied, reweight_rate

Example:
python src/data_builder.py \
  --corpus-path /root/corpus/wiki-25/all/wiki-25-512-final.jsonl \
  --traj-dir /root/runs/wiki-25/tongyi/true_false/qwen3-0.6b_topk10_true \
  --output-path /root/training_data/tongyi/qwen3-0.6b/v3.jsonl \
  --tokenizer-path /root/PLM/Qwen3-Embedding-0.6B \
  --judge-api-url http://xx.xxx.xx.x:xxxx/v1/chat/completions \
  --judge-model auto \
  --max-workers 32 \
  --future-timeout 15
"""

import os
import re
import json
import logging
import math
import time
import string
import statistics
import argparse
from typing import Dict, Any, List, Optional

import requests
from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError


logging.basicConfig(
    level=getattr(logging, os.getenv("LRAT_LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


BROWSE_TOOLS = {"get_document", "visit"}


# -------------------------
# IO
# -------------------------
def load_corpus_jsonl(path: str) -> Dict[str, str]:
    """Load corpus jsonl into {docid(str): text(str)}."""
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading corpus: {os.path.basename(path)}"):
            if not line.strip():
                continue
            data = json.loads(line)
            result[str(data["docid"])] = data["text"]
    return result


def load_all_trajectories(dir_path: str, verbose: bool = True) -> List[Dict[str, Any]]:
    """Read all JSON trajectory files under a directory. Keep those containing 'result'."""
    trajectories = []
    total = loaded = failed = 0

    for fname in os.listdir(dir_path):
        fpath = os.path.join(dir_path, fname)
        if not os.path.isfile(fpath):
            continue
        total += 1
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                traj = json.load(f)
            if isinstance(traj, dict) and "result" in traj:
                trajectories.append(traj)
                loaded += 1
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if verbose:
                logger.warning("Failed to read %s: %s", fname, e)

    if verbose:
        logger.info(
            "Trajectory loading summary | total=%s loaded=%s skipped_or_failed=%s",
            total,
            loaded,
            failed,
        )

    return trajectories


# -------------------------
# Judge (LLM API)
# -------------------------
JUDGE_PROMPT = r"""
You are an LLM judge. You will classify whether the AnalysisText suggests the browsing text is relevant or not relevant, using a bias aligned with typical browsing behavior: models often keep searching even when content is relevant, but they only say “not relevant” when it’s clearly off-topic.

Input
AnalysisText: another model’s analysis of a browsing page.
Decision rule (important)
Output NOT_RELEVANT only if the analysis contains a clear negative judgment of relevance (explicit or unmistakable), such as: “not relevant / irrelevant / unrelated / off-topic / doesn’t help / cannot answer / no useful info,” or it clearly concludes the content is about a different topic and provides no value for the task.
Otherwise output RELEVANT.
    - This includes cases where the analysis:
    - extracts useful facts/steps/details from the page,
    - says the page is partially helpful,
    - suggests using it as background/context,
    - recommends continuing to search for more sources (continuing search does not imply irrelevance).
Output (strict)
Return exactly one token:

RELEVANT
NOT_RELEVANT

Classify this:
AnalysisText:
{{ANALYSIS_TEXT}}
""".strip()


def call_chat_completions(
    api_url: str,
    model_name: str,
    prompt: str,
    timeout_s: int = 60 * 60,
    n: int = 1,
    stop: Optional[Any] = None,
    top_p: float = 0.95,
    top_k: int = 50,
    temperature: float = 0.8,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "top_p": top_p,
        "top_k": top_k,
        "temperature": temperature,
        "n": n,
        "stop": stop,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def judge_relevance(
    analysis_text: str,
    *,
    judge_api_url: str,
    judge_model: str,
    headers: Optional[Dict[str, str]] = None,
    retries: int = 2,
    sleep_sec: float = 1.0,
) -> bool:
    """Return True for RELEVANT, False for NOT_RELEVANT (conservative fallback: False on errors)."""
    prompt = JUDGE_PROMPT.replace("{{ANALYSIS_TEXT}}", str(analysis_text))

    last_err = None
    for _ in range(retries + 1):
        try:
            result = call_chat_completions(
                api_url=judge_api_url,
                model_name=judge_model,
                prompt=prompt,
                headers=headers,
            )
            content = result["choices"][0]["message"]["content"].strip()
            # be robust to extra whitespace/newlines
            first_token = content.split()[0] if content else ""
            return first_token == "RELEVANT"
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)

    return False


# -------------------------
# Trajectory parsing helpers
# -------------------------
def _get_docid_from_browse_step(step: Dict[str, Any]) -> Optional[str]:
    args = json.loads(step.get("arguments", "{}") or "{}")
    doc = args.get("docid", None)
    if isinstance(doc, list):
        doc = doc[0] if doc else None
    return str(doc).split(":")[-1] if doc is not None else None

def _extract_reasoning_text_from_next_step(steps: List[Dict[str, Any]], i: int) -> str:
    if i + 1 < len(steps) and steps[i + 1].get("type") == "reasoning":
        out = steps[i + 1].get("output", "")
        if isinstance(out, list):
            return " ".join(str(x) for x in out)
        return str(out)
    return ""


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


def _parse_search(step: Dict[str, Any]) -> (str, List[str]):
    args = json.loads(step.get("arguments", "{}") or "{}")
    q = args.get("query", [""])
    query = q[0] if isinstance(q, list) and q else str(q)

    docs = []
    output = step.get("output", "") or ""
    for line in output.split("\n"):
        if line.startswith("DocID:"):
            docs.append(line.split(":", 1)[1].strip())
    return query, docs


def _collect_unbrowsed_docs(history_search_results: List[List[str]], browsed_set: set) -> List[str]:
    """Most recent search first; within each search keep original order."""
    res = []
    for docs in reversed(history_search_results):
        for d in docs:
            ds = str(d)
            if ds not in browsed_set:
                res.append(ds)
    return res


def _unique_preserve_order(seq: List[Optional[str]]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        xs = str(x)
        if xs in seen:
            continue
        seen.add(xs)
        out.append(xs)
    return out


# -------------------------
# Sample extraction
# -------------------------
def extract_pairs_with_satisfaction_groups(
    traj: Dict[str, Any],
    tokenizer,
    corpus_data: Dict[str, str],
    judge_fn,
) -> List[Dict[str, Any]]:
    steps = traj["result"]
    samples = []

    current_query = None
    current_search_docs = None

    history_search_results = []    # list of doc lists, per search
    history_browsed_unsat = []     # time order list of unsatisfied browsed docids

    i = 0
    while i < len(steps):
        step = steps[i]

        if step.get("type") == "tool_call" and step.get("tool_name") == "search":
            current_query, current_search_docs = _parse_search(step)
            history_search_results.append(current_search_docs)
            i += 1
            continue

        if step.get("type") == "tool_call" and step.get("tool_name") in BROWSE_TOOLS:
            if not current_query:
                current_query = ""

            run = []
            j = i
            while j < len(steps):
                st = steps[j]
                if not (st.get("type") == "tool_call" and st.get("tool_name") in BROWSE_TOOLS):
                    break

                docid = _get_docid_from_browse_step(st)
                if docid is None:
                    j += 1
                    continue

                reasoning_text = _extract_reasoning_text_from_next_step(steps, j)
                rlen = _token_len(tokenizer, reasoning_text)
                sat = bool(judge_fn(reasoning_text))

                run.append(
                    {
                        "docid": docid,
                        "reasoning_text": reasoning_text,
                        "reasoning_len": rlen,
                        "satisfied": sat,
                    }
                )

                if j + 1 < len(steps) and steps[j + 1].get("type") == "reasoning":
                    j += 2
                else:
                    j += 1

            run_sat_docs = [x["docid"] for x in run if x["satisfied"]]
            run_unsat_docs = [x["docid"] for x in run if not x["satisfied"]]

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
                neg = [d for d in neg if d not in sat_set and d != pos_doc]
                return neg

            def make_neg_for_sat_sample() -> List[str]:
                neg = _unique_preserve_order(run_unsat_near_first + prev_unsat_near_first + unbrowsed_docs)
                neg = [d for d in neg if d not in sat_set]
                return neg

            all_searched_docids = []
            for docs in history_search_results:
                all_searched_docids.extend(docs)

            for x in run:
                if x["docid"] not in all_searched_docids:
                    continue
                if x["docid"] not in corpus_data:
                    continue

                if x["satisfied"]:
                    neg_ids = make_neg_for_sat_sample()
                else:
                    neg_ids = make_neg_for_unsat_doc(x["docid"])

                neg_texts = [corpus_data[d] for d in neg_ids if d in corpus_data]

                samples.append(
                    {
                        "query": current_query,
                        "pos": [corpus_data[x["docid"]]],
                        "neg": neg_texts,
                        "pos_id": [x["docid"]],
                        "neg_id": [d for d in neg_ids if d in corpus_data],
                        "reasoning_len": x["reasoning_len"],
                        "satisfied": bool(x["satisfied"]),
                    }
                )

            history_browsed_unsat.extend(run_unsat_docs)

            if run_sat_docs:
                history_search_results = []
                history_browsed_unsat = []
                current_query = None
                current_search_docs = None

            i = j
            continue

        i += 1
    return samples

# -------------------------
# Reweighting
# -------------------------
def add_reweight_rate(samples: List[Dict[str, Any]]):
    all_lens = []
    for s in samples:
        rl = s.get("reasoning_len", None)
        if isinstance(rl, (int, float)) and rl > 0:
            all_lens.append(float(rl))

    half_life = statistics.median(all_lens) if all_lens else 1.0
    ln2 = math.log(2.0)

    all_raw = []
    raw_per_sample = []

    for s in samples:
        rl = s.get("reasoning_len", None)
        if isinstance(rl, (int, float)) and rl > 0:
            w = 1 - math.exp(-(float(rl)) * ln2 / float(half_life))
            raw_per_sample.append(w)
            all_raw.append(w)
        else:
            raw_per_sample.append(None)

    mean_w = (sum(all_raw) / len(all_raw)) if all_raw else 1.0

    for s, raw in zip(samples, raw_per_sample):
        if isinstance(raw, (int, float)):
            s["reweight_rate"] = (raw / mean_w) if mean_w != 0 else 1.0
        else:
            s["reweight_rate"] = 1.0

    return half_life, mean_w


# -------------------------
# Multiprocessing worker
# -------------------------
_GLOBALS = {}


def _init_worker(tokenizer_path: str, corpus_data: Dict[str, str], judge_api_url: str, judge_model: str, headers: Dict[str, str]):
    # store as process globals
    _GLOBALS["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
    _GLOBALS["corpus_data"] = corpus_data
    _GLOBALS["judge_api_url"] = judge_api_url
    _GLOBALS["judge_model"] = judge_model
    _GLOBALS["headers"] = headers


def _worker(traj: Dict[str, Any]) -> List[Dict[str, Any]]:
    tok = _GLOBALS["tokenizer"]
    corpus_data = _GLOBALS["corpus_data"]

    def _judge_fn(text: str) -> bool:
        return judge_relevance(
            text,
            judge_api_url=_GLOBALS["judge_api_url"],
            judge_model=_GLOBALS["judge_model"],
            headers=_GLOBALS["headers"],
        )

    return extract_pairs_with_satisfaction_groups(traj, tok, corpus_data, _judge_fn)


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-path", type=str, required=True, help="Path to corpus jsonl: {docid, text} per line")
    ap.add_argument("--traj-dir", type=str, required=True, help="Directory containing trajectory JSON files")
    ap.add_argument("--output-path", type=str, required=True, help="Output jsonl path")

    ap.add_argument("--tokenizer-path", type=str, required=True, help="Tokenizer path for counting reasoning tokens")

    ap.add_argument("--judge-api-url", type=str, required=True, help="OpenAI-compatible chat completions endpoint")
    ap.add_argument("--judge-model", type=str, default="auto", help="Judge model name passed to API")
    ap.add_argument("--judge-headers", type=str, default="", help='Optional JSON string for HTTP headers, e.g. \'{"Authorization":"Bearer ..."}\'')

    ap.add_argument("--max-workers", type=int, default=32)
    ap.add_argument("--future-timeout", type=int, default=15, help="Per-trajectory timeout (seconds)")

    return ap.parse_args()


def main():
    args = parse_args()

    headers = json.loads(args.judge_headers) if args.judge_headers.strip() else {"Content-Type": "application/json"}

    corpus_data = load_corpus_jsonl(args.corpus_path)
    trajectories = load_all_trajectories(args.traj_dir, verbose=True)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_samples: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(
        max_workers=args.max_workers,
        initializer=_init_worker,
        initargs=(args.tokenizer_path, corpus_data, args.judge_api_url, args.judge_model, headers),
    ) as ex:
        futures = {ex.submit(_worker, traj): i for i, traj in enumerate(trajectories)}
        for fu in tqdm(as_completed(futures), total=len(futures), desc="Processing trajectories"):
            idx = futures[fu]
            try:
                ss = fu.result(timeout=args.future_timeout)
            except TimeoutError:
                logger.warning("Timeout on trajectory index: %s", idx)
                continue
            except Exception as e:
                logger.warning("Error on trajectory index %s: %r", idx, e)
                continue

            if ss:
                all_samples.extend(ss)

    half_life, mean_w = add_reweight_rate(all_samples)
    logger.info(
        "Finished building samples | half_life=%s mean_raw_weight=%s sample_count=%s",
        half_life,
        mean_w,
        len(all_samples),
    )

    with open(args.output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
