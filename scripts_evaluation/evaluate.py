import argparse
import json
import csv
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

JUDGE_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. 

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


class DatasetLoader:
    @staticmethod
    def load_tsv(path: Path) -> Dict[str, Dict[str, str]]:
        gt_data: Dict[str, Dict[str, str]] = {}
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GT file not found: {path}")

        print(f"Loading GT from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for row in reader:
                if len(row) < 3:
                    continue
                q_id = row[0].strip()
                question = row[1].strip()
                answer = row[2].strip()
                gt_data[q_id] = {"question": question, "answer": answer}

        print(f"Loaded {len(gt_data)} samples.")
        return gt_data


def load_qrel_evidence(qrel_path: Path) -> Dict[str, List[str]]:
    qrel_data = defaultdict(list)
    if qrel_path is None:
        return dict(qrel_data)

    qrel_path = Path(qrel_path)
    if not qrel_path.exists():
        return dict(qrel_data)

    with qrel_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"Expected 4 columns in qrel line: {line}")
            query_id = parts[0]
            doc_id = parts[2]
            qrel_data[query_id].append(doc_id)

    return dict(qrel_data)


def parse_judge_result(text: str) -> bool:
    if not text:
        return False
    match = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    return False


def chunked(seq, batch_size: int):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory containing agent output JSON files")
    parser.add_argument("--gt_path", required=True, help="Path to the ground-truth TSV file")
    parser.add_argument("--dataset_type", choices=["browsecomp-plus", "InfoSeek-Eval"], required=True)
    parser.add_argument("--output_file", default="eval_results.json", help="Path to save evaluation results JSON")
    parser.add_argument("--model_path", required=True, help="Local model path for vLLM")
    parser.add_argument("--qrel_path", default=None, help="Optional qrel file path (only used for browsecomp-plus)")
    parser.add_argument("--tensor_parallel_size", type=int, default=8, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be a positive integer")

    ground_truth = DatasetLoader.load_tsv(Path(args.gt_path))
    qrel_data = {}
    if args.dataset_type == "browsecomp-plus":
        if args.qrel_path is None:
            raise ValueError("--qrel_path is required when --dataset_type is browsecomp-plus")
        qrel_data = load_qrel_evidence(Path(args.qrel_path))

    input_dir = Path(args.input_dir)
    input_files = sorted(list(input_dir.glob("*.json")), key=lambda x: x.name)

    evaluation_queue = []
    recalls = []

    print(f"Pre-processing {len(input_files)} files...")
    for json_file in tqdm(input_files, desc="Reading JSON"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                run_data = json.load(f)
        except Exception:
            continue

        qid = str(run_data.get("query_id"))
        if qid.endswith(".0"):
            qid = qid[:-2]
        if qid not in ground_truth:
            continue

        gt_item = ground_truth[qid]

        model_response = ""
        if isinstance(run_data.get("result"), list) and run_data["result"]:
            last_msg = run_data["result"][-1]
            if isinstance(last_msg, dict):
                model_response = last_msg.get("output", "") or ""


        prompt = JUDGE_TEMPLATE.format(
            question=gt_item["question"],
            response=model_response,
            correct_answer=gt_item["answer"],
        )

        tool_call_counts = run_data.get("tool_call_counts", {}) or {}
        if not isinstance(tool_call_counts, dict):
            tool_call_counts = {}
        steps = sum(v for v in tool_call_counts.values() if isinstance(v, (int, float)))

        recall = 0.0
        if args.dataset_type == "browsecomp-plus" and qid in qrel_data:
            retrieved = set(run_data.get("retrieved_docids", []) or [])
            gold_docs = set(qrel_data.get(qid, []))
            if gold_docs:
                recall = len(retrieved.intersection(gold_docs)) / float(len(gold_docs))

        evaluation_queue.append(
            {
                "qid": qid,
                "prompt": prompt,
                "steps": int(steps),
                "recall": float(recall),
                "file": json_file.name,
                "model_response": model_response,
            }
        )
        recalls.append(recall)

    if not evaluation_queue:
        print("No valid data to evaluate.")
        return

    print(f"Loading model: {args.model_path}")
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=512,
        seed=2026,
    )

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    all_results = []
    correct_count = 0

    print(f"Starting batched inference: batch_size={args.batch_size}, total={len(evaluation_queue)}")
    for batch in tqdm(list(chunked(evaluation_queue, args.batch_size)), desc="Inference"):
        prompts = [item["prompt"] for item in batch]
        outputs = llm.generate(prompts, sampling_params)

        if len(outputs) != len(batch):
            raise RuntimeError(f"Output size mismatch: got {len(outputs)} outputs for {len(batch)} prompts")

        for meta, output in zip(batch, outputs):
            judge_text = output.outputs[0].text if output.outputs else ""
            
            model_response = meta["model_response"]
            if model_response is None or 'tool_call' in model_response or 'user_query' in model_response: 
                is_correct=False
            else:
                is_correct = parse_judge_result(judge_text)

            if is_correct:
                correct_count += 1

            all_results.append(
                {
                    "query_id": meta["qid"],
                    "correct": bool(is_correct),
                    "steps": meta["steps"],
                    "recall": meta["recall"],
                    "judge_reason": judge_text,
                    "file": meta["file"],
                }
            )

    total = len(all_results)
    final_metrics = {
        "Success Rate": correct_count / float(total),
        "Avg Steps": float(np.mean([x["steps"] for x in all_results])) if total else 0.0,
    }
    if args.dataset_type == "browsecomp-plus":
        final_metrics["Evidence Recall"] = float(np.mean([x["recall"] for x in all_results])) if total else 0.0

    print("\n" + "=" * 30)
    print("Evaluation Summary")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")
    print("=" * 30)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": final_metrics, "details": all_results}, f, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
