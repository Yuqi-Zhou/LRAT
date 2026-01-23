## Evaluation (vLLM Judge)

This evaluation script reads the agent outputs (one `*.json` per query) and uses a **local judge model** loaded by **vLLM** to decide whether each final response is correct given the ground-truth answer. It reports summary metrics (e.g., Success Rate, Avg Steps) and saves per-sample judging details to a JSON file.

### Inputs

- `--input_dir`: Directory containing agent output JSON files (`*.json`).
- `--gt_path`: Ground-truth TSV file (must contain at least: `query_id`, `question`, `answer`).
- `--dataset_type`: `browsecomp-plus` or `InfoSeek-Eval`.
- `--qrel_path`: Only required for `browsecomp-plus`, used to compute evidence recall.
- `--model_path`: Local judge model path (loaded by vLLM).
- `--batch_size`: Inference batch size (fixed to 32 in the examples below).

### Example: Evaluate `browsecomp-plus`

```bash
#!/usr/bin/env bash
set -euo pipefail

python scripts_evaluation/evaluate.py \
  --input_dir /path/to/agent_output_json_dir \
  --gt_path /path/to/browsecomp-plus.tsv \
  --dataset_type browsecomp-plus \
  --qrel_path /path/to/qrel_evidence.txt \
  --output_file /path/to/save/eval_results.json \
  --model_path /path/to/local_judge_model \
  --tensor_parallel_size <NUM_GPUS> \
  --gpu_memory_utilization <GPU_MEM_UTIL> \
  --batch_size 32
```

### Example: Evaluate `InfoSeek-Eval` (no qrels required)

```bash
#!/usr/bin/env bash
set -euo pipefail

python scripts_evaluation/evaluate.py \
  --input_dir /path/to/agent_output_json_dir \
  --gt_path /path/to/InfoSeek-Eval.tsv \
  --dataset_type InfoSeek-Eval \
  --output_file /path/to/save/eval_results.json \
  --model_path /path/to/local_judge_model \
  --tensor_parallel_size <NUM_GPUS> \
  --gpu_memory_utilization <GPU_MEM_UTIL> \
  --batch_size 32
```

### Output

The script writes a JSON file to `--output_file` with the following structure:

- `metrics`: aggregated scores (e.g., `Success Rate`, `Avg Steps`, and `Evidence Recall` for browsecomp-plus)
- `details`: per-sample records, including the judge text and whether the answer is correct
