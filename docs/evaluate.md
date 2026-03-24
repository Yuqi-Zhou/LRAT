# Evaluation

This repository evaluates final agent responses with a local judge model loaded through vLLM.

## Script

```text
scripts_evaluation/evaluate.py
```

## What the Script Does

For each saved trajectory output JSON:

1. load the matching ground-truth question and answer,
2. extract the final agent response,
3. ask a local judge model whether the response is correct, and
4. aggregate success and trajectory statistics.

For `browsecomp-plus`, the script can also compute evidence recall from qrels.

## Inputs

| Argument | Meaning |
| --- | --- |
| `--input_dir` | Directory containing per-query run JSON files |
| `--gt_path` | Ground-truth TSV file with at least `query_id`, `question`, `answer` |
| `--dataset_type` | `browsecomp-plus` or `InfoSeek-Eval` |
| `--output_file` | Path to save the evaluation JSON |
| `--model_path` | Local vLLM judge model path |
| `--qrel_path` | Required for `browsecomp-plus` if you want evidence recall |
| `--tensor_parallel_size` | Tensor-parallel size for vLLM |
| `--gpu_memory_utilization` | GPU memory utilization ratio |
| `--batch_size` | Batched judge inference size |

## Example: BrowseComp-Plus

```bash
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

## Example: InfoSeek-Eval

```bash
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

## Output JSON

The output file written to `--output_file` contains:

- `metrics`: aggregated metrics such as success rate, average steps, and evidence recall
- `details`: per-sample judge results

## Important Matching Rule

The evaluation script matches trajectory files to ground truth through `query_id`, so make sure:

- your trajectory JSON files keep valid `query_id` values, and
- those IDs match the first column of the ground-truth TSV.

## Recommended Practice

- Use the same `input_dir` naming convention as your training/evaluation experiment names
- Save one evaluation JSON per model / retriever / dataset combination
- Keep judge settings fixed when comparing multiple retrievers
