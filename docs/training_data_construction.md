# Training Data Construction

This document covers the scripts that convert trajectory JSON files into retriever-training JSONL files.

## Overview

There are now four related utilities in `src/`:

- `src/data_builder.py`: build the default training data from full trajectories
- `src/data_builder_segmented.py`: build segment-specific subsets such as `front30`, `middle30`, `back30`, and `full100`
- `src/merge_segmented_data.py`: merge multiple segmented JSONL files into one final training file
- `src/training_data_stats.py`: compute quick quality statistics for any generated JSONL file

## 1. Full-Trajectory Builder

Script:

```text
src/data_builder.py
```

This script reads:

1. a corpus JSONL file,
2. a directory of trajectory JSON files, and
3. an OpenAI-compatible judge endpoint.

It then labels browse steps as relevant / not relevant and writes samples with fields such as:

- `query`
- `pos`
- `neg`
- `pos_id`
- `neg_id`
- `reasoning_len`
- `satisfied`
- `reweight_rate`

### Required Inputs

- `--corpus-path`: path to the corpus JSONL
- `--traj-dir`: directory containing trajectory JSON files
- `--output-path`: output JSONL path
- `--tokenizer-path`: tokenizer path used to count reasoning length
- `--judge-api-url`: OpenAI-compatible judge endpoint

### Example

```bash
python src/data_builder.py \
  --corpus-path /path/to/your/corpus.jsonl \
  --traj-dir /path/to/your/trajectory_dir \
  --output-path /path/to/save/output.jsonl \
  --tokenizer-path /path/to/your/tokenizer_or_model_dir \
  --judge-api-url http://<JUDGE_HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL_NAME> \
  --max-workers 32 \
  --future-timeout 30
```

## 2. Segmented Builder

Script:

```text
src/data_builder_segmented.py
```

This variant keeps only a chosen portion of browse actions within each search run.

Supported modes:

- `front30`
- `middle30`
- `back30`
- `full100`

### Extra Arguments

- `--segment-mode`: which segment to keep
- `--segment-save-tag`: optional metadata tag saved into each sample
- `--min-browse-per-search`: skip runs with too few browse actions
- `--segment-boundary`: optional JSON override for segment ranges
- `--summary-path`: optional JSON summary output

### Example: Front Segment

```bash
python src/data_builder_segmented.py \
  --segment-mode front30 \
  --corpus-path /path/to/your/corpus.jsonl \
  --traj-dir /path/to/your/trajectory_dir \
  --output-path training_data/segmented/front30.jsonl \
  --summary-path training_data/logs/front30_summary.json \
  --tokenizer-path /path/to/your/tokenizer_or_model_dir \
  --judge-api-url http://<JUDGE_HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL_NAME> \
  --max-workers 32 \
  --future-timeout 30
```

### Example: Custom Boundaries

```bash
python src/data_builder_segmented.py \
  --segment-mode middle30 \
  --segment-boundary "{\"front\":[0,0.3],\"middle\":[0.35,0.65],\"back\":[0.7,1.0]}" \
  --corpus-path /path/to/your/corpus.jsonl \
  --traj-dir /path/to/your/trajectory_dir \
  --output-path training_data/segmented/middle30.jsonl \
  --tokenizer-path /path/to/your/tokenizer_or_model_dir \
  --judge-api-url http://<JUDGE_HOST>:<PORT>/v1/chat/completions
```

## 3. Merge Segmented Outputs

Script:

```text
src/merge_segmented_data.py
```

This script merges multiple JSONL files into one final dataset.

Supported merge policies:

- `equal`: downsample each input to the smallest size and merge them
- `concat`: keep all samples and concatenate them

### Example

```bash
python src/merge_segmented_data.py \
  --inputs \
    training_data/segmented/front30.jsonl \
    training_data/segmented/middle30.jsonl \
    training_data/segmented/back30.jsonl \
    training_data/segmented/full100.jsonl \
  --output-path training_data/segmented/mix_final.jsonl \
  --sample-mode equal \
  --shuffle \
  --summary-path training_data/logs/mix_final_summary.json
```

## 4. Quick Statistics

Script:

```text
src/training_data_stats.py
```

This helper computes a quick JSON summary for a training-data JSONL file.

Reported fields include:

- sample count
- unique query count
- average negative count
- satisfied ratio
- reasoning length mean / median / P90
- empty-sample ratio
- segment counts

### Example

```bash
python src/training_data_stats.py \
  --input-path training_data/segmented/front30.jsonl \
  --output-path training_data/logs/front30_stats.json
```

## 5. Recommended Workflow

1. Generate full trajectories with one of the agent clients.
2. Run `src/data_builder.py` for the baseline full-trajectory training set.
3. Run `src/data_builder_segmented.py` for `front30`, `middle30`, `back30`, and `full100`.
4. Inspect the generated JSON summaries.
5. Merge segmented subsets with `src/merge_segmented_data.py` if you want a mixed final set.
6. Use the resulting JSONL file in your existing FlagEmbedding training recipe.

## 6. Common Pitfalls

- Keep trajectory `query_id` stable if you also plan to evaluate later.
- Make sure the corpus used here matches the corpus that the trajectories retrieved from.
- If a judge endpoint is slow, increase `--future-timeout`.
- If `--output-path` is in the current directory, the scripts now handle that case correctly.
