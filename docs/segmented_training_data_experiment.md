# Segmented Training-Data Experiment

This note describes a clean way to run segmented trajectory experiments in this repository.

## Goal

Instead of training a retriever from all browse actions inside agent trajectories, we split browse behavior by relative position within each search run and compare:

- `front30`
- `middle30`
- `back30`
- `full100`
- `mix_final`

The question is whether early, middle, late, or mixed browse behavior produces better retriever supervision.

## Scripts

The experiment is now supported by the following utilities:

- `src/data_builder.py`
- `src/data_builder_segmented.py`
- `src/merge_segmented_data.py`
- `src/training_data_stats.py`

## Segment Definition

Within each search run, browse actions are assigned a normalized position:

```text
position = browse_idx / max(1, browse_count - 1)
```

Default ranges:

- `front30`: `[0.00, 0.30)`
- `middle30`: `[0.35, 0.65)`
- `back30`: `[0.70, 1.00]`
- `full100`: keep everything

You can override these boundaries through `--segment-boundary`.

## Suggested Directory Layout

```text
training_data/
  segmented/
    front30.jsonl
    middle30.jsonl
    back30.jsonl
    full100.jsonl
    mix_final.jsonl
  logs/
    front30_summary.json
    middle30_summary.json
    back30_summary.json
    full100_summary.json
    mix_final_summary.json
```

## Step 1. Build the Baseline Full Set

```bash
python src/data_builder.py \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/trajectory_dir \
  --output-path training_data/segmented/full_baseline.jsonl \
  --tokenizer-path /path/to/tokenizer_or_model \
  --judge-api-url http://<HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL> \
  --max-workers 32 \
  --future-timeout 30
```

## Step 2. Build the Four Segmented Sets

### front30

```bash
python src/data_builder_segmented.py \
  --segment-mode front30 \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/trajectory_dir \
  --output-path training_data/segmented/front30.jsonl \
  --summary-path training_data/logs/front30_summary.json \
  --tokenizer-path /path/to/tokenizer_or_model \
  --judge-api-url http://<HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL> \
  --max-workers 32 \
  --future-timeout 30
```

### middle30

```bash
python src/data_builder_segmented.py \
  --segment-mode middle30 \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/trajectory_dir \
  --output-path training_data/segmented/middle30.jsonl \
  --summary-path training_data/logs/middle30_summary.json \
  --tokenizer-path /path/to/tokenizer_or_model \
  --judge-api-url http://<HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL> \
  --max-workers 32 \
  --future-timeout 30
```

### back30

```bash
python src/data_builder_segmented.py \
  --segment-mode back30 \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/trajectory_dir \
  --output-path training_data/segmented/back30.jsonl \
  --summary-path training_data/logs/back30_summary.json \
  --tokenizer-path /path/to/tokenizer_or_model \
  --judge-api-url http://<HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL> \
  --max-workers 32 \
  --future-timeout 30
```

### full100

```bash
python src/data_builder_segmented.py \
  --segment-mode full100 \
  --corpus-path /path/to/corpus.jsonl \
  --traj-dir /path/to/trajectory_dir \
  --output-path training_data/segmented/full100.jsonl \
  --summary-path training_data/logs/full100_summary.json \
  --tokenizer-path /path/to/tokenizer_or_model \
  --judge-api-url http://<HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL> \
  --max-workers 32 \
  --future-timeout 30
```

## Step 3. Merge into mix_final

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

## Step 4. Check Data Quality

Run quick stats on each generated file:

```bash
python src/training_data_stats.py \
  --input-path training_data/segmented/front30.jsonl \
  --output-path training_data/logs/front30_stats.json
```

Recommended checks:

- sample count
- unique query count
- average negative count
- satisfied true / false ratio
- reasoning length mean / median / P90
- empty sample ratio
- segment distribution

## Suggested Training Matrix

Keep the training recipe fixed and only swap the training data:

1. `full100`
2. `front30`
3. `middle30`
4. `back30`
5. `mix_final`

Recommended training entry:

```text
FlagEmbedding/examples/finetune/embedder/run.sh
```

## Evaluation Tracking

Evaluate each trained retriever with the same judge and benchmark setup.

A simple results table can look like this:

```text
exp_name,train_data,seed,retriever_model,eval_dataset,Recall@10,nDCG@10,MRR@10,notes
```

## Common Pitfalls

- Make sure the trajectory corpus and the training-data corpus are the same source
- Do not compare runs with different total sample counts unless you explicitly want that effect
- If the judge endpoint is slow, increase `--future-timeout`
- Keep the random seed fixed when merging segmented datasets
