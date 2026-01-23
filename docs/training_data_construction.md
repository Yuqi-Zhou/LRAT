## Build Training Pairs from Trajectories

This script reads:
1) a corpus file in JSONL format (`{"docid": ..., "text": ...}` per line),  
2) a directory of trajectory JSON files (each must contain a `result` field),  
then calls an OpenAI-compatible **LLM Judge** API to label each browse step as `RELEVANT/NOT_RELEVANT`, and finally writes extracted `(query, pos, neg, ...)` samples to a JSONL output file.

### Arguments

- `--corpus-path` (**required**): Path to the corpus JSONL file (`docid/text` per line).
- `--traj-dir` (**required**): Directory containing trajectory JSON files.
- `--output-path` (**required**): Output path for generated training samples (JSONL).
- `--tokenizer-path` (**required**): Tokenizer path (used to count `reasoning_len`).
- `--judge-api-url` (**required**): OpenAI-compatible endpoint, e.g. `http://host:8000/v1/chat/completions`.
- `--judge-model` (default: empty): Model name passed to the judge API.
- `--judge-headers` (default: empty): Optional HTTP headers as a JSON string, e.g.
  `{"Authorization":"Bearer YOUR_KEY","Content-Type":"application/json"}`.
- `--max-workers` (default: `32`): Number of parallel worker processes.
- `--future-timeout` (default: `15`): Timeout (seconds) per trajectory job.

### Example

```bash
python build_pairs.py \
  --corpus-path /path/to/your/corpus.jsonl \
  --traj-dir /path/to/your/trajectory_dir \
  --output-path /path/to/save/output.jsonl \
  --tokenizer-path /path/to/your/tokenizer_or_model_dir \
  --judge-api-url http://<JUDGE_HOST>:<PORT>/v1/chat/completions \
  --judge-model <JUDGE_MODEL_NAME> \
  --max-workers 32 \
  --future-timeout 30

