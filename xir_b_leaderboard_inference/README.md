# XIR-B BrowseComp-Plus inference

This folder contains the organizer-side inference runner used to compare the
six submitted retrievers with three research-agent backends:

- DeepSeek-V4-Flash-0731;
- GPT-OSS-120B; and
- Qwen3.5-122B-A10B-FP8.

The official competition page is the source of truth for rules and rankings:
[CCIR CUP / XIR competition 1170](https://www.xir.cn/competition/1170/ranking?sch=2660).
This folder documents the inference implementation; it does not replace the
competition rules.

## Evaluation contract

- Input: the 830 fixed rows in [`../topics-qrels/queries.tsv`](../topics-qrels/queries.tsv).
- Retrieval: six independent local MCP services, one per submitted retriever.
- Work per model: `830 queries × 6 retrievers = 4,980 trajectories`.
- Agent behavior: the model autonomously chooses `search`, `get_document`, or
  a final response through `tool_choice=auto`; the runner does not impose a
  fixed tool order.
- **Hard upper bound: 50 model/Agent calls per trajectory.** Tool calls returned
  by one model response do not consume extra model-call slots. The worker's
  `_charge_llm()` updates both `interaction_steps` and `llm_call_count`, and
  refuses call 51.
- The last available chat-completions turn is reserved for a tool-free final
  answer. GPT-OSS keeps the official Harmony Responses tool protocol through
  its 50-call loop.
- Per-call output cap: 4,096 tokens. A `get_document` result contributes at
  most 8,192 tokenizer tokens to the model-facing transcript; the raw tool
  return is retained in the trajectory audit.
- Default concurrency: 32 query workers.
- No judge runs during trajectory generation.

Natural model completion can occur before call 50. A trajectory that reaches
the limit is accepted as `terminated_by=max_steps_limit` only when the stored
record proves 50 healthy model turns, valid tool calls/results, nonempty raw
history, and no API, MCP, model, process, context, or serialization error. The
runner never retries solely because of that healthy terminal condition.

## Layout

| Path | Purpose |
| --- | --- |
| `configs/campaign.example.json` | Three model profiles, six retriever slots, ports, budgets, and pinned model revisions |
| `search_agent/query_worker.py` | Autonomous model/tool loop and atomic trajectory writer |
| `search_agent/campaign_common.py` | Configuration, strict trajectory validation, and max-step policy |
| `scripts/run_shard.py` | Resumable 830-query sharding with 32-way execution |
| `scripts/validate_run.py` | Strict all-expected-trajectories validator |
| `servers/` | Reference vLLM commands for the three model families |
| `searcher/README.md` | Six-service MCP retrieval setup |
| `tests/` | Offline agent-loop, tool-schema, budget, and 50-call regressions |

Generated weights, indexes, corpora, API credentials, trajectories, and logs
are intentionally not committed.

## 1. Install the runner environment

Use a Python 3.10+ environment. The model server can live in a separate,
model-specific environment.

```bash
python -m venv .venv-xir-b
source .venv-xir-b/bin/activate
python -m pip install -r xir_b_leaderboard_inference/requirements.txt
```

Do not put Hugging Face or API tokens in the configuration or process command
line. Use the provider's normal login/environment mechanism when required.

## 2. Configure paths

Copy the example and replace only local model, index, retriever, corpus, Python,
and output paths:

```bash
cp xir_b_leaderboard_inference/configs/campaign.example.json \
  xir_b_leaderboard_inference/configs/campaign.local.json
```

Keep `max_interaction_steps=50`. The six retriever entries preserve the
submission identities, while their index/model/corpus paths are local
placeholders. Match each retriever's declared pooling, normalization, and
prefix settings when launching its MCP service.

## 3. Start six retrieval services

Follow [`searcher/README.md`](searcher/README.md). The default ports are
`38190`–`38195`, and every service must expose both `search` and
`get_document`. The runner checks all six TCP endpoints before claiming work.

## 4. Start one model service

The reference scripts fail closed unless `MODEL_PATH` is explicit:

```bash
MODEL_PATH=/models/DeepSeek-V4-Flash-0731 \
  xir_b_leaderboard_inference/servers/serve_deepseek_v4_flash.sh

MODEL_PATH=/models/gpt-oss-120b \
  xir_b_leaderboard_inference/servers/serve_gpt_oss_120b.sh

MODEL_PATH=/models/Qwen3.5-122B-A10B-FP8 \
  xir_b_leaderboard_inference/servers/serve_qwen35_122b.sh
```

Run only the service matching the selected profile, or assign non-overlapping
GPUs and ports. DeepSeek uses its `deepseek_v4` tokenizer/reasoning/tool
parsers; Qwen uses one official chat template with `qwen3` reasoning and
`qwen3_coder` tool parsing; GPT-OSS uses the backend's Harmony-native Responses
path and deliberately has no Qwen/DeepSeek parser flags.

## 5. Run inference

One host can run all 830 source rows:

```bash
python xir_b_leaderboard_inference/scripts/run_shard.py \
  --profile deepseek-v4-flash \
  --shard-index 0 \
  --shard-count 1 \
  --run-id ds-v4-flash-xir-b \
  --concurrency 32 \
  --attempt-limit 5 \
  --worker-timeout 900 \
  --config xir_b_leaderboard_inference/configs/campaign.local.json
```

For two hosts, use the same `run-id`, `shard-count 2`, and distinct
`shard-index 0` and `1`. Sharding follows fixed row order rather than numeric
query IDs. Existing strictly valid trajectories are reused; a
`context_exhausted` record is permanent and is not replayed with the same full
context.

Run the other model profiles by changing `--profile` to `gpt-oss-120b` or
`qwen3.5-122b-a10b-fp8` and selecting a new run ID.

## 6. Validate a run

```bash
python xir_b_leaderboard_inference/scripts/validate_run.py \
  --profile deepseek-v4-flash \
  --run-id ds-v4-flash-xir-b \
  --shard-index 0 \
  --shard-count 1 \
  --config xir_b_leaderboard_inference/configs/campaign.local.json
```

The validator succeeds only if all `4,980` expected trajectory records for
that model pass. A two-host shard expects `415 × 6 = 2,490` valid records on
each host.

## Tests

```bash
python -m pytest -q xir_b_leaderboard_inference/tests
```

The tests use declared fake model decisions and do not contact a model or MCP
service. In particular, `test_step50_cap.py` checks the configuration, runner,
worker, and validator for the same exact 50-call contract.
