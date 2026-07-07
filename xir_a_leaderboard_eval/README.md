# XIR A-Leaderboard BrowseComp-Plus Evaluation

This folder contains the actual Qwen3.5-4B inference and evaluation scripts used for the XIR competition A-leaderboard setting:

https://www.xir.cn/competition/1170

The evaluated submission is a Hugging Face dense retriever repository. The expected retriever family is Qwen3-Embedding-0.6B-compatible dense retrievers.

## Setting

- Dataset: BrowseComp-Plus
- Retriever: dense retriever from a Hugging Face repo
- Agent model: Qwen/Qwen3.5-4B served by vLLM
- Judge model: Qwen/Qwen3.5-4B
- Tools: `search` and `get_document`
- Tool/LLM-call budget: 50
- Persistent workspace root: `/mnt/bn/search-tiktok-nas-au/yuqizhou/competition`

## One-command Evaluation

On the evaluation server:

```bash
cd /mnt/bn/search-tiktok-nas-au/yuqizhou/competition
bash /path/to/xir_a_leaderboard_eval/run_browsecomp_plus_eval.sh \
  --skip-llm-compat-check \
  --index-gpu-devices 0,1,2,3,4,5,6,7 \
  --index-shards 8 \
  --retrieval-batch-size 128 \
  --attn-implementation flash_attention_2 \
  --agent-gpu-devices 0,1,2,3,4,5,6,7 \
  --tensor-parallel-size 8 \
  --vllm-max-model-len 262144 \
  --vllm-max-num-seqs 16 \
  --max-tokens 4096 \
  --search-gpu-devices 0,1,2,3 \
  --agent-workers 64 \
  --keep-models \
  --keep-indexes \
  Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B
```

The script records models, indexes, logs, raw agent outputs, and judge outputs under `/mnt/bn/search-tiktok-nas-au/yuqizhou/competition`.

## Fast Resume With Existing vLLM

For repeated A-leaderboard runs on the same worker, keep the four Qwen3.5 vLLM OpenAI-compatible servers alive on ports `8001-8004`, then use:

```bash
python xir_a_leaderboard_eval/resume_existing_vllm_w64_multigpu_search.py \
  --retriever-repo Yuqi-Zhou/LRAT-Qwen3-Embedding-0.6B \
  --max-tokens 4096
```

This runner:

- reuses existing vLLM servers;
- starts only the MCP retrieval server;
- uses multi-GPU search, normally `CUDA_VISIBLE_DEVICES=0,1,2,3`;
- launches 64 Qwen-agent workers;
- caps per-call output tokens at 4096 while relying on the vLLM server `--max-model-len` for the long context window;
- preloads existing `run_*.json` outputs by `query_id` and skips completed queries.

## Search-Speed Patch

`patch_search_tools_cache.py` adds exact-query, snippet, and `get_document` caches to BrowseComp-Plus `searcher/tools.py`. The current A-leaderboard run uses this patch because the agent often repeats exact or near-exact search queries.

```bash
python xir_a_leaderboard_eval/patch_search_tools_cache.py \
  /mnt/bn/search-tiktok-nas-au/yuqizhou/competition/src/BrowseComp-Plus/searcher/tools.py
```

## Monitor

```bash
python xir_a_leaderboard_eval/monitor_eval.py \
  --root /mnt/bn/search-tiktok-nas-au/yuqizhou/competition \
  --run-glob '*qwen35*multigpu_search*'
```

It prints result count, recent throughput, live process counts, vLLM queue metrics, MCP errors, and GPU memory/utilization.
