# Running the Search Agent (BM25 / Dense Retriever)

After you prepare the environment and required files (indexes, models, datasets), you can run the scripts below. **All paths and values are placeholders**—replace them with your own.

---

## 0) Start the model service with vLLM (template)

Choose one model/agent to serve (examples shown as templates):

```bash
# Example: AgentCPM-Explore
vllm serve /path/to/your/AgentCPM-Explore \
  --port <PORT> \
  --tensor-parallel-size <TP_SIZE> \
  --gpu-memory-utilization <GPU_MEM_UTIL> \
  --enable-reasoning

# Example: WebExplorer
vllm serve /path/to/your/WebExplorer \
  --port <PORT> \
  --tensor-parallel-size <TP_SIZE> \
  --gpu-memory-utilization <GPU_MEM_UTIL> \
  --enable-reasoning

# Example: Tongyi DeepResearch
vllm serve /path/to/your/Tongyi-DeepResearch \
  --port <PORT> \
  --tensor-parallel-size <TP_SIZE> \
  --gpu-memory-utilization <GPU_MEM_UTIL>

# Example: GPT-OSS (when trust-remote-code is needed)
vllm serve /path/to/your/GPT-OSS \
  --port <PORT> \
  --tensor-parallel-size <TP_SIZE> \
  --gpu-memory-utilization <GPU_MEM_UTIL> \
  --trust-remote-code
```

Below we use **Tongyi agent** as an example and provide templates for generating trajectories with BM25 and a dense retriever (FAISS).

---

## 1) BM25 Retriever (generate trajectories)

Script: `search_agent/tongyi_client.py`

```bash
python search_agent/tongyi_client.py \
  --output-dir /path/to/output/dir \
  --searcher-type bm25 \
  --index-path /path/to/bm25/index/dir \
  --num-threads 32 \
  --model /path/to/agent_or_llm_dir \
  --snippet-max-tokens 64 \
  --query /path/to/queries.tsv \
  --port <PORT> \
  --k 10
```

---

## 2) Dense Retriever (FAISS, generate trajectories)

Script: `search_agent/tongyi_client.py`

> Note: Dense retrieval requires an embedding index (e.g., sharded `*.pkl`) and an embedding model path. Also, make pooling explicit.

```bash
python search_agent/tongyi_client.py \
  --output-dir /path/to/output/dir \
  --searcher-type faiss \
  --index-path "/path/to/embeddings/index-*.pkl" \
  --model-name /path/to/embedding/model \
  --pooling <mean|eos> \
  --normalize \
  --num-threads 32 \
  --snippet-max-tokens 64 \
  --query /path/to/queries.tsv \
  --port <PORT> \
  --dataset-name /path/to/corpus_or_dataset \
  --model /path/to/agent_or_llm_dir \
  --k 10
```

- `--pooling`: `mean` or `eos` (default: `eos`; choose based on your embedding model)

---

## 3) API-deployed models (e.g., MiniMax / GLM)

For models served via an OpenAI-compatible API (instead of local vLLM), set your environment variables first:

```bash
export API_KEY=<YOUR_API_KEY>
export URL=<YOUR_OPENAI_COMPATIBLE_BASE_URL>
```

Then run (template):

Script: `search_agent/openai_client.py`

```bash
python search_agent/openai_client.py \
  --model <AGENT_NAME> \
  --output-dir /path/to/output/dir \
  --searcher-type faiss \
  --index-path "/path/to/embeddings/index-*.pkl" \
  --model-name /path/to/embedding/model \
  --pooling <mean|eos> \
  --normalize \
  --num-threads 32 \
  --max-tokens <MAX_TOKENS> \
  --query-template <QUERY_TEMPLATE_NAME_OR_PATH> \
  --snippet-max-tokens 64 \
  --get-document \
  --dataset-name /path/to/corpus_dir \
  --max-iterations <MAX_ITERS> \
  --query /path/to/queries.tsv \
  --k 10
```

---

## Notes (quick)

- `--searcher-type bm25`: lexical BM25 retrieval (`--index-path` should be your BM25 index directory)
- `--searcher-type faiss`: dense retrieval (`--index-path` usually matches embedding shards like `index-*.pkl`)
- `--model-name`: embedding model path (used to encode queries)
- `--normalize`: enable embedding normalization (commonly for cosine similarity)
- `--pooling`: embedding pooling strategy (`mean` / `eos`)
- `--k`: top-k retrieved docs per query
- `--snippet-max-tokens`: truncation for retrieved snippets
- `--num-threads`: concurrency
- `--port`: must match the served model port
