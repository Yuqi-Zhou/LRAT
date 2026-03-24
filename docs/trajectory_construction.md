# Trajectory Construction

This document shows how to generate agent trajectories with the repository's local retrieval tools.

## Overview

Trajectory files are saved as one JSON file per query and can later be used for:

- training-data construction,
- qualitative inspection, and
- end-to-end benchmark evaluation.

The repository currently provides four main trajectory-generation entry points:

- `search_agent/tongyi_client.py`
- `search_agent/webexplorer_client.py`
- `search_agent/agentcmp_client.py`
- `search_agent/openai_client.py`

There is also a vLLM Responses-API-oriented client:

- `search_agent/oss_client.py`

## 1. Start the Model Service

For local model serving, start your model with vLLM first.

Example:

```bash
vllm serve /path/to/your/model \
  --port <PORT> \
  --tensor-parallel-size <TP_SIZE> \
  --gpu-memory-utilization <GPU_MEM_UTIL>
```

You can swap in different models depending on the client you want to test, such as:

- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`
- `hkust-nlp/WebExplorer-8B`
- `openbmb/AgentCPM-Explore`
- `openai/gpt-oss-*`

## 2. Common Retrieval Arguments

Most clients share the same retrieval-side options:

| Argument | Meaning |
| --- | --- |
| `--searcher-type` | `bm25`, `faiss`, `reasonir`, or other registered searchers |
| `--index-path` | BM25 index directory or dense-embedding glob |
| `--model-name` | Query-side embedding model for dense retrieval |
| `--dataset-name` | Corpus path / dataset used for document lookup in dense retrieval |
| `--pooling` | Pooling strategy such as `eos` or `mean` |
| `--normalize` | Enable query embedding normalization |
| `--snippet-max-tokens` | Snippet truncation length |
| `--k` | Number of retrieved results |
| `--query` | Either a single query string or a TSV dataset |
| `--output-dir` | Directory for saved run JSON files |

## 3. Tongyi Client

Script:

```text
search_agent/tongyi_client.py
```

### BM25 Example

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

### FAISS Example

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

### Single-Query Mode

You can also pass a literal question through `--query`, not only a TSV file.

## 4. WebExplorer Client

Script:

```text
search_agent/webexplorer_client.py
```

Example:

```bash
python search_agent/webexplorer_client.py \
  --output-dir /path/to/output/dir \
  --searcher-type faiss \
  --index-path "/path/to/embeddings/index-*.pkl" \
  --model-name /path/to/embedding/model \
  --pooling eos \
  --normalize \
  --num-threads 16 \
  --query /path/to/queries.tsv \
  --dataset-name /path/to/corpus_or_dataset \
  --model /path/to/webexplorer_model \
  --port <PORT> \
  --k 10
```

## 5. AgentCPM Client

Script:

```text
search_agent/agentcmp_client.py
```

Example:

```bash
python search_agent/agentcmp_client.py \
  --output-dir /path/to/output/dir \
  --searcher-type bm25 \
  --index-path /path/to/bm25/index/dir \
  --num-threads 16 \
  --model /path/to/agentcpm_model \
  --query /path/to/queries.tsv \
  --port <PORT> \
  --k 10
```

## 6. OpenAI-Compatible API Client

Script:

```text
search_agent/openai_client.py
```

Before running it, set your API credentials.

Supported environment-variable fallbacks:

- API key: `API_KEYS` or `API_KEY`
- Base URL: `URL` or `BASE_URL`

Example:

```bash
export API_KEY=<YOUR_API_KEY>
export BASE_URL=<YOUR_OPENAI_COMPATIBLE_BASE_URL>
```

Then run:

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
  --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT \
  --snippet-max-tokens 64 \
  --get-document \
  --dataset-name /path/to/corpus_dir \
  --max-iterations <MAX_ITERS> \
  --query /path/to/queries.tsv \
  --k 10
```

This client now supports both TSV-dataset mode and single-query mode.

## 7. vLLM Responses API Client

Script:

```text
search_agent/oss_client.py
```

Example:

```bash
python search_agent/oss_client.py \
  --model openai/gpt-oss-20b \
  --model-url http://localhost:8000/v1 \
  --output-dir /path/to/output/dir \
  --searcher-type faiss \
  --index-path "/path/to/embeddings/index-*.pkl" \
  --model-name /path/to/embedding/model \
  --dataset-name /path/to/corpus_or_dataset \
  --pooling eos \
  --normalize \
  --query /path/to/queries.tsv \
  --query-template QUERY_TEMPLATE_NO_GET_DOCUMENT \
  --num-threads 8 \
  --max-iterations 50 \
  --k 10
```

Useful optional flags:

- `--reasoning-effort`
- `--get-document`
- `--hf-token`
- `--hf-home`
- `--verbose`

## 8. Output Format

Each processed query is saved as one `run_*.json` file under `--output-dir`.

The output typically includes:

- `query_id`
- `tool_call_counts`
- `retrieved_docids`
- `result`
- `raw_messages`

These files are the inputs expected by:

- `src/data_builder.py`
- `src/data_builder_segmented.py`
- `scripts_evaluation/evaluate.py`

## 9. Common Tips

- Use TSV dataset mode when you want resumable batch generation
- Keep `query_id` stable across trajectory generation and evaluation
- Match `--port` to the port used in `vllm serve`
- For dense retrieval, make sure `--model-name`, `--dataset-name`, and `--index-path` belong to the same corpus / embedding setup
