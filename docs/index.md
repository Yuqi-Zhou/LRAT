# Index Construction

This document summarizes how to prepare retrieval indexes for the repository.

## Overview

The repository currently supports three retrieval backends:

- `bm25`: lexical retrieval backed by Lucene / Pyserini
- `faiss`: dense retrieval over embedding shards
- `reasonir`: dense retrieval with a ReasonIR-style encoder

The main local entry point for index construction is:

```bash
python src/index_builder.py ...
```

## Corpus Format

There are two slightly different corpus conventions in the current repository:

- `src/data_builder.py` and `searcher/searchers/faiss_searcher.py` expect records that expose `docid` and `text`
- `searcher/searchers/bm25_searcher.py` reads indexed raw documents from Lucene and expects stored `contents`

For that reason, before building indexes, it is worth double-checking which stage you are preparing data for.

### Recommended JSONL for Training Data and FAISS Lookup

```json
{"docid": "5412", "text": "document content"}
```

### BM25 Note

If you build a BM25 index through Pyserini and later use `searcher/searchers/bm25_searcher.py`, make sure the indexed raw JSON keeps a `contents` field available in the Lucene document payload.

In practice, if your source corpus is not already aligned, prepare a BM25-specific JSONL variant and verify that `raw["contents"]` is present after indexing.

## BM25 Index

Entry point:

```bash
python src/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path /path/to/your/bm25_corpus.jsonl \
  --save_dir /path/to/save/index
```

### Notes

- `--corpus_path` should point to the corpus file used for Lucene indexing
- `--save_dir` is the output folder for the BM25 index
- Java 21 and Pyserini dependencies should already be available in the environment

## Dense Index

The repository uses Tevatron-style encoding to produce embedding shards and then reads them through the FAISS searcher.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path /path/to/your/embedding_model \
  --dataset_path /path/to/your/corpus.jsonl \
  --encode_output_path /path/to/save/embeddings.pkl \
  --passage_max_len 512 \
  --normalize \
  --pooling <eos|mean> \
  --passage_prefix "" \
  --per_device_eval_batch_size 512 \
  --padding_side left \
  --fp16
```

### Pooling Recommendations

- Qwen3 embedding models: `--pooling eos`
- E5-style models: `--pooling mean`

### Sharded Embeddings

If you want to create multiple shards, run encoding multiple times and save them with a glob-friendly pattern such as:

```text
embeddings/index-000.pkl
embeddings/index-001.pkl
embeddings/index-002.pkl
```

These shards can later be consumed by:

```bash
--index-path "/path/to/embeddings/index-*.pkl"
```

## Searcher-Side Dense Retrieval Inputs

When you later run a dense agent client, the FAISS searcher additionally needs:

- `--model-name`: embedding model used to encode queries
- `--dataset-name`: corpus source used for document lookup
- `--pooling`: query embedding pooling strategy
- `--normalize`: whether to normalize embeddings

Example shape:

```bash
--searcher-type faiss \
--index-path "/path/to/embeddings/index-*.pkl" \
--model-name /path/to/embedding_model \
--dataset-name /path/to/corpus_or_dataset \
--pooling eos \
--normalize
```

## Optional: Retrieval as an MCP Server

If you want to expose the local retriever as an MCP endpoint, use:

```bash
python searcher/mcp_server.py \
  --searcher-type faiss \
  --index-path "/path/to/embeddings/index-*.pkl" \
  --model-name /path/to/embedding_model \
  --dataset-name /path/to/corpus_or_dataset \
  --pooling eos \
  --normalize \
  --transport sse \
  --port 8000
```

Useful flags:

- `--get-document`: also register the `get_document` tool
- `--transport`: one of `stdio`, `streamable-http`, or `sse`
- `--public`: create an ngrok-backed public endpoint

## Quick Checklist

- Confirm which corpus schema each stage expects: `text` or `contents`
- Build BM25 and dense assets separately if needed
- Keep dense embedding shard paths glob-friendly
- Verify `--dataset-name` resolves correctly before running FAISS-based agent clients
