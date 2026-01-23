## Corpus Format Fix (Required)

Due to the original dataset format, please first convert the corpus `.jsonl` file so that **each line** follows this schema:

```json
{"docid": "5412", "text": "xxx"}
```

Notes:
- `docid` must be a **string**
- `text` is the document content used for indexing/retrieval

---

## Building Indexes

After downloading the dataset and setting up the environment, you can build retrieval indexes.

### 1) Build a BM25 Index

```bash
corpus_file=/path/to/your/corpus.jsonl
save_dir=/path/to/save/bm25_index
retriever_name=bm25

python src/index_builder.py \
  --retrieval_method ${retriever_name} \
  --corpus_path ${corpus_file} \
  --save_dir ${save_dir}
```

---

### 2) Build Dense Embeddings (FAISS / Embedding Index)

This command encodes the corpus into embedding shards (e.g., `.pkl`) that can be used by a FAISS-based dense retriever.

Pooling recommendation:
- For **Qwen3 embedding models**: use `--pooling eos`
- For **E5 (e5-large, etc.)**: use `--pooling mean`

Template:

```bash
#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="/path/to/your/corpus.jsonl"
MODEL_PATH="/path/to/your/embedding_model"
OUTPUT_PKL="/path/to/save/embeddings.pkl"

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --encode_output_path "${OUTPUT_PKL}" \
  --passage_max_len 512 \
  --normalize \
  --pooling <eos|mean> \
  --passage_prefix "" \
  --per_device_eval_batch_size 512 \
  --padding_side left \
  --fp16
```

If you want multiple shards, run encoding multiple times with different `CUDA_VISIBLE_DEVICES` and different output names, or follow your existing sharding pipeline (e.g., `embeddings-*.pkl`).
