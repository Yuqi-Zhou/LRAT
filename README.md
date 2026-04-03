# Learning to Retrieve from Agent Trajectories

Retrieval is no longer optimized only for human searchers. As large language model agents increasingly issue queries, inspect snippets, browse documents, and reason over retrieved evidence, the target of retrieval training has shifted from human interaction to agent interaction. **LRAT** studies this paradigm shift and learns retrievers directly from multi-step agent trajectories.

<p align="center">
  <img src="assets/paper-figures/intro_5.png" width="92%" alt="Paradigm shift from human-centric retrieval training to agent-centric retrieval learning">
</p>

<p align="center">
  <em>Modern retrieval supervision should match how search agents actually search, browse, and consume evidence.</em>
</p>

<p align="center">
  <a href="https://yuqi-zhou.github.io/LRAT-homepage/">
    <img src="https://img.shields.io/badge/Homepage-LRAT-2563eb.svg?style=flat&logo=githubpages&logoColor=white" alt="Homepage">
  </a>
  <a href="https://github.com/Yuqi-Zhou/LRAT">
    <img src="https://img.shields.io/badge/GitHub-Code-blue.svg?style=flat&logo=github" alt="GitHub Code">
  </a>
  <a href="https://huggingface.co/collections/Yuqi-Zhou/lrat">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Collection-ffbd45.svg?style=flat" alt="Hugging Face Collection">
  </a>
  <a href="https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_HuggingFace-Dataset-ffbd45.svg?style=flat" alt="Hugging Face Dataset">
  </a>
  <a href="assets/paper/LRAT.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-b31b1b?logo=arxiv&logoColor=white&style=flat" alt="Paper PDF">
  </a>
</p>

## Introduction

**LRAT** studies how to train retrievers from the intermediate behaviors of strong search agents rather than from only final answers. The repository focuses on a practical pipeline for:

- collecting long-horizon search trajectories from agentic systems,
- converting trajectories into retrieval supervision,
- training retrievers on the resulting samples, and
- evaluating both retrieval quality and end-to-end task success.

<p align="center">
  <img src="assets/paper-figures/method_7_readme.png" width="92%" alt="LRAT method overview from the paper">
</p>

## News

- `2026/04/03`: Paper accepted to **SIGIR 2026**.
- `2026/03/24`: Open-sourced [model checkpoints](https://huggingface.co/collections/Yuqi-Zhou/lrat) and the [LRAT-Train dataset](https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train).

## Highlights

- **Trajectory-first retrieval learning**: build retriever supervision from agent search and browse traces instead of relying only on static relevance labels.
- **Agent-friendly data collection**: run local or API-based research agents and save each query as structured trajectory JSON.
- **Training data construction with an LLM judge**: turn trajectories into `(query, pos, neg, ...)` training pairs with reasoning-aware annotations.
- **Benchmark-oriented evaluation**: evaluate outputs on `BrowseComp-Plus` and `InfoSeek-Eval` with a local vLLM judge.

## Resources

| Resource | Status |
| --- | --- |
| Homepage | [Homepage](https://yuqi-zhou.github.io/LRAT-homepage/) |
| Model Checkpoint | [LRAT Collection](https://huggingface.co/collections/Yuqi-Zhou/lrat) |
| Dataset Release | [LRAT-Train](https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train) |
| Paper | [PDF](assets/paper/LRAT.pdf) · Accepted by **SIGIR 2026** |

## Repository Overview

| Path | Description |
| --- | --- |
| `src/` | Core utilities for index construction and trajectory-to-training-data conversion |
| `search_agent/` | Agent clients for Tongyi DeepResearch, WebExplorer, AgentCPM, OpenAI-compatible APIs, and related prompts/utilities |
| `searcher/` | Search backends and local retrieval interfaces |
| `docs/` | Step-by-step documentation for indexing, trajectory construction, training data construction, and evaluation |
| `datasets/` | Benchmark files used in evaluation |
| `topics-qrels/` | Query and qrel files for retrieval experiments |
| `trajectory/` | Example trajectory artifacts |
| `FlagEmbedding/` | Local copy of FlagEmbedding used for retriever training |
| `tevatron/` | Local copy of Tevatron utilities used in dense retrieval workflows |
| `scripts_evaluation/` | Evaluation scripts for end-to-end judging |

## Vendored Dependencies

- `FlagEmbedding/` is a vendored and locally modified copy based on the upstream FlagEmbedding project. In this repository, it reflects user-side modifications layered on top of upstream work and earlier external changes.
- `tevatron/` is a vendored upstream dependency used to support dense retrieval utilities and encoding workflows.
- More details are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Supported Components

### Retriever Backends

- `bm25`
- `faiss`

### Agent / LLM Backends

- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`
- `hkust-nlp/WebExplorer-8B`
- `openbmb/AgentCPM-Explore`
- `openai/gpt-oss-120b`
- OpenAI-compatible API services such as MiniMax / GLM-style endpoints

## Quick Start

The quickest way to understand the repository is:

1. set up the environment,
2. build a retrieval index,
3. generate agent trajectories,
4. convert trajectories into retriever training data,
5. train a retriever, and
6. run benchmark evaluation.

### 1. Environment Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment
uv sync

# Optional: activate the environment
source .venv/bin/activate

# Install flash-attn if needed by your environment
uv pip install --no-build-isolation flash-attn
```

Install Java 21 for the Lucene / Pyserini-based BM25 pipeline:

```bash
conda install -c conda-forge openjdk=21
```

Install the local FlagEmbedding package:

```bash
cd FlagEmbedding
pip install -e .
cd ..
```

### 2. Build Retrieval Indexes

See [docs/index_construction.md](docs/index_construction.md) for the full indexing notes.

BM25 template:

```bash
python src/index_builder.py \
  --retrieval_method bm25 \
  --corpus_path /path/to/your/corpus.jsonl \
  --save_dir /path/to/save/index
```

Dense embedding template via Tevatron:

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

### 3. Generate Agent Trajectories

See [docs/trajectory_construction.md](docs/trajectory_construction.md) for full examples.

Example with `Tongyi` and a `bm25` backend:

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

Example with `Tongyi` and a `faiss` backend:

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

### 4. Build Training Data from Trajectories

See [docs/training_data_construction.md](docs/training_data_construction.md).

If you do not want to build training data from scratch, you can directly use the released [LRAT-Train](https://huggingface.co/datasets/Yuqi-Zhou/LRAT-Train) dataset. If you prefer to control filtering or supervision design yourself, you can also start from saved agent trajectories and rerun pair extraction with `src/data_builder.py`.

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

### 5. Train the Retriever

The repository currently uses the local `FlagEmbedding` training recipe. Start from:

- [FlagEmbedding/examples/finetune/embedder/run.sh](FlagEmbedding/examples/finetune/embedder/run.sh)
- [FlagEmbedding/examples/finetune/embedder/README.md](FlagEmbedding/examples/finetune/embedder/README.md)

You can plug the JSONL generated by `src/data_builder.py` into your existing training setup without changing the repository-level presentation structure.

### 6. Evaluate End-to-End Performance

See [docs/evaluate.md](docs/evaluate.md).

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

## Documentation

### Core Guides

| Topic | Link |
| --- | --- |
| Index Construction | [docs/index_construction.md](docs/index_construction.md) |
| Trajectory Construction | [docs/trajectory_construction.md](docs/trajectory_construction.md) |
| Training Data Construction | [docs/training_data_construction.md](docs/training_data_construction.md) |
| Minimal Reproduction | [docs/minimal_repro.md](docs/minimal_repro.md) |
| Evaluation | [docs/evaluate.md](docs/evaluate.md) |

### Advanced Notes

| Topic | Link |
| --- | --- |
| Experiment Layout | [docs/advanced/experiment_layout.md](docs/advanced/experiment_layout.md) |
| Segmented Training Data Experiments | [docs/advanced/segmented_training_data_experiment.md](docs/advanced/segmented_training_data_experiment.md) |

## Data and Outputs

- Example benchmark files are stored in `datasets/`.
- Query and qrel files are stored in `topics-qrels/`.
- Example trajectory outputs are stored in `trajectory/`.
- Generated run artifacts are saved as one JSON file per query by the agent clients.

## Acknowledgements

This repository builds on and benefits from several excellent open-source projects and public resources:

- [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
- [openclaw/openclaw](https://github.com/openclaw/openclaw)
- [FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)
- [Tevatron](https://github.com/texttron/tevatron)

## License

This repository is released under the Apache License 2.0. See [LICENSE](LICENSE).

Vendored components keep their own upstream licenses, especially:

- `FlagEmbedding/` under its upstream MIT license
- `tevatron/` under Apache License 2.0

## Citation

If you find this repository useful, please cite our **SIGIR 2026** paper below. The current PDF is also available directly in this repository at [assets/paper/LRAT.pdf](assets/paper/LRAT.pdf).

```bibtex
@inproceedings{zhou2026lrat,
  title={Learning to Retrieve from Agent Trajectories},
  author={Zhou, Yuqi and Dai, Sunhao and Qu, Changle and Pang, Liang and Xu, Jun and Wen, Ji-Rong},
  booktitle={Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2026}
}
```
