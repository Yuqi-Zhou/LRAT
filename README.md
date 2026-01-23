# Learning to Retrieve from Agent Trajectories (SIGIR 2026 Submission)

---

## 📚 Navigation

| Section | Description |
|---------|-------------|
| [Repository Layout](#repository-layout) | Overview of trajectory data, training data, evaluation datasets, and prompts |
| [Dataset Preparation](#dataset-preparation) | Instructions for corpus download & preprocessing |
| [Model List](#model-list) | Available retrievers & agents |
| [Environment Setup](#0-environment-setup) | Python, Java, and dependencies installation |
| [Index Construction](#1-index-construction) | Creating indexes for retrieval |
| [Trajectory Construction](#2-trajectory-construction) | Generating agent trajectories |
| [Training Data Construction](#3-training-data-construction-from-trajectories) | Building retriever training datasets |
| [Retriever Training](#4-retriever-training) | Fine-tuning the retriever |
| [Evaluation](#5-evaluation) | Evaluating trajectories and retrieval performance |
| [Acknowledgements](#acknowledgements) | Credits for datasets, repositories, and tools |

---

## Repository Layout

### Trajectory Data
- Example trajectories are available under: `trajectory/bm25/`  

### Training Data
- Sample training set: `FlagEmbedding/examples/finetune/embedder/example_data/sample_data.jsonl`  

### Evaluation Data
- Evaluation datasets: `datasets/` (includes **BrowseComp-Plus** and **InfoSeek-Eval**)  

### Prompts
- Prompts for each pipeline stage are located in their respective folders.

---

## Dataset Preparation

### Corpus Preparation

1️⃣ **BrowseComp-Plus Corpus (Tevatron)**

```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
````

2️⃣ **Wiki-25-512 Corpus (InfoSeek / wikidump-25)**

```python
from datasets import load_dataset
ds = load_dataset("Lk123/wiki-25-512")
```

---

## Model List

### Retriever (Embedding Models)

* `intfloat/multilingual-e5-large-instruct`
* `Qwen3/Qwen3-Embedding-0.6b`
* `Qwen3/Qwen3-Embedding-4b`
* `Qwen3/Qwen3-Embedding-8b`

### Agent / LLM

* `openbmb/AgentCPM-Explore`
* `hkust-nlp/WebExplorer-8B`
* `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`
* `openai/gpt-oss-120b`

---

## 0. Environment Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment and activate
uv sync
source .venv/bin/activate

# Install dependencies
uv pip install --no-build-isolation flash-attn
```

### Java 21 Dependency

```bash
conda install -c conda-forge openjdk=21
# or via apt
sudo apt update
sudo apt install -y openjdk-21-jdk
```

### Install Modified FlagEmbedding

```bash
cd FlagEmbedding
pip install -e .
```

---

## 1. Index Construction

* Guide: `docs/index.md`

---

## 2. Trajectory Construction

* Guide: `docs/trajectory_construction.md`

---

## 3. Training Data Construction (from Trajectories)

* Guide: `docs/training_data_construction.md`

### Why InfoSeekQA?

* Multi-step information-seeking tasks are essential.
* Other benchmarks (NQ, TriviaQA, HotpotQA, etc.) allow shallow retrieval (<3 calls/query).
* **InfoSeekQA** enforces iterative search and browsing for richer trajectories.

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Lk123-purple?logo=huggingface)](https://huggingface.co/datasets/Lk123/InfoSeek)

---

## 4. Retriever Training

* Script: `FlagEmbedding/examples/finetune/embedder/run.sh`

> Run after preparing training data.

---

## 5. Evaluation

* Guide: `docs/evaluate.md`

---

## Acknowledgements

We gratefully acknowledge the following repositories and datasets:

| Repository                | Maintainer | Link                                                                                                                                     |
| ------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 🌐 **BrowseComp-Plus**    | Texttron   | [![GitHub](https://img.shields.io/badge/GitHub-Texttron-blue?logo=github)](https://github.com/texttron/BrowseComp-Plus)                  |
| ⚡ **FlagEmbedding**       | FlagOpen   | [![GitHub](https://img.shields.io/badge/GitHub-FlagOpen-orange?logo=github)](https://github.com/FlagOpen/FlagEmbedding)                  |
| 📚 **InfoSeekQA Dataset** | Lk123      | [![HuggingFace](https://img.shields.io/badge/HuggingFace-Lk123-purple?logo=huggingface)](https://huggingface.co/datasets/Lk123/InfoSeek) |

> These repositories and datasets provided the foundation for trajectory construction, retriever training, and evaluation pipelines in our SIGIR 2026 submission. We sincerely thank all maintainers for making their resources publicly available.
