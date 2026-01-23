# Learning to Retrieve for Agentic Search (SIGIR 2026 Submission)

This repository contains our implementation for the SIGIR 2026 submission on **Learning to Retrieve**, **Agentic Search**, and **Agent Trajectory** analysis/training. It includes:
- Trajectory construction with different retrievers (BM25 / dense)
- Training data construction from trajectories
- Retriever fine-tuning (based on FlagEmbedding)
- Evaluation pipelines on standard benchmarks

---

## Repository Layout (What We Provide)

### Trajectory Data
- Example trajectories are provided under:
  - `trajectory/bm25/`

### Training Data
- A small example training set is provided at:
  - `FlagEmbedding-master/examples/finetune/embedder/example_data/sample_data.jsonl`

### Evaluation Data
- We place evaluation datasets under:
  - `datasets/`
  - including **Browsecomp-Plus** and **InfoSeek-Eval**

### Prompts
- Prompts for each pipeline stage are located in their corresponding folders.

---

## -1. Dataset Preparation

### Corpus Preparation

We follow prior work and use two corpora:

1) **Browsecomp-Plus Corpus** (Tevatron)
```python
from datasets import load_dataset
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")
```

2) **Wiki-25-512 Corpus** (InfoSeek / wikidump-25)
- Hugging Face: https://huggingface.co/datasets/Lk123/wiki-25-512
```python
from datasets import load_dataset
ds = load_dataset("Lk123/wiki-25-512")
```

---

## Model List

### Retriever (Embedding Model) HF IDs
- `intfloat/multilingual-e5-large-instruct`
- `Qwen3/Qwen3-Embedding-0.6b`
- `Qwen3/Qwen3-Embedding-4b`
- `Qwen3/Qwen3-Embedding-8b`

### Agent / LLM HF IDs
- `openbmb/AgentCPM-Explore`
- `hkust-nlp/WebExplorer-8B`
- `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`
- `openai/gpt-oss-120b`

---

## 0. Environment Setup

We use `uv` with Python 3.10 to manage the environment.

Install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:
```bash
uv sync
source .venv/bin/activate
uv pip install --no-build-isolation flash-attn
```

### Java 21 Dependency
This repo depends on Java 21. Install via conda:
```bash
conda install -c conda-forge openjdk=21
```

Or via apt (requires sudo):
```bash
sudo apt update
sudo apt install -y openjdk-21-jdk
```

### Install the Modified FlagEmbedding
You must install our modified FlagEmbedding:
```bash
cd FlagEmbedding
pip install -e .
```

---

## Index Construction

Please follow:
- `docs/index.md`

---

## 1. Trajectory Construction

To generate agent trajectories, see:
- `docs/trajectory_construction.md`

---

## 2. Training Data Construction (from Trajectories)

After trajectories are generated, build retriever training data by following:
- `docs/training_data_construction.md`

### Why We Use InfoSeekQA for Trajectories
A core requirement for trajectory analysis is that the underlying tasks genuinely demand multi-step information seeking. Many existing open-domain QA datasets do not meet this requirement in practice. Single-hop datasets such as NQ and TriviaQA, and even commonly used multi-hop datasets such as HotpotQA, 2WikiMultihopQA, and MuSiQue, often allow shallow retrieval strategies. Prior work shows that agents trained on these datasets make fewer than three retrieval calls per task on average, indicating limited multi-step interaction. Therefore, our trajectories are built from **InfoSeekQA** to better reflect multi-step browsing behavior.

---

## 3. Retriever Training

The training script is provided here:
- `FlagEmbedding-master/examples/finetune/embedder/run.sh`

Run it directly after preparing the training data.

---

## 4. Evaluation

To evaluate final trajectory results, see:
- `docs/evaluate.md`

---

## Acknowledgements

We thank the following repositories for their implementations and contributions:
- BrowseComp-Plus: https://github.com/texttron/BrowseComp-Plus  
- FlagEmbedding: https://github.com/FlagOpen/FlagEmbedding