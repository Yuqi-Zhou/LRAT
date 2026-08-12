# Retrieval services

The CCIR-B agent uses the repository's existing BrowseComp-Plus-compatible MCP
server at [`../../searcher/mcp_server.py`](../../searcher/mcp_server.py). Start
one local service per retriever. Every service must expose both `search` and
`get_document`, and its port must match `configs/campaign.example.json`.

For a FAISS submission, the shape of one launch command is:

```bash
python searcher/mcp_server.py \
  --searcher-type faiss \
  --transport streamable-http \
  --port 38190 \
  --get-document \
  --k 10 \
  --index-path '/path/to/index-*.pkl' \
  --model-name /path/to/retriever \
  --dataset-name /path/to/corpus.jsonl \
  --pooling eos \
  --normalize
```

Repeat with the other five retriever paths and ports `38191` through `38195`.
Use each submission's declared pooling, normalization, query prefix, and
passage prefix; those are retriever properties and must not be guessed by the
inference runner.
