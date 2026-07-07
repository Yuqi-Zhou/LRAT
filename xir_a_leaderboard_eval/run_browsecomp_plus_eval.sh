#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_browsecomp_plus_eval.sh [options] [--repo-file FILE] [HF_REPO ...]

Purpose:
  End-to-end BrowseComp-Plus evaluation for one or more retrieval model repos.
  Each HF_REPO is treated as an embedding/retrieval model and gets its own
  downloaded model directory, FAISS index, agent runs, judge results, and logs.

Defaults match the requested setting:
  - tools: search + get_document
  - max tool/LLM-call budget: 50
  - agent model: Qwen/Qwen3.5-4B served by vLLM
  - judge model: Qwen/Qwen3.5-4B via BrowseComp-Plus judge script
  - all persistent state under /mnt/bn/search-tiktok-nas-au/yuqizhou/competition

Options:
  --root DIR                  Persistent workspace root
  --source-tarball FILE       fallback BrowseComp-Plus source tarball
  --tevatron-tarball FILE     fallback Tevatron source tarball
  --hf-token TOKEN            Hugging Face token for private retriever repos
  --repo-file FILE            newline-delimited HF repo list; blank/# lines ignored
  --agent-model REPO          vLLM-served agent model
  --judge-model REPO          vLLM judge model
  --tensor-parallel-size N    vLLM tensor parallel size
  --gpu-devices LIST          default CUDA device list for all stages
  --index-gpu-devices LIST    CUDA devices used for parallel corpus indexing
  --agent-gpu-devices LIST    CUDA devices used by the vLLM agent server
  --search-gpu-devices LIST   CUDA devices used by the MCP FAISS search server
  --judge-gpu-devices LIST    CUDA devices used by the vLLM judge
  --index-shards N            corpus embedding shards; defaults to index GPU count
  --agent-workers N           parallel qwen-agent worker processes
  --agent-port N              preferred vLLM OpenAI-compatible server port
  --mcp-port N                preferred MCP FAISS search server port
  --llm-stack NAME            qwen35 (default) uses an isolated modern vLLM env; legacy uses the main env
  --llm-venv DIR              isolated LLM venv path for --llm-stack qwen35
  --llm-vllm-package SPEC     vLLM package spec for the isolated LLM env
  --llm-pypi-index-url URL    package index used for the isolated LLM env
  --vllm-max-model-len N      vLLM server context length; default 262144
  --vllm-max-num-seqs N       vLLM server max concurrent sequences
  --vllm-gpu-memory-util F    vLLM GPU memory utilization
  --max-steps N               Qwen-agent max LLM/tool loop budget
  --max-tokens N              agent max output tokens per answer
  --retrieval-batch-size N    Tevatron corpus encoding batch size
  --attn-implementation NAME  retriever attention backend: eager, sdpa, flash_attention_2
  --index-passage-max-len N   max passage tokens for corpus indexing; default 4096
  --dataset-download-workers N parallel curl workers for HF dataset snapshots
  --corpus-limit N            optional first-N corpus smoke index
  --eval-limit N              optional first-N query smoke run
  --keep-models               do not delete downloaded retriever model dirs
  --keep-indexes              do not delete generated FAISS indexes
  --skip-env                  do not install/update Python dependencies
  --recreate-env              delete and recreate the Python venv
  --prepare-only              setup env/data only, then exit before repo eval
  --skip-index                skip index build; requires existing corpus.shard*.pkl
  --skip-run                  skip Qwen end-to-end agent run
  --skip-judge                skip LLM judge
  --skip-llm-compat-check     skip agent/judge model config compatibility preflight
  --fail-fast                 stop at the first failed repo
  --force-index               rebuild index even if corpus.shard*.pkl exists
  -h, --help                  show this help

Proxy credentials:
  Put BCP_PROXY_HOST/PORT/USER/PWD in the environment or in $ROOT/proxy.env.
EOF
}

ROOT="/mnt/bn/search-tiktok-nas-au/yuqizhou/competition"
SOURCE_TARBALL="${BCP_SOURCE_TARBALL:-}"
TEVATRON_TARBALL="${BCP_TEVATRON_TARBALL:-}"
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
REPO_FILE="${BCP_REPO_FILE:-}"
AGENT_MODEL="Qwen/Qwen3.5-4B"
JUDGE_MODEL="Qwen/Qwen3.5-4B"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
INDEX_GPU_DEVICES="${BCP_INDEX_GPU_DEVICES:-$GPU_DEVICES}"
AGENT_GPU_DEVICES="${BCP_AGENT_GPU_DEVICES:-0}"
SEARCH_GPU_DEVICES="${BCP_SEARCH_GPU_DEVICES:-1}"
JUDGE_GPU_DEVICES="${BCP_JUDGE_GPU_DEVICES:-0}"
INDEX_SHARDS="${BCP_INDEX_SHARDS:-}"
AGENT_WORKERS="${BCP_AGENT_WORKERS:-4}"
AGENT_TENSOR_PARALLEL_SIZE="${BCP_AGENT_TENSOR_PARALLEL_SIZE:-$TENSOR_PARALLEL_SIZE}"
JUDGE_TENSOR_PARALLEL_SIZE="${BCP_JUDGE_TENSOR_PARALLEL_SIZE:-$TENSOR_PARALLEL_SIZE}"
VLLM_MAX_MODEL_LEN="${BCP_VLLM_MAX_MODEL_LEN:-262144}"
VLLM_MAX_NUM_SEQS="${BCP_VLLM_MAX_NUM_SEQS:-8}"
VLLM_GPU_MEMORY_UTIL="${BCP_VLLM_GPU_MEMORY_UTIL:-0.85}"
AGENT_PORT="${BCP_AGENT_PORT:-8000}"
MCP_PORT="${BCP_MCP_PORT:-8080}"
LLM_STACK="${BCP_LLM_STACK:-qwen35}"
LLM_VENV="${BCP_LLM_VENV:-}"
LLM_VLLM_PACKAGE="${BCP_LLM_VLLM_PACKAGE:-vllm==0.24.0}"
LLM_PYPI_INDEX_URL="${BCP_LLM_PYPI_INDEX_URL:-https://bytedpypi.byted.org/simple/}"
LLM_PYTHON=""
LLM_VLLM_BIN=""
MAX_STEPS=50
MAX_TOKENS=4096
RETRIEVAL_BATCH_SIZE=32
ATTN_IMPLEMENTATION="${BCP_ATTN_IMPLEMENTATION:-eager}"
INDEX_PASSAGE_MAX_LEN="${BCP_INDEX_PASSAGE_MAX_LEN:-4096}"
DATASET_DOWNLOAD_WORKERS="${BCP_DATASET_DOWNLOAD_WORKERS:-6}"
CORPUS_LIMIT="${BCP_CORPUS_LIMIT:-}"
EVAL_LIMIT=""
KEEP_MODELS=0
KEEP_INDEXES=0
SKIP_ENV=0
RECREATE_ENV=0
FORCE_INDEX=0
PREPARE_ONLY=0
SKIP_INDEX=0
SKIP_RUN=0
SKIP_JUDGE=0
SKIP_LLM_COMPAT_CHECK="${BCP_SKIP_LLM_COMPAT_CHECK:-0}"
FAIL_FAST=0
BATCH_STATUS=""
INDEX_GPU_DEVICES_EXPLICIT=0
AGENT_GPU_DEVICES_EXPLICIT=0
SEARCH_GPU_DEVICES_EXPLICIT=0
JUDGE_GPU_DEVICES_EXPLICIT=0
[[ -n "${BCP_INDEX_GPU_DEVICES:-}" ]] && INDEX_GPU_DEVICES_EXPLICIT=1
[[ -n "${BCP_AGENT_GPU_DEVICES:-}" ]] && AGENT_GPU_DEVICES_EXPLICIT=1
[[ -n "${BCP_SEARCH_GPU_DEVICES:-}" ]] && SEARCH_GPU_DEVICES_EXPLICIT=1
[[ -n "${BCP_JUDGE_GPU_DEVICES:-}" ]] && JUDGE_GPU_DEVICES_EXPLICIT=1

REPOS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --source-tarball) SOURCE_TARBALL="$2"; shift 2 ;;
    --tevatron-tarball) TEVATRON_TARBALL="$2"; shift 2 ;;
    --repo-file) REPO_FILE="$2"; shift 2 ;;
    --hf-token)
      HF_TOKEN="$2"
      export HF_TOKEN
      export HUGGINGFACE_HUB_TOKEN="$2"
      shift 2
      ;;
    --agent-model) AGENT_MODEL="$2"; shift 2 ;;
    --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
    --tensor-parallel-size)
      TENSOR_PARALLEL_SIZE="$2"
      AGENT_TENSOR_PARALLEL_SIZE="$2"
      JUDGE_TENSOR_PARALLEL_SIZE="$2"
      shift 2
      ;;
    --gpu-devices)
      GPU_DEVICES="$2"
      if [[ "$INDEX_GPU_DEVICES_EXPLICIT" -eq 0 ]]; then INDEX_GPU_DEVICES="$2"; fi
      shift 2
      ;;
    --index-gpu-devices) INDEX_GPU_DEVICES="$2"; INDEX_GPU_DEVICES_EXPLICIT=1; shift 2 ;;
    --agent-gpu-devices) AGENT_GPU_DEVICES="$2"; AGENT_GPU_DEVICES_EXPLICIT=1; shift 2 ;;
    --search-gpu-devices) SEARCH_GPU_DEVICES="$2"; SEARCH_GPU_DEVICES_EXPLICIT=1; shift 2 ;;
    --judge-gpu-devices) JUDGE_GPU_DEVICES="$2"; JUDGE_GPU_DEVICES_EXPLICIT=1; shift 2 ;;
    --index-shards) INDEX_SHARDS="$2"; shift 2 ;;
    --agent-workers) AGENT_WORKERS="$2"; shift 2 ;;
    --agent-port) AGENT_PORT="$2"; shift 2 ;;
    --mcp-port) MCP_PORT="$2"; shift 2 ;;
    --llm-stack) LLM_STACK="$2"; shift 2 ;;
    --llm-venv) LLM_VENV="$2"; shift 2 ;;
    --llm-vllm-package) LLM_VLLM_PACKAGE="$2"; shift 2 ;;
    --llm-pypi-index-url) LLM_PYPI_INDEX_URL="$2"; shift 2 ;;
    --vllm-max-model-len) VLLM_MAX_MODEL_LEN="$2"; shift 2 ;;
    --vllm-max-num-seqs) VLLM_MAX_NUM_SEQS="$2"; shift 2 ;;
    --vllm-gpu-memory-util) VLLM_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --max-steps) MAX_STEPS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --retrieval-batch-size) RETRIEVAL_BATCH_SIZE="$2"; shift 2 ;;
    --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
    --index-passage-max-len) INDEX_PASSAGE_MAX_LEN="$2"; shift 2 ;;
    --dataset-download-workers) DATASET_DOWNLOAD_WORKERS="$2"; shift 2 ;;
    --corpus-limit) CORPUS_LIMIT="$2"; shift 2 ;;
    --eval-limit) EVAL_LIMIT="$2"; shift 2 ;;
    --keep-models) KEEP_MODELS=1; shift ;;
    --keep-indexes) KEEP_INDEXES=1; shift ;;
    --skip-env) SKIP_ENV=1; shift ;;
    --recreate-env) RECREATE_ENV=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --skip-index) SKIP_INDEX=1; shift ;;
    --skip-run) SKIP_RUN=1; shift ;;
    --skip-judge) SKIP_JUDGE=1; shift ;;
    --skip-llm-compat-check) SKIP_LLM_COMPAT_CHECK=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --force-index) FORCE_INDEX=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) REPOS+=("$1"); shift ;;
  esac
done

if [[ -n "$REPO_FILE" ]]; then
  [[ -f "$REPO_FILE" ]] || { echo "--repo-file does not exist: $REPO_FILE" >&2; exit 2; }
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "$line" ]] || continue
    REPOS+=("$line")
  done < "$REPO_FILE"
fi

if [[ ${#REPOS[@]} -eq 0 && "$PREPARE_ONLY" -eq 0 ]]; then
  echo "At least one HF repo name is required." >&2
  usage >&2
  exit 2
fi

timestamp() { date +"%Y%m%d_%H%M%S"; }

safe_name() {
  python3 - "$1" <<'PY'
import re, sys
s = sys.argv[1].strip()
print(re.sub(r"[^A-Za-z0-9._-]+", "__", s).strip("_") or "repo")
PY
}

init_batch_status() {
  BATCH_STATUS="$ROOT/logs/$(timestamp)__batch_status.tsv"
  printf 'timestamp\trepo\tsafe_name\tstatus\trc\tlog_dir\tsummary_path\n' > "$BATCH_STATUS"
  log "batch status: $BATCH_STATUS"
}

record_repo_status() {
  local repo="$1"
  local safe="$2"
  local status="$3"
  local rc="$4"
  local log_dir="$5"
  local summary="${6:-}"
  [[ -n "$BATCH_STATUS" ]] || return 0
  python3 - "$BATCH_STATUS" "$repo" "$safe" "$status" "$rc" "$log_dir" "$summary" <<'PY'
from datetime import datetime
from pathlib import Path
import sys

path, repo, safe, status, rc, log_dir, summary = sys.argv[1:8]
def clean(value: str) -> str:
    return value.replace("\t", " ").replace("\n", " ")
row = [
    datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    repo,
    safe,
    status,
    rc,
    log_dir,
    summary,
]
with Path(path).open("a", encoding="utf-8") as f:
    f.write("\t".join(clean(x) for x in row) + "\n")
PY
}

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

redact_stream() {
  python3 -c 'import sys
secret = sys.argv[1] if len(sys.argv) > 1 else ""
for line in sys.stdin:
    if secret:
        line = line.replace(secret, "***")
    sys.stdout.write(line)
    sys.stdout.flush()
' "${BCP_PROXY_PWD:-}"
}

start_bg() {
  local name="$1"; shift
  local logfile="$1"; shift
  log "starting $name; log=$logfile" >&2
  "$@" >"$logfile" 2>&1 &
  echo $!
}

stop_pid() {
  local pid="${1:-}"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 3
  fi
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
}

wait_http() {
  local url="$1"
  local name="$2"
  local timeout_s="${3:-900}"
  local start
  start=$(date +%s)
  while true; do
    if curl -k -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      log "$name is ready at $url"
      return 0
    fi
    if (( $(date +%s) - start > timeout_s )); then
      echo "$name did not become ready within ${timeout_s}s: $url" >&2
      return 1
    fi
    sleep 5
  done
}

find_free_port() {
  local preferred="$1"
  local name="$2"
  local port
  port=$(python3 - "$preferred" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for port in range(start, min(start + 200, 65535)):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit(f"no free localhost port found from {start}")
PY
)
  if [[ "$port" != "$preferred" ]]; then
    log "$name preferred port $preferred is busy; using $port" >&2
  fi
  echo "$port"
}

split_csv_list() {
  local value="$1"
  value="${value// /}"
  IFS=',' read -r -a _BCP_SPLIT_RESULT <<< "$value"
}

count_csv_list() {
  local value="$1"
  value="${value// /}"
  if [[ -z "$value" ]]; then
    echo 0
  else
    awk -F',' '{print NF}' <<< "$value"
  fi
}

finalize_gpu_defaults() {
  split_csv_list "$GPU_DEVICES"
  local -a gpus=("${_BCP_SPLIT_RESULT[@]}")
  if [[ "$AGENT_GPU_DEVICES_EXPLICIT" -eq 0 && ${#gpus[@]} -ge 1 ]]; then
    AGENT_GPU_DEVICES="${gpus[0]}"
  fi
  if [[ "$SEARCH_GPU_DEVICES_EXPLICIT" -eq 0 ]]; then
    if [[ ${#gpus[@]} -ge 2 ]]; then
      SEARCH_GPU_DEVICES="${gpus[1]}"
    elif [[ ${#gpus[@]} -ge 1 ]]; then
      SEARCH_GPU_DEVICES="${gpus[0]}"
    fi
  fi
  if [[ "$JUDGE_GPU_DEVICES_EXPLICIT" -eq 0 && ${#gpus[@]} -ge 1 ]]; then
    JUDGE_GPU_DEVICES="${gpus[0]}"
  fi
}

configure_proxy() {
  if [[ -f "$ROOT/proxy.env" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT/proxy.env"
  fi

  export BCP_PROXY_HOST="${BCP_PROXY_HOST:-}"
  export BCP_PROXY_PORT="${BCP_PROXY_PORT:-8080}"
  export BCP_PROXY_USER="${BCP_PROXY_USER:-}"
  export BCP_PROXY_PWD="${BCP_PROXY_PWD:-}"

  if [[ -z "$BCP_PROXY_HOST" || -z "$BCP_PROXY_USER" || -z "$BCP_PROXY_PWD" ]]; then
    log "proxy credentials are incomplete; set BCP_PROXY_* or create $ROOT/proxy.env"
    return 0
  fi

  local proxy="http://${BCP_PROXY_USER}:${BCP_PROXY_PWD}@${BCP_PROXY_HOST}:${BCP_PROXY_PORT}"
  export http_proxy="$proxy"
  export https_proxy="$proxy"
  export HTTP_PROXY="$proxy"
  export HTTPS_PROXY="$proxy"
  export no_proxy="localhost,127.0.0.1,::1,.byted.org"
  export NO_PROXY="$no_proxy"

  export GIT_SSL_NO_VERIFY=true
  export CURL_CA_BUNDLE=""
  export REQUESTS_CA_BUNDLE=""
  export HF_HUB_DISABLE_SSL_VERIFICATION=1
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
  export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
  export BCP_INSECURE_HF_SSL="${BCP_INSECURE_HF_SSL:-1}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  log "proxy configured via BCP_PROXY_*"
}

prepare_dirs() {
  mkdir -p \
    "$ROOT"/{src,data,models,indexes,runs,evals,logs,tmp,hf_home,hf_datasets_cache,artifacts}
  export ROOT
  SOURCE_TARBALL="${SOURCE_TARBALL:-$ROOT/artifacts/BrowseComp-Plus.source.tar.gz}"
  TEVATRON_TARBALL="${TEVATRON_TARBALL:-$ROOT/artifacts/tevatron.source.tar.gz}"
  export HF_HOME="$ROOT/hf_home"
  export HF_DATASETS_CACHE="$ROOT/hf_datasets_cache"
  export TRANSFORMERS_CACHE="$ROOT/hf_home/transformers"
  export HF_HUB_CACHE="$ROOT/hf_home/hub"
  export TMPDIR="$ROOT/tmp"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OTEL_SDK_DISABLED="${OTEL_SDK_DISABLED:-true}"
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  if [[ -n "$HF_TOKEN" ]]; then
    export HF_TOKEN
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  fi
  write_hf_ssl_bootstrap
}

write_hf_ssl_bootstrap() {
  mkdir -p "$ROOT/tmp"
  cat > "$ROOT/tmp/bcp_hf_ssl_bootstrap.py" <<'PY'
import os

if os.getenv("BCP_INSECURE_HF_SSL", "0") == "1":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass
    try:
        import urllib3
        urllib3.disable_warnings()
    except Exception:
        pass
    try:
        import requests
        _bcp_original_request = requests.sessions.Session.request
        def _bcp_insecure_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _bcp_original_request(self, method, url, **kwargs)
        requests.sessions.Session.request = _bcp_insecure_request
    except Exception:
        pass
    try:
        import requests
        from huggingface_hub import configure_http_backend
        def _bcp_backend_factory():
            session = requests.Session()
            session.verify = False
            return session
        configure_http_backend(backend_factory=_bcp_backend_factory)
    except Exception:
        pass
PY
}

snapshot_hf_repo() {
  local repo="$1"
  local repo_type="$2"
  local dest="$3"
  local complete="$4"
  if [[ -s "$complete" ]]; then
    log "HF $repo_type snapshot already present: $repo -> $dest"
    return 0
  fi
  mkdir -p "$dest"
  log "snapshotting HF $repo_type $repo to $dest"
  local manifest="$ROOT/tmp/$(safe_name "$repo").${repo_type}.files.tsv"
  python - "$repo" "$repo_type" "$dest" "$manifest" <<'PY'
import os, runpy, sys
from pathlib import Path

runpy.run_path(os.path.join(os.environ["ROOT"], "tmp", "bcp_hf_ssl_bootstrap.py"))

from huggingface_hub import HfApi, hf_hub_url

repo, repo_type, dest, manifest = sys.argv[1:5]
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
api = HfApi(token=token)
files = api.list_repo_files(repo_id=repo, repo_type=repo_type)
Path(manifest).parent.mkdir(parents=True, exist_ok=True)
with open(manifest, "w", encoding="utf-8") as out:
    for filename in files:
        if filename.endswith("/"):
            continue
        if repo_type == "model":
            url = hf_hub_url(repo_id=repo, filename=filename)
        else:
            url = hf_hub_url(repo_id=repo, filename=filename, repo_type=repo_type)
        local = Path(dest, filename)
        out.write(f"{url}\t{local}\n")
PY

  local auth_header_file=""
  local hf_auth_token="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
  if [[ -n "$hf_auth_token" ]]; then
    auth_header_file="$ROOT/tmp/$(safe_name "$repo").${repo_type}.curl_headers"
    python3 - "$auth_header_file" "$hf_auth_token" <<'PY'
from pathlib import Path
import os
import sys

path = Path(sys.argv[1])
token = sys.argv[2]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(f"Authorization: Bearer {token}\n", encoding="utf-8")
os.chmod(path, 0o600)
PY
  fi

  local active=0
  local failed=0
  local url out tmp
  while IFS=$'\t' read -r url out; do
    [[ -n "$url" && -n "$out" ]] || continue
    if [[ -s "$out" ]]; then
      log "dataset file already present: $out"
      continue
    fi
    mkdir -p "$(dirname "$out")"
    tmp="${out}.incomplete"
    (
      set -Eeuo pipefail
      local -a auth_args=()
      if [[ -n "$auth_header_file" ]]; then
        auth_args=(-H "@$auth_header_file")
      fi
      log "downloading HF $repo_type file: $repo/${out#"$dest"/}"
      curl --no-progress-meter -k -L --fail --retry 10 --retry-delay 5 --retry-all-errors \
        --connect-timeout 30 --speed-limit 1024 --speed-time 300 \
        "${auth_args[@]}" \
        -C - -o "$tmp" "$url"
      if [[ -s "$tmp" ]]; then
        mv "$tmp" "$out"
      elif [[ -s "$out" ]]; then
        log "HF $repo_type file completed despite missing temp file: $out"
      else
        echo "curl finished but neither temp nor output file exists: $tmp" >&2
        exit 1
      fi
    ) &
    active=$((active + 1))
    if (( active >= DATASET_DOWNLOAD_WORKERS )); then
      if ! wait -n; then
        failed=1
      fi
      active=$((active - 1))
    fi
  done < "$manifest"
  while (( active > 0 )); do
    if ! wait -n; then
      failed=1
    fi
    active=$((active - 1))
  done
  if (( failed )); then
    [[ -z "$auth_header_file" ]] || rm -f "$auth_header_file"
    echo "Failed to download one or more HF $repo_type files for $repo" >&2
    return 1
  fi
  [[ -z "$auth_header_file" ]] || rm -f "$auth_header_file"
  printf 'ok\n' > "$complete"
}

snapshot_hf_dataset() {
  local repo="$1"
  local dest="$2"
  snapshot_hf_repo "$repo" "dataset" "$dest" "$dest/.bcp_snapshot_complete"
}

snapshot_hf_model() {
  local repo="$1"
  local dest="$2"
  snapshot_hf_repo "$repo" "model" "$dest" "$dest/.bcp_snapshot_complete"
}

prepare_model_path() {
  local model="$1"
  local namespace="$2"
  BCP_PREPARED_MODEL_PATH="$model"
  if [[ "$model" == */* && ! -e "$model" ]]; then
    local dest="$ROOT/models/$namespace/$(safe_name "$model")"
    snapshot_hf_model "$model" "$dest"
    BCP_PREPARED_MODEL_PATH="$dest"
  fi
  export BCP_PREPARED_MODEL_PATH
}

preflight_checks() {
  log "preflight: root=$ROOT"
  python3 - <<'PY'
import os, sys
major, minor = sys.version_info[:2]
print(f"python={sys.executable} version={major}.{minor}")
if (major, minor) < (3, 10):
    raise SystemExit("Python >=3.10 is required")
PY
  command -v curl >/dev/null || { echo "curl is required" >&2; exit 2; }
  command -v git >/dev/null || log "git is not available; source tarball fallback will be required"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
  else
    log "nvidia-smi not found"
  fi
  if [[ -n "${http_proxy:-}" || -n "${https_proxy:-}" ]]; then
    curl -k -fsSI -L --max-time 30 https://huggingface.co/api/models?limit=1 >/dev/null \
      && log "preflight: Hugging Face reachable" \
      || log "preflight warning: Hugging Face probe failed; downloads may fail"
  else
    log "preflight warning: no proxy env configured"
  fi
}

clone_or_update_repo() {
  local src="$ROOT/src/BrowseComp-Plus"
  if [[ -d "$src/.git" ]]; then
    log "updating BrowseComp-Plus at $src"
    if git -C "$src" -c http.sslVerify=false pull --ff-only; then
      return 0
    fi
    log "git pull failed; keeping existing source tree"
    return 0
  elif [[ -s "$src/pyproject.toml" ]]; then
    log "using existing BrowseComp-Plus source tree at $src"
  else
    log "cloning BrowseComp-Plus to $src"
    if command -v git >/dev/null && git -c http.sslVerify=false clone https://github.com/texttron/BrowseComp-Plus "$src"; then
      return 0
    fi
    if [[ ! -s "$SOURCE_TARBALL" ]] || ! tar -tzf "$SOURCE_TARBALL" >/dev/null 2>&1; then
      log "git clone failed; downloading BrowseComp-Plus codeload archive to $SOURCE_TARBALL"
      curl -k -L --retry 3 --connect-timeout 20 --max-time 180 \
        -o "$SOURCE_TARBALL" \
        https://codeload.github.com/texttron/BrowseComp-Plus/tar.gz/refs/heads/main
    fi
    if [[ -s "$SOURCE_TARBALL" ]] && tar -tzf "$SOURCE_TARBALL" >/dev/null 2>&1; then
      log "git clone failed; extracting BrowseComp-Plus from $SOURCE_TARBALL"
      local tmp_extract="$ROOT/tmp/extract_browsecomp_plus"
      rm -rf "$src"
      rm -rf "$tmp_extract"
      mkdir -p "$tmp_extract"
      tar -xzf "$SOURCE_TARBALL" -C "$tmp_extract"
      local extracted
      extracted=$(find "$tmp_extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)
      mv "$extracted" "$src"
      [[ -s "$src/pyproject.toml" ]] || { echo "source tarball did not create $src" >&2; exit 1; }
    else
      echo "Cannot obtain BrowseComp-Plus: git failed and no valid tarball exists at $SOURCE_TARBALL" >&2
      exit 1
    fi
  fi
}

install_tevatron() {
  local tevatron_src="$ROOT/src/tevatron"
  if [[ ! -s "$TEVATRON_TARBALL" ]] || ! tar -tzf "$TEVATRON_TARBALL" >/dev/null 2>&1; then
    log "downloading Tevatron codeload archive to $TEVATRON_TARBALL"
    curl -k -L --retry 3 --connect-timeout 20 --max-time 180 \
      -o "$TEVATRON_TARBALL" \
      https://codeload.github.com/texttron/tevatron/tar.gz/refs/heads/main
  fi
  if [[ -s "$TEVATRON_TARBALL" ]] && tar -tzf "$TEVATRON_TARBALL" >/dev/null 2>&1; then
    log "installing Tevatron from local tarball: $TEVATRON_TARBALL"
    local tmp_extract="$ROOT/tmp/extract_tevatron"
    rm -rf "$tevatron_src"
    rm -rf "$tmp_extract"
    mkdir -p "$tmp_extract"
    tar -xzf "$TEVATRON_TARBALL" -C "$tmp_extract"
    local extracted
    extracted=$(find "$tmp_extract" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    mv "$extracted" "$tevatron_src"
    python -m pip install -e "$tevatron_src" \
      -c "$ROOT/tmp/bcp_constraints.txt" \
      --trusted-host pypi.org --trusted-host files.pythonhosted.org
    return 0
  fi
  log "installing Tevatron from GitHub"
  python -m pip install "git+https://github.com/texttron/tevatron.git@main" \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
}

write_sitecustomize() {
  local py="${1:-python}"
  "$py" - <<'PY'
from pathlib import Path
import site

site_dirs = site.getsitepackages()
target = Path(site_dirs[0]) / "sitecustomize.py"
patch = Path(site_dirs[0]) / "bcp_ssl_patch.py"
pth = Path(site_dirs[0]) / "bcp_ssl_patch.pth"
code = r'''
import os

if os.getenv("BCP_INSECURE_HF_SSL", "0") == "1":
    try:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    except Exception:
        pass

    try:
        import urllib3
        urllib3.disable_warnings()
    except Exception:
        pass

    try:
        import requests
        from huggingface_hub import configure_http_backend

        _bcp_original_request = requests.sessions.Session.request

        def _bcp_insecure_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _bcp_original_request(self, method, url, **kwargs)

        requests.sessions.Session.request = _bcp_insecure_request

        def _bcp_backend_factory():
            session = requests.Session()
            session.verify = False
            return session

        configure_http_backend(backend_factory=_bcp_backend_factory)
    except Exception:
        pass
'''
target.write_text(code)
patch.write_text(code)
pth.write_text("import bcp_ssl_patch\n")
print(f"wrote {target}")
print(f"wrote {patch}")
print(f"wrote {pth}")
PY
}

llm_needed() {
  [[ "$SKIP_RUN" -eq 0 || "$SKIP_JUDGE" -eq 0 ]]
}

resolve_llm_paths() {
  case "$LLM_STACK" in
    legacy)
      LLM_VENV="$ROOT/envs/browsecomp-plus"
      ;;
    qwen35)
      LLM_VENV="${LLM_VENV:-$ROOT/envs/browsecomp-plus-llm-qwen35}"
      ;;
    *)
      echo "Unknown --llm-stack: $LLM_STACK (expected qwen35 or legacy)" >&2
      return 2
      ;;
  esac
  LLM_PYTHON="$LLM_VENV/bin/python"
  LLM_VLLM_BIN="$LLM_VENV/bin/vllm"
  export LLM_STACK LLM_VENV LLM_PYTHON LLM_VLLM_BIN
}

setup_llm_env() {
  llm_needed || return 0
  resolve_llm_paths
  if [[ "$LLM_STACK" == "legacy" ]]; then
    log "LLM stack: legacy main env ($LLM_VENV)"
    [[ -x "$LLM_PYTHON" ]] || { echo "main env is missing: $LLM_PYTHON" >&2; return 1; }
    return 0
  fi

  mkdir -p "$(dirname "$LLM_VENV")"
  if [[ "$RECREATE_ENV" -eq 1 && -d "$LLM_VENV" ]]; then
    log "removing existing LLM venv: $LLM_VENV"
    rm -rf "$LLM_VENV"
  fi
  if [[ ! -x "$LLM_PYTHON" ]]; then
    log "creating isolated Qwen3.5 LLM venv: $LLM_VENV"
    python3 -m venv "$LLM_VENV"
  fi

  log "installing isolated LLM stack: $LLM_VLLM_PACKAGE"
  if command -v uv >/dev/null 2>&1; then
    (
      set -Eeuo pipefail
      # shellcheck disable=SC1091
      source "$LLM_VENV/bin/activate"
      UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-120}" UV_LINK_MODE="${UV_LINK_MODE:-copy}" \
        uv pip install -U "$LLM_VLLM_PACKAGE" "tqdm>=4.67.1" \
          --index-url "$LLM_PYPI_INDEX_URL"
    )
  else
    "$LLM_PYTHON" -m pip install --upgrade pip setuptools wheel \
      --index-url "$LLM_PYPI_INDEX_URL"
    "$LLM_PYTHON" -m pip install --upgrade "$LLM_VLLM_PACKAGE" "tqdm>=4.67.1" \
      --index-url "$LLM_PYPI_INDEX_URL"
  fi
  write_sitecustomize "$LLM_PYTHON"
  "$LLM_PYTHON" - <<'PY'
mods = ["torch", "transformers", "vllm", "tqdm"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"import {name}: OK {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        raise SystemExit(f"import {name}: FAILED {type(exc).__name__}: {exc}")
PY
}

activate_llm_env() {
  llm_needed || return 0
  resolve_llm_paths
  if [[ ! -x "$LLM_PYTHON" ]]; then
    echo "LLM env is missing: $LLM_PYTHON. Run without --skip-env or set --llm-venv." >&2
    return 1
  fi
  if [[ "$LLM_STACK" != "legacy" && ! -x "$LLM_VLLM_BIN" ]]; then
    echo "vLLM binary is missing in LLM env: $LLM_VLLM_BIN. Run without --skip-env." >&2
    return 1
  fi
  write_sitecustomize "$LLM_PYTHON"
  log "LLM stack active: $LLM_STACK ($LLM_VENV)"
}

patch_transformers_disable_kernels() {
  python - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("transformers")
if spec is None or spec.origin is None:
    raise SystemExit(0)
root = Path(spec.origin).parent
p = root / "utils" / "import_utils.py"
if not p.exists():
    raise SystemExit(0)
s = p.read_text()
old = '_kernels_available = _is_package_available("kernels")'
new = '_kernels_available = False  # BCP: disable incompatible system kernels package'
if old in s:
    p.write_text(s.replace(old, new, 1))
elif new not in s:
    raise SystemExit("Cannot locate transformers _kernels_available assignment")
else:
    p.write_text(s)

s = p.read_text()
marker = "# BCP: disable incompatible system flash_attn package"
if marker not in s:
    p.write_text(
        s
        + f'''

{marker}
def is_flash_attn_2_available():
    return False


def is_flash_attn_3_available():
    return False


def is_flash_attn_greater_or_equal_2_10():
    return False


def is_flash_attn_greater_or_equal(library_version: str):
    return False
'''
    )
print(f"patched {p}")
PY
}

patch_hf_hub_dataclasses() {
  python - <<'PY'
from pathlib import Path
import importlib.util

spec = importlib.util.find_spec("huggingface_hub")
if spec is None or spec.origin is None:
    raise SystemExit(0)
p = Path(spec.origin).parent / "dataclasses.py"
if not p.exists():
    raise SystemExit(0)
s = p.read_text()
old = '            cls.validate(self)  # type: ignore [attr-defined]\n'
new = '''            try:
                cls.validate(self)  # type: ignore [attr-defined]
            except StrictDataclassFieldValidationError as exc:
                if "Unsupported type for field" not in str(exc):
                    raise
'''
if new in s:
    raise SystemExit(0)
if old not in s:
    raise SystemExit("Cannot locate huggingface_hub dataclass validate hook")
p.write_text(s.replace(old, new, 1))
print(f"patched {p}")
PY
}

setup_env() {
  local src="$ROOT/src/BrowseComp-Plus"
  local venv="$ROOT/envs/browsecomp-plus"
  mkdir -p "$(dirname "$venv")"
  if [[ "$RECREATE_ENV" -eq 1 && -d "$venv" ]]; then
    log "removing existing venv: $venv"
    rm -rf "$venv"
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    log "creating venv with system site packages: $venv"
    python3 -m venv --system-site-packages "$venv"
  fi
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  export PYTHONPATH="$ROOT/src/BrowseComp-Plus:${PYTHONPATH:-}"
  write_sitecustomize
  cat > "$ROOT/tmp/bcp_constraints.txt" <<'EOF'
huggingface_hub>=0.33.0,<0.34
transformers==4.53.2
click<8.3
vllm==0.9.2
openai>=1.52.0,<=1.90.0
pydantic>=2.10,<2.12
EOF
  python -m pip install --upgrade pip setuptools wheel \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
  python -m pip install --upgrade \
    "huggingface_hub>=0.33.0,<0.34" \
    "transformers==4.53.2" \
    "click<8.3" \
    "pydantic>=2.10,<2.12" \
    -c "$ROOT/tmp/bcp_constraints.txt" \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
  install_tevatron
  log "installing BrowseComp-Plus runtime dependencies"
  python -m pip install \
    "accelerate>=1.9.0" \
    "datasets>=4.0.0" \
    "faiss-cpu>=1.11.0.post1" \
    "fastmcp==2.9.2" \
    "openai>=1.52.0,<=1.90.0" \
    "pydantic>=2.10,<2.12" \
    "peft>=0.16.0" \
    "pyngrok>=7.2.12" \
    "python-dotenv>=1.1.1" \
    "qwen-agent[mcp,rag]==0.0.27" \
    "rich>=14.0.0" \
    "torchvision" \
    "tqdm>=4.67.1" \
    "vllm==0.9.2" \
    -c "$ROOT/tmp/bcp_constraints.txt" \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org
  patch_transformers_disable_kernels
  patch_hf_hub_dataclasses
  python - <<'PY'
mods = ["torch", "vllm", "datasets", "huggingface_hub", "transformers", "faiss", "qwen_agent", "tevatron"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"import {name}: OK {getattr(mod, '__version__', '?')}")
    except Exception as exc:
        raise SystemExit(f"import {name}: FAILED {type(exc).__name__}: {exc}")
PY
}

patch_dense_searcher_lazy_imports() {
  local src="$ROOT/src/BrowseComp-Plus"
  python3 - "$src/searcher/searchers/__init__.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
if "BCP lazy searcher import patch" in s:
    raise SystemExit(0)

p.write_text('''"""
Searchers package for different search implementations.

BCP lazy searcher import patch:
Import concrete searcher implementations only when selected.  This keeps the
dense FAISS path independent from optional BM25/pyserini dependencies.
"""

from enum import Enum
from importlib import import_module

from .base import BaseSearcher


class SearcherType(Enum):
    """Enum for managing available searcher types and their CLI mappings."""

    BM25 = ("bm25", "bm25_searcher", "BM25Searcher")
    FAISS = ("faiss", "faiss_searcher", "FaissSearcher")
    REASONIR = ("reasonir", "faiss_searcher", "ReasonIrSearcher")
    CUSTOM = ("custom", "custom_searcher", "CustomSearcher")

    def __init__(self, cli_name, module_name, class_name):
        self.cli_name = cli_name
        self.module_name = module_name
        self.class_name = class_name

    @classmethod
    def get_choices(cls):
        """Get list of CLI choices for argument parser."""
        return [searcher_type.cli_name for searcher_type in cls]

    @classmethod
    def get_searcher_class(cls, cli_name):
        """Get searcher class by CLI name."""
        for searcher_type in cls:
            if searcher_type.cli_name == cli_name:
                module = import_module(f".{searcher_type.module_name}", __name__)
                return getattr(module, searcher_type.class_name)
        raise ValueError(f"Unknown searcher type: {cli_name}")


__all__ = ["BaseSearcher", "SearcherType"]
''')
PY
}

patch_faiss_searcher_dataset_path() {
  local src="$ROOT/src/BrowseComp-Plus"
  python3 - "$src/searcher/searchers/faiss_searcher.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
changed = False

if "BCP dataset-path patch" not in s:
    old = '''        parser.add_argument(
            "--dataset-name",
            default="Tevatron/browsecomp-plus-corpus",
            help="Dataset name for document retrieval in FAISS search (default: Tevatron/browsecomp-plus-corpus)",
        )
'''
    new = old + '''        parser.add_argument(
            "--dataset-path",
            default=None,
            help="Optional local JSONL corpus path for document retrieval in FAISS search. BCP dataset-path patch.",
        )
'''
    if old not in s:
        raise SystemExit("Cannot locate --dataset-name parser block in faiss_searcher.py")
    s = s.replace(old, new, 1)
    changed = True

old = '''            ds = load_dataset(
                self.args.dataset_name, split="train", cache_dir=cache_dir
            )
            self.docid_to_text = {row["docid"]: row["text"] for row in ds}
'''
new = '''            if getattr(self.args, "dataset_path", None):
                ds = load_dataset(
                    "json",
                    data_files=self.args.dataset_path,
                    split="train",
                    cache_dir=cache_dir,
                )
            else:
                ds = load_dataset(
                    self.args.dataset_name, split="train", cache_dir=cache_dir
                )
            self.docid_to_text = {row["docid"]: row["text"] for row in ds}
'''
if old in s:
    s = s.replace(old, new, 1)
    changed = True
elif "getattr(self.args, \"dataset_path\", None)" not in s:
    raise SystemExit("Cannot locate load_dataset block in faiss_searcher.py")

old = '''        model_args = ModelArguments(
            model_name_or_path=self.args.model_name,
            normalize=self.args.normalize,
            pooling=self.args.pooling,
            cache_dir=cache_dir,
        )
'''
new = '''        model_args = ModelArguments(
            model_name_or_path=self.args.model_name,
            normalize=self.args.normalize,
            pooling=self.args.pooling,
            cache_dir=cache_dir,
            attn_implementation=os.getenv("BCP_ATTN_IMPLEMENTATION", "eager"),
        )
'''
if old in s:
    s = s.replace(old, new, 1)
    changed = True
elif 'attn_implementation=os.getenv("BCP_ATTN_IMPLEMENTATION", "eager")' not in s:
    raise SystemExit("Cannot locate ModelArguments block in faiss_searcher.py")

if changed:
    p.write_text(s)
PY
}

patch_searcher_tools_tokenizer() {
  local src="$ROOT/src/BrowseComp-Plus"
  python3 - "$src/searcher/tools.py" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
s = p.read_text()
old = '        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")\n'
new = '''        tokenizer = AutoTokenizer.from_pretrained(
            getattr(getattr(searcher, "args", None), "model_name", "Qwen/Qwen3-0.6B")
        )
'''
if old in s:
    p.write_text(s.replace(old, new, 1))
elif 'getattr(getattr(searcher, "args", None), "model_name"' not in s:
    raise SystemExit("Cannot locate snippet tokenizer load in searcher/tools.py")
PY
}

activate_env() {
  # shellcheck disable=SC1091
  source "$ROOT/envs/browsecomp-plus/bin/activate"
  export PYTHONPATH="$ROOT/src/BrowseComp-Plus:${PYTHONPATH:-}"
  write_sitecustomize
  patch_transformers_disable_kernels
  patch_hf_hub_dataclasses
}

patch_qwen_step_limit() {
  local src="$ROOT/src/BrowseComp-Plus"
  python3 - "$src/search_agent/qwen_client.py" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
changed = False

if "BCP_MAX_TOOL_STEPS" not in s:
    needle = "load_dotenv()\n"
    insert = r'''

# BrowseComp-Plus local automation patch:
# qwen-agent uses this global setting to limit agent/tool loop iterations.
try:
    import qwen_agent.settings as _bcp_qwen_settings
    _bcp_qwen_settings.MAX_LLM_CALL_PER_RUN = int(os.getenv("BCP_MAX_TOOL_STEPS", "50"))
except Exception as _bcp_exc:
    print(f"[BCP warning] failed to set BCP_MAX_TOOL_STEPS: {_bcp_exc}")
'''
    if needle not in s:
        raise SystemExit("Cannot locate load_dotenv() in qwen_client.py")
    s = s.replace(needle, needle + insert, 1)
    changed = True

old = '''    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")
'''
new = '''    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    safe_qid = re.sub(r"[^A-Za-z0-9._-]+", "_", str(query_id or "single")).strip("_")
    filename = os.path.join(out_dir, f"run_{safe_qid}_{ts}.json")
'''
if old in s:
    s = s.replace(old, new, 1)
    changed = True

if changed:
    p.write_text(s)
PY
}

check_llm_model_compatibility() {
  if [[ "$SKIP_LLM_COMPAT_CHECK" -eq 1 ]]; then
    log "skipping LLM compatibility preflight"
    return 0
  fi
  local -a models=()
  if [[ "$SKIP_RUN" -eq 0 ]]; then
    models+=("$AGENT_MODEL")
  fi
  if [[ "$SKIP_JUDGE" -eq 0 && "$JUDGE_MODEL" != "$AGENT_MODEL" ]]; then
    models+=("$JUDGE_MODEL")
  elif [[ "$SKIP_JUDGE" -eq 0 && ${#models[@]} -eq 0 ]]; then
    models+=("$JUDGE_MODEL")
  fi
  if [[ ${#models[@]} -eq 0 ]]; then
    return 0
  fi
  activate_llm_env
  local -a model_paths=()
  local model
  for model in "${models[@]}"; do
    prepare_model_path "$model" "_llm"
    model_paths+=("$BCP_PREPARED_MODEL_PATH")
  done
  log "checking LLM model config compatibility with $LLM_STACK stack: ${models[*]}"
  "$LLM_PYTHON" - "${model_paths[@]}" <<'PY'
import os
import runpy
import sys
from pathlib import Path

runpy.run_path(os.path.join(os.environ["ROOT"], "tmp", "bcp_hf_ssl_bootstrap.py"))

from transformers import AutoConfig, AutoTokenizer

errors = []
for model in sys.argv[1:]:
    print(f"checking {model}")
    is_local = Path(model).exists()
    try:
        cfg = AutoConfig.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        print(
            "  transformers AutoConfig OK:",
            getattr(cfg, "model_type", None),
            getattr(cfg, "architectures", None),
        )
    except Exception as exc:
        errors.append(f"{model}: transformers AutoConfig failed: {type(exc).__name__}: {exc}")
        continue
    try:
        tok = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=True,
            local_files_only=is_local,
        )
        print("  tokenizer OK: chat_template=", bool(getattr(tok, "chat_template", None)))
    except Exception as exc:
        errors.append(f"{model}: tokenizer failed: {type(exc).__name__}: {exc}")
    try:
        from vllm.transformers_utils.config import get_config

        vcfg = get_config(model, trust_remote_code=True)
        print(
            "  vLLM get_config OK:",
            getattr(vcfg, "model_type", None),
            getattr(vcfg, "architectures", None),
        )
        try:
            from vllm.model_executor.models.registry import ModelRegistry

            archs = getattr(vcfg, "architectures", None) or []
            if hasattr(ModelRegistry, "get_supported_archs"):
                supported = set(ModelRegistry.get_supported_archs())
                hits = [arch for arch in archs if arch in supported]
                if archs and not hits:
                    errors.append(f"{model}: vLLM registry has no support for architectures {archs}")
                else:
                    print("  vLLM registry OK:", hits)
        except Exception as exc:
            print(f"  vLLM registry probe warning: {type(exc).__name__}: {exc}")
    except Exception as exc:
        errors.append(f"{model}: vLLM get_config failed: {type(exc).__name__}: {exc}")

if errors:
    print("LLM compatibility preflight failed:", file=sys.stderr)
    for err in errors:
        print(f"- {err}", file=sys.stderr)
    print(
        "Use --agent-model/--judge-model with a supported model, switch --llm-stack, "
        "or pass --skip-llm-compat-check if you intentionally want to bypass this early check.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

prepare_dataset() {
  local src="$ROOT/src/BrowseComp-Plus"
  cd "$src"
  mkdir -p data topics-qrels
  if [[ ! -s data/browsecomp_plus_decrypted.jsonl || ! -s topics-qrels/queries.tsv ]]; then
    log "downloading/decrypting BrowseComp-Plus dataset"
    local ds_snapshot="$ROOT/data/hf_snapshots/browsecomp-plus"
    snapshot_hf_dataset "Tevatron/browsecomp-plus" "$ds_snapshot"
    python - "$ds_snapshot" data/browsecomp_plus_decrypted.jsonl topics-qrels/queries.tsv <<'PY'
import importlib.util
import os
import runpy
import sys

runpy.run_path(os.path.join(os.environ["ROOT"], "tmp", "bcp_hf_ssl_bootstrap.py"))

from datasets import load_dataset as hf_load_dataset

snapshot, output, query_tsv = sys.argv[1:4]
script = "scripts_build_index/decrypt_dataset.py"
spec = importlib.util.spec_from_file_location("bcp_decrypt_dataset", script)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

def load_dataset_local(name, *args, **kwargs):
    if name == "Tevatron/browsecomp-plus":
        name = snapshot
    return hf_load_dataset(name, *args, **kwargs)

mod.load_dataset = load_dataset_local
sys.argv = [script, "--output", output, "--generate-tsv", query_tsv]
mod.main()
PY
  else
    log "dataset already prepared"
  fi
  if [[ -n "$EVAL_LIMIT" ]]; then
    head -n "$EVAL_LIMIT" topics-qrels/queries.tsv > "topics-qrels/queries.first${EVAL_LIMIT}.tsv"
  fi
}

prepare_corpus_inputs() {
  local safe="$1"
  BCP_CORPUS_DATASET_NAME="$ROOT/data/hf_snapshots/browsecomp-plus-corpus"
  BCP_CORPUS_DATASET_PATH=""
  if [[ -n "$CORPUS_LIMIT" ]]; then
    BCP_CORPUS_DATASET_NAME="json"
    BCP_CORPUS_DATASET_PATH="$ROOT/data/corpus_smoke/${safe}.first${CORPUS_LIMIT}.jsonl"
    if [[ ! -s "$BCP_CORPUS_DATASET_PATH" ]]; then
      mkdir -p "$(dirname "$BCP_CORPUS_DATASET_PATH")"
      log "creating first-$CORPUS_LIMIT corpus smoke file: $BCP_CORPUS_DATASET_PATH"
      if ! python - "$BCP_CORPUS_DATASET_PATH" "$CORPUS_LIMIT" <<'PY'
import itertools
import json
import os
import runpy
import sys

runpy.run_path(os.path.join(os.environ["ROOT"], "tmp", "bcp_hf_ssl_bootstrap.py"))

from datasets import load_dataset

output, limit = sys.argv[1], int(sys.argv[2])
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train", streaming=True)
with open(output, "w", encoding="utf-8") as f:
    for row in itertools.islice(ds, limit):
        json.dump(row, f, ensure_ascii=False)
        f.write("\n")
PY
      then
        local got
        got=$(python - "$BCP_CORPUS_DATASET_PATH" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
print(sum(1 for _ in p.open("r", encoding="utf-8")) if p.exists() else 0)
PY
)
        if (( got >= CORPUS_LIMIT )); then
          log "corpus smoke file reached $got rows despite streaming teardown warning"
        else
          echo "Failed to create corpus smoke file; got $got rows, expected $CORPUS_LIMIT" >&2
          return 1
        fi
      fi
    fi
  else
    snapshot_hf_dataset "Tevatron/browsecomp-plus-corpus" "$BCP_CORPUS_DATASET_NAME"
  fi
  export BCP_CORPUS_DATASET_NAME
  export BCP_CORPUS_DATASET_PATH
}

prepare_sorted_corpus_input() {
  local safe="$1"
  local model_dir="$2"
  local source_name="$BCP_CORPUS_DATASET_NAME"
  local source_path="$BCP_CORPUS_DATASET_PATH"
  local suffix="full"
  if [[ -n "$CORPUS_LIMIT" ]]; then
    suffix="first${CORPUS_LIMIT}"
  fi
  local sort_shards="${INDEX_SHARDS:-1}"
  local sorted_path="$ROOT/data/corpus_sorted/${safe}.${suffix}.shards${sort_shards}.len_sorted.max${INDEX_PASSAGE_MAX_LEN}.jsonl"
  if [[ -s "$sorted_path" ]]; then
    log "using shard-aware length-sorted corpus input: $sorted_path"
  else
    mkdir -p "$(dirname "$sorted_path")"
    log "creating shard-aware length-sorted corpus input: $sorted_path"
    python - "$source_name" "$source_path" "$model_dir" "$sorted_path" "$INDEX_PASSAGE_MAX_LEN" "$sort_shards" <<'PY'
import json
import os
import runpy
import sys

runpy.run_path(os.path.join(os.environ["ROOT"], "tmp", "bcp_hf_ssl_bootstrap.py"))

from datasets import concatenate_datasets, load_dataset
from transformers import AutoTokenizer

source_name, source_path, model_dir, sorted_path, max_len_s, sort_shards_s = sys.argv[1:7]
max_len = int(max_len_s)
sort_shards = max(1, int(sort_shards_s))

if source_path:
    ds = load_dataset("json", data_files=source_path, split="train")
else:
    ds = load_dataset(source_name, split="train")

tok = AutoTokenizer.from_pretrained(
    model_dir,
    trust_remote_code=True,
    local_files_only=os.path.isdir(model_dir),
)

def add_len(batch):
    texts = []
    for i, text in enumerate(batch.get("text", [])):
        title = batch.get("title", [None] * len(batch["text"]))[i]
        if title:
            text = f"{title} {text or ''}"
        texts.append((text or "").strip())
    encoded = tok(
        texts,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_attention_mask=False,
    )
    return {"_bcp_sort_len": [len(ids) for ids in encoded["input_ids"]]}

ds = ds.map(add_len, batched=True, batch_size=1024, desc="computing token lengths")
if sort_shards > 1:
    # Tevatron shards the dataset before encoding. Sort inside each future shard,
    # not globally, so no GPU gets only the longest tail of the corpus.
    parts = []
    for shard_idx in range(sort_shards):
        part = ds.shard(num_shards=sort_shards, index=shard_idx)
        parts.append(part.sort("_bcp_sort_len"))
    ds = concatenate_datasets(parts)
else:
    ds = ds.sort("_bcp_sort_len")
cols = [c for c in ds.column_names if c != "_bcp_sort_len"]
ds = ds.select_columns(cols)
ds.to_json(sorted_path, force_ascii=False)
PY
  fi
  BCP_CORPUS_DATASET_NAME="json"
  BCP_CORPUS_DATASET_PATH="$sorted_path"
  export BCP_CORPUS_DATASET_NAME
  export BCP_CORPUS_DATASET_PATH
}

download_retriever_model() {
  local repo="$1"
  local model_dir="$2"
  if [[ -s "$model_dir/config.json" || -s "$model_dir/model.safetensors.index.json" ]]; then
    log "retriever model already present: $model_dir"
    return 0
  fi
  mkdir -p "$model_dir"
  log "downloading retriever model $repo to $model_dir"
  snapshot_hf_model "$repo" "$model_dir"
}

build_faiss_index() {
  local repo="$1"
  local model_dir="$2"
  local index_dir="$3"
  local safe="$4"
  if [[ "$SKIP_INDEX" -eq 1 ]]; then
    if compgen -G "$index_dir/corpus.shard*.pkl" >/dev/null; then
      log "skipping index build; existing shards found at $index_dir"
      return 0
    fi
    echo "--skip-index was set but no corpus.shard*.pkl exists in $index_dir" >&2
    return 1
  fi
  if [[ "$FORCE_INDEX" -eq 0 ]] && compgen -G "$index_dir/corpus.shard*.pkl" >/dev/null; then
    log "index already exists: $index_dir"
    return 0
  fi
  rm -rf "$index_dir"
  mkdir -p "$index_dir"
  split_csv_list "$INDEX_GPU_DEVICES"
  local -a index_gpus=("${_BCP_SPLIT_RESULT[@]}")
  local gpu_count=${#index_gpus[@]}
  if (( gpu_count == 0 )); then
    echo "No index GPUs configured; set --index-gpu-devices." >&2
    return 2
  fi
  if [[ -z "$INDEX_SHARDS" ]]; then
    INDEX_SHARDS="$gpu_count"
  fi
  prepare_corpus_inputs "$safe"
  prepare_sorted_corpus_input "$safe" "$model_dir"

  log "building FAISS corpus embeddings for $repo with ${INDEX_SHARDS} shard(s) on GPU(s): $INDEX_GPU_DEVICES"
  local -a pids=()
  local shard gpu out log_file
  for (( shard=0; shard<INDEX_SHARDS; shard++ )); do
    gpu="${index_gpus[$((shard % gpu_count))]}"
    out="$index_dir/corpus.shard$(printf '%02d' "$shard").pkl"
    log_file="$index_dir/encode.shard$(printf '%02d' "$shard").log"
    (
      set -Eeuo pipefail
      CUDA_VISIBLE_DEVICES="$gpu" python -c "import runpy; runpy.run_path('$ROOT/tmp/bcp_hf_ssl_bootstrap.py'); runpy.run_module('tevatron.retriever.driver.encode', run_name='__main__')" \
        --model_name_or_path "$model_dir" \
        --attn_implementation "$ATTN_IMPLEMENTATION" \
        --dataset_name "$BCP_CORPUS_DATASET_NAME" \
        ${BCP_CORPUS_DATASET_PATH:+--dataset_path "$BCP_CORPUS_DATASET_PATH"} \
        --encode_output_path "$out" \
        --dataset_number_of_shards "$INDEX_SHARDS" \
        --dataset_shard_index "$shard" \
        --passage_max_len "$INDEX_PASSAGE_MAX_LEN" \
        --normalize \
        --pooling eos \
        --passage_prefix "" \
        --per_device_eval_batch_size "$RETRIEVAL_BATCH_SIZE" \
        --fp16
    ) >"$log_file" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "At least one index shard failed. Check $index_dir/encode.shard*.log" >&2
    return 1
  fi
  log "index shards completed: $index_dir/corpus.shard*.pkl"
}

split_query_file() {
  local query_file="$1"
  local out_dir="$2"
  local workers="$3"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  python - "$query_file" "$out_dir" "$workers" <<'PY'
from pathlib import Path
import sys

query_file = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
workers = max(1, int(sys.argv[3]))
handles = []
for i in range(workers):
    handles.append((out_dir / f"queries.worker{i:02d}.tsv").open("w", encoding="utf-8"))
try:
    with query_file.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            handles[n % workers].write(line)
finally:
    for h in handles:
        h.close()
for p in sorted(out_dir.glob("queries.worker*.tsv")):
    if p.stat().st_size:
        print(p)
    else:
        p.unlink()
PY
}

run_agent_and_judge() {
  local repo="$1"
  local safe="$2"
  local model_dir="$3"
  local index_dir="$4"
  local log_dir="$5"
  local src="$ROOT/src/BrowseComp-Plus"
  local run_id
  run_id=$(basename "$log_dir")
  local run_dir="$ROOT/runs/$safe/$run_id"
  local eval_root="$ROOT/evals/$safe"
  local query_file="topics-qrels/queries.tsv"
  [[ -n "$EVAL_LIMIT" ]] && query_file="topics-qrels/queries.first${EVAL_LIMIT}.tsv"

  mkdir -p "$run_dir" "$eval_root"
  cd "$src"
  export BCP_MAX_TOOL_STEPS="$MAX_STEPS"
  prepare_corpus_inputs "$safe"
  prepare_sorted_corpus_input "$safe" "$model_dir"
  local corpus_dataset="$BCP_CORPUS_DATASET_NAME"
  local corpus_dataset_path="$BCP_CORPUS_DATASET_PATH"

  if [[ "$SKIP_RUN" -eq 1 ]]; then
    log "skipping Qwen end-to-end run"
  else
  prepare_model_path "$AGENT_MODEL" "_llm"
  local agent_model_path="$BCP_PREPARED_MODEL_PATH"
  local vllm_port
  local mcp_port
  vllm_port=$(find_free_port "$AGENT_PORT" "vLLM agent")
  mcp_port=$(find_free_port "$MCP_PORT" "MCP search server")
  local vllm_pid=""
  local mcp_pid=""
  trap 'stop_pid "$mcp_pid"; stop_pid "$vllm_pid"' RETURN

  log "starting vLLM agent server: $AGENT_MODEL"
  vllm_pid=$(start_bg "vllm-agent" "$log_dir/vllm_agent.log" \
    env CUDA_VISIBLE_DEVICES="$AGENT_GPU_DEVICES" "$LLM_VLLM_BIN" serve "$agent_model_path" \
      --host 127.0.0.1 \
      --port "$vllm_port" \
      --served-model-name "$AGENT_MODEL" \
      --trust-remote-code \
      --tensor-parallel-size "$AGENT_TENSOR_PARALLEL_SIZE" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTIL" \
      --enable-prefix-caching)
  wait_http "http://127.0.0.1:${vllm_port}/v1/models" "vLLM agent" 1200

  log "starting MCP search server for $repo"
  mcp_pid=$(start_bg "mcp-search" "$log_dir/mcp_search.log" \
    env CUDA_VISIBLE_DEVICES="$SEARCH_GPU_DEVICES" BCP_ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION" python searcher/mcp_server.py \
      --searcher-type faiss \
      --index-path "$index_dir/corpus.shard*.pkl" \
      --model-name "$model_dir" \
      --dataset-name "$corpus_dataset" \
      ${corpus_dataset_path:+--dataset-path "$corpus_dataset_path"} \
      --normalize \
      --get-document \
      --snippet-max-tokens 512 \
      --k 5 \
      --transport sse \
      --port "$mcp_port")
  wait_http "http://127.0.0.1:${mcp_port}/mcp" "MCP search server" 900 || true
  sleep 10

  log "running Qwen agent on $query_file with search + get_document, max steps=$MAX_STEPS, workers=$AGENT_WORKERS"
  local -a chunk_files=()
  if (( AGENT_WORKERS > 1 )); then
    mapfile -t chunk_files < <(split_query_file "$src/$query_file" "$log_dir/query_chunks" "$AGENT_WORKERS")
  else
    chunk_files=("$src/$query_file")
  fi

  local -a agent_pids=()
  local chunk idx
  idx=0
  for chunk in "${chunk_files[@]}"; do
    local agent_pid
    agent_pid=$(start_bg "qwen-agent-$idx" "$log_dir/qwen_agent.worker$(printf '%02d' "$idx").log" \
      python search_agent/qwen_client.py \
        --query "$chunk" \
        --model "$AGENT_MODEL" \
        --model-server "http://127.0.0.1:${vllm_port}/v1" \
        --mcp-url "http://127.0.0.1:${mcp_port}/mcp" \
        --output-dir "$run_dir" \
        --query-template QUERY_TEMPLATE \
        --max_tokens "$MAX_TOKENS")
    agent_pids+=("$agent_pid")
    idx=$((idx + 1))
  done

  local failed=0
  for pid in "${agent_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if (( failed )); then
    echo "At least one Qwen agent worker failed. Check $log_dir/qwen_agent.worker*.log" >&2
    return 1
  fi

  stop_pid "$mcp_pid"; mcp_pid=""
  stop_pid "$vllm_pid"; vllm_pid=""
  fi

  if [[ "$SKIP_JUDGE" -eq 1 ]]; then
    log "skipping judge"
    return 0
  fi
  log "judging run outputs with $JUDGE_MODEL"
  prepare_model_path "$JUDGE_MODEL" "_llm"
  local judge_model_path="$BCP_PREPARED_MODEL_PATH"
  CUDA_VISIBLE_DEVICES="$JUDGE_GPU_DEVICES" "$LLM_PYTHON" scripts_evaluation/evaluate_run.py \
    --input_dir "$run_dir" \
    --ground_truth data/browsecomp_plus_decrypted.jsonl \
    --eval_dir "$eval_root" \
    --model "$judge_model_path" \
    --tensor_parallel_size "$JUDGE_TENSOR_PARALLEL_SIZE" \
    --batch_size "${BCP_JUDGE_BATCH_SIZE:-64}" \
    --force \
    2>&1 | tee "$log_dir/judge.log"

  local summary
  summary=$(find "$eval_root" -path "*${run_id}*/evaluation_summary.json" -print | sort | tail -n 1 || true)
  if [[ -n "$summary" ]]; then
    log "summary for $repo:"
    python - "$summary" <<'PY'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
keys = ["LLM", "Retriever", "Accuracy (%)", "Recall (%)", "avg_tool_stats", "Calibration Error (%)", "Evaluation Date"]
for k in keys:
    print(f"{k}: {d.get(k)}")
print(f"summary_path: {p}")
PY
  else
    log "no evaluation_summary.json found for $repo"
  fi
}

cleanup_model_and_index() {
  local model_dir="$1"
  local index_dir="$2"
  if [[ "$KEEP_MODELS" -eq 0 ]]; then
    log "removing retriever model dir: $model_dir"
    rm -rf "$model_dir"
  fi
  if [[ "$KEEP_INDEXES" -eq 0 ]]; then
    log "removing index dir: $index_dir"
    rm -rf "$index_dir"
  fi
}

run_one_repo() {
  local repo="$1"
  local safe
  safe=$(safe_name "$repo")
  local model_dir="$ROOT/models/$safe"
  local index_dir="$ROOT/indexes/$safe"
  local log_dir="$ROOT/logs/$(timestamp)__${safe}"
  local rc=0
  local cleanup_rc=0
  local status="ok"
  local summary=""
  local run_id=""
  mkdir -p "$log_dir"
  run_id=$(basename "$log_dir")

  log "===== BEGIN $repo ====="
  log "logs: $log_dir"
  set +e
  (
    set -Eeuo pipefail
    download_retriever_model "$repo" "$model_dir" || exit $?
    build_faiss_index "$repo" "$model_dir" "$index_dir" "$safe" || exit $?
    run_agent_and_judge "$repo" "$safe" "$model_dir" "$index_dir" "$log_dir" || exit $?
  ) 2>&1 | redact_stream | tee "$log_dir/pipeline.log"
  rc=${PIPESTATUS[0]}
  cleanup_model_and_index "$model_dir" "$index_dir" 2>&1 | redact_stream | tee -a "$log_dir/pipeline.log"
  cleanup_rc=${PIPESTATUS[0]}
  set -e
  summary=$(find "$ROOT/evals/$safe" -path "*${run_id}*/evaluation_summary.json" -print | sort | tail -n 1 || true)
  if (( rc != 0 )); then
    status="failed"
  elif (( cleanup_rc != 0 )); then
    status="cleanup_failed"
    rc="$cleanup_rc"
  fi
  record_repo_status "$repo" "$safe" "$status" "$rc" "$log_dir" "$summary"
  if [[ -n "$summary" ]]; then
    log "recorded summary: $summary"
  fi
  log "===== END $repo ====="
  return "$rc"
}

main() {
  finalize_gpu_defaults
  configure_proxy
  prepare_dirs
  preflight_checks
  clone_or_update_repo
  patch_dense_searcher_lazy_imports
  patch_faiss_searcher_dataset_path
  patch_searcher_tools_tokenizer
  if [[ "$SKIP_ENV" -eq 0 ]]; then
    setup_env
    setup_llm_env
  else
    activate_env
    if [[ "$PREPARE_ONLY" -eq 0 ]]; then
      activate_llm_env
    fi
  fi
  patch_qwen_step_limit
  if [[ "$PREPARE_ONLY" -eq 0 ]]; then
    check_llm_model_compatibility
  fi
  prepare_dataset
  if [[ "$PREPARE_ONLY" -eq 1 ]]; then
    log "prepare-only complete"
    exit 0
  fi
  init_batch_status
  local failed=0
  for repo in "${REPOS[@]}"; do
    if ! run_one_repo "$repo"; then
      failed=1
      if [[ "$FAIL_FAST" -eq 1 ]]; then
        log "fail-fast enabled; stopping after failed repo: $repo"
        break
      fi
    fi
  done
  log "batch status written to $BATCH_STATUS"
  exit "$failed"
}

main "$@"
