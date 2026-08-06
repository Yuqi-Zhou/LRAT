#!/usr/bin/env python3
"""Shared, side-effect-free helpers for the BC-Plus campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import socket
import statistics
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ROOT = Path(
    os.environ.get("BCP_CAMPAIGN_ROOT", str(PACKAGE_ROOT))
).resolve()
# XIR-B treats the exact autonomous-call cap as a valid terminal condition
# only when the persisted transcript proves that the cap, rather than an
# infrastructure or protocol error, ended the trajectory.
MAX_INTERACTION_STEPS = 50
MAX_STEP_LEGACY_FAILURES = frozenset(
    {
        "final_answer_last",
        "final_answer_nonempty",
        "required_final_format",
        "status_completed",
    }
)
MAX_STEP_ZERO_ERROR_COUNTERS = (
    "native_protocol_retry_count",
    "malformed_tool_call_retry_count",
    "native_tool_rejection_count",
    "native_tool_context_truncation_count",
    "native_history_compaction_count",
    "final_markup_repair_count",
)
# The campaign watchdog reads the unified three-agent config.  BCP_CAMPAIGN_CONFIG
# remains supported for offline preparation/audit helpers, but all live campaign
# children use the same campaign root and durable queue.
CONFIG_PATH = Path(
    os.environ.get(
        "BCP_CAMPAIGN_CONFIG",
        str(PACKAGE_ROOT / "configs" / "campaign.example.json"),
    )
).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def resolve_path(value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = CONFIG_PATH.parent / path
        return str(path.resolve())

    for key in (
        "campaign_root",
        "lab_root",
        "competition_root",
        "browsecomp_source",
        "query_path",
        "output_root",
    ):
        if str(config.get(key, "")).strip():
            config[key] = resolve_path(str(config[key]))
    if Path(config["campaign_root"]).resolve() != CAMPAIGN_ROOT:
        raise RuntimeError("config campaign_root does not match script location")
    if int(config["max_interaction_steps"]) != MAX_INTERACTION_STEPS:
        raise RuntimeError(
            f"campaign interaction limit must be exactly {MAX_INTERACTION_STEPS}"
        )
    if int(config["max_output_tokens"]) != 4096:
        raise RuntimeError("campaign output-token budget drifted from proven 4096")
    if not config["models"] or len(config["retrievers"]) != 6:
        raise RuntimeError("campaign config must contain at least 1 model and 6 retrievers")
    endpoints = config.get("model_endpoints") or {}
    ports = config.get("mcp_ports") or {}
    if set(endpoints) != {str(item["profile"]) for item in config["models"]}:
        raise RuntimeError("campaign config must persist one endpoint for every model profile")
    if len(ports) != 6 or set(ports) != {str(item["slug"]) for item in config["retrievers"]}:
        raise RuntimeError("campaign config must persist exactly six retriever MCP ports")
    for model in config["models"]:
        profile = str(model.get("profile"))
        if not str(model.get("model_endpoint", endpoints.get(profile, ""))).strip():
            raise RuntimeError(f"model endpoint missing for {profile}")
        if profile == "deepseek-v4-flash":
            args = [str(value) for value in model.get("server_args", [])]
            required = {
                "--tokenizer-mode": "deepseek_v4",
                "--tool-call-parser": "deepseek_v4",
                "--reasoning-parser": "deepseek_v4",
            }
            if any(flag not in args or args[args.index(flag) + 1] != value for flag, value in required.items()):
                raise RuntimeError("DeepSeek profile is missing official deepseek_v4 parser flags")
            if "--enable-auto-tool-choice" not in args:
                raise RuntimeError("DeepSeek profile must keep tool_choice=auto server support")
    return config


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        (json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n").encode(
            "utf-8"
        ),
    )


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def safe_name(value: str) -> str:
    result = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return result.strip("._-") or "item"


def combination_root(config: dict[str, Any], profile: str, retriever_slug: str) -> Path:
    return (
        Path(config["campaign_root"])
        / "combinations"
        / safe_name(profile)
        / safe_name(retriever_slug)
    )


def query_rows(config: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(config["query_path"]).open(newline="", encoding="utf-8") as handle:
        for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), 1):
            if len(row) != 2:
                raise RuntimeError(f"malformed query row at line {line_number}")
            query_id, query_text = row[0].strip(), row[1].strip()
            if not query_id or not query_text:
                raise RuntimeError(f"empty query field at line {line_number}")
            rows.append((query_id, query_text))
    ids = [query_id for query_id, _ in rows]
    if len(rows) != 830 or len(set(ids)) != 830:
        raise RuntimeError(
            f"expected 830 rows and unique IDs, got rows={len(rows)} unique={len(set(ids))}"
        )
    return rows


def file_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        stat = path.stat()
        evidence.update(
            {
                "kind": "file",
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    elif path.is_dir():
        evidence["kind"] = "directory"
    return evidence


def sha256_small(path: Path, limit: int = 16 * 1024 * 1024) -> str | None:
    if not path.is_file() or path.stat().st_size > limit:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_weight_evidence(model_path: Path) -> dict[str, Any]:
    weights = sorted(model_path.glob("*.safetensors")) if model_path.is_dir() else []
    index_path = model_path / "model.safetensors.index.json"
    expected_files: set[str] = set()
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text(encoding="utf-8")).get(
                "weight_map", {}
            )
            expected_files = {str(name) for name in weight_map.values()}
        except (OSError, json.JSONDecodeError):
            expected_files = set()
    actual_names = {path.name for path in weights}
    missing = sorted(expected_files - actual_names)
    return {
        "weight_file_count": len(weights),
        "weight_bytes": sum(path.stat().st_size for path in weights),
        "index_path": str(index_path),
        "index_exists": index_path.is_file(),
        "index_expected_weight_files": len(expected_files),
        "missing_weight_files_count": len(missing),
        "missing_weight_files_sample": missing[:12],
        "last_weight_file": weights[-1].name if weights else None,
    }


def expected_weight_names(model: dict[str, Any]) -> set[str]:
    """Return the exact weight filenames pinned by a model manifest.

    Most snapshots use one-based names such as ``model-00001-of-00046``.
    gpt-oss-120b is an official exception: its public revision uses
    ``model-00000-of-00014`` through ``model-00014-of-00014``.  The manifest
    may therefore provide explicit names; silently deriving a different
    naming convention would make a completed download unverifiable.
    """
    explicit = model.get("weight_file_names")
    if explicit:
        return {str(name) for name in explicit}
    count = int(model["expected_weight_shards"])
    start = int(model.get("weight_shard_start", 1))
    total = int(model.get("weight_shard_total", count))
    return {
        f"model-{index:05d}-of-{total:05d}.safetensors"
        for index in range(start, start + count)
    }


def model_completion_evidence(model: dict[str, Any]) -> dict[str, Any]:
    """Return strict, profile-specific model snapshot evidence.

    The completion marker is necessary but never sufficient: every indexed
    shard and every required tokenizer/config file must be present.
    """
    model_path = Path(model["model_path"])
    marker_path = model_path / ".model_lab_download.json"
    marker = read_json(marker_path, {}) or {}
    weight_info = model_weight_evidence(model_path)
    expected_count = int(model["expected_weight_shards"])
    expected_names = expected_weight_names(model)
    actual_paths = sorted(model_path.glob("model-*-of-*.safetensors"))
    actual_names = {path.name for path in actual_paths}
    required_paths = [
        model_path / relative for relative in model.get("required_model_files", [])
    ]
    index_path = model_path / "model.safetensors.index.json"
    index_valid = False
    index_names: set[str] = set()
    index_error_class = None
    try:
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        index_names = {str(value) for value in weight_map.values()}
        index_valid = bool(weight_map) and index_names == expected_names
    except Exception as exc:
        index_error_class = type(exc).__name__
    complete = bool(
        marker.get("complete") is True
        and marker.get("repo_id") == model["repo_id"]
        and marker.get("revision") == model["revision"]
        and actual_names == expected_names
        and all(path.is_file() and path.stat().st_size > 0 for path in actual_paths)
        and all(path.is_file() and path.stat().st_size > 0 for path in required_paths)
        and index_valid
        and weight_info["weight_bytes"] >= int(int(model["expected_model_bytes"]) * 0.99)
    )
    return {
        "passed": complete,
        "model_path": str(model_path),
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "expected_model_bytes": int(model["expected_model_bytes"]),
        "expected_weight_shards": expected_count,
        "expected_weight_names": sorted(expected_names),
        "actual_weight_shards": len(actual_paths),
        "missing_weight_shards_count": len(expected_names - actual_names),
        "missing_weight_shards_sample": sorted(expected_names - actual_names)[:12],
        "unexpected_weight_shards": sorted(actual_names - expected_names),
        "required_files": [file_evidence(path) for path in required_paths],
        "index_valid": index_valid,
        "index_weight_shards": len(index_names),
        "index_error_class": index_error_class,
        "completion_manifest_path": str(marker_path),
        "completion_manifest_exists": marker_path.is_file(),
        "completion_manifest_complete": marker.get("complete") is True,
        "completion_manifest_repo_matches": marker.get("repo_id") == model["repo_id"],
        "completion_manifest_revision_matches": marker.get("revision")
        == model["revision"],
        **weight_info,
    }


def static_prerequisite_audit(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, evidence: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})

    try:
        rows = query_rows(config)
        add(
            "bc_plus_queries_exact_830_unique",
            True,
            {
                "path": config["query_path"],
                "rows": len(rows),
                "unique_query_ids": len({row[0] for row in rows}),
            },
        )
    except Exception as exc:
        add(
            "bc_plus_queries_exact_830_unique",
            False,
            {"path": config["query_path"], "error_class": type(exc).__name__},
        )

    source = Path(config["browsecomp_source"])
    required_source = [
        source / "searcher" / "mcp_server.py",
        source / "search_agent" / "prompts.py",
        Path(config["lab_root"]) / "src" / "model_agent_client.py",
        Path(config["search_python"]),
    ]
    add(
        "browsecomp_runtime_assets",
        all(path.exists() for path in required_source),
        [file_evidence(path) for path in required_source],
    )

    for model in config["models"]:
        profile = model["profile"]
        model_path = Path(model["model_path"])
        env_path = Path(model["environment_path"])
        backend_path = Path(model["backend_status"])
        completion = model_completion_evidence(model)
        add(
            f"agent_model:{profile}",
            completion["passed"],
            completion,
        )
        environment_manifest = read_json(
            env_path / ".model_lab_environment.json", {}
        )
        add(
            f"agent_environment:{profile}",
            bool(environment_manifest) and (env_path / "bin" / "python").is_file(),
            {
                "environment_path": str(env_path),
                "manifest_path": str(env_path / ".model_lab_environment.json"),
                "manifest_exists": bool(environment_manifest),
                "python_exists": (env_path / "bin" / "python").is_file(),
            },
        )
        backend = read_json(backend_path, {})
        add(
            f"agent_backend:{profile}",
            bool(backend.get("passed")),
            {
                "path": str(backend_path),
                "exists": backend_path.is_file(),
                "passed": bool(backend.get("passed")),
                "checked_at": backend.get("checked_at"),
                "versions": backend.get("versions"),
            },
        )

    expected_shards = {f"corpus.shard{index:02d}.pkl" for index in range(8)}
    for retriever in config["retrievers"]:
        slug = retriever["slug"]
        index_path = Path(retriever["index_path"])
        shards = sorted(index_path.glob("corpus.shard*.pkl"))
        shard_names = {path.name for path in shards}
        add(
            f"retriever_index:{slug}",
            shard_names == expected_shards and all(path.stat().st_size > 0 for path in shards),
            {
                "repo_id": retriever["repo_id"],
                "path": str(index_path),
                "shard_count": len(shards),
                "shard_names": sorted(shard_names),
                "total_bytes": sum(path.stat().st_size for path in shards),
                "all_nonempty": bool(shards)
                and all(path.stat().st_size > 0 for path in shards),
            },
        )
        retriever_model = Path(retriever["model_path"])
        retriever_weights = sorted(retriever_model.glob("*.safetensors"))
        required_model_files = [
            retriever_model / ".bcp_snapshot_complete",
            retriever_model / "config.json",
            retriever_model / "tokenizer_config.json",
        ]
        add(
            f"retriever_model:{slug}",
            all(path.is_file() for path in required_model_files)
            and bool(retriever_weights)
            and all(path.stat().st_size > 0 for path in retriever_weights),
            {
                "repo_id": retriever["repo_id"],
                "path": str(retriever_model),
                "required_files": [file_evidence(path) for path in required_model_files],
                "weight_file_count": len(retriever_weights),
                "weight_bytes": sum(path.stat().st_size for path in retriever_weights),
            },
        )
        corpus = Path(retriever["corpus_path"])
        add(
            f"retrieval_corpus:{slug}",
            corpus.is_file() and corpus.stat().st_size > 0,
            file_evidence(corpus),
        )

    failed = [check for check in checks if not check["passed"]]
    return {
        "checked_at": now(),
        "passed": not failed,
        "check_count": len(checks),
        "failed_count": len(failed),
        "failed_checks": [check["name"] for check in failed],
        "checks": checks,
    }


def gpu_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        run = subprocess.run(
            command, text=True, capture_output=True, timeout=20, check=False
        )
    except Exception as exc:
        return {"checked_at": now(), "passed": False, "error_class": type(exc).__name__}
    gpus: list[dict[str, Any]] = []
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_total_mib": int(parts[3]),
                        "memory_used_mib": int(parts[4]),
                        "utilization_percent": int(parts[5]),
                    }
                )
            except ValueError:
                continue
    passed = run.returncode == 0 and [gpu["index"] for gpu in gpus] == list(range(8))
    return {
        "checked_at": now(),
        "passed": passed,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "nvidia_smi_returncode": run.returncode,
    }


def cuda_compute_inventory() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        run = subprocess.run(
            command, text=True, capture_output=True, timeout=20, check=False
        )
    except Exception as exc:
        return {"passed": False, "error_class": type(exc).__name__, "entries": []}
    entries: list[dict[str, Any]] = []
    if run.returncode == 0:
        for line in run.stdout.splitlines():
            parts = [part.strip() for part in line.split(",", 3)]
            if len(parts) != 4:
                continue
            try:
                entries.append(
                    {
                        "reported_pid": int(parts[0]),
                        "gpu_uuid": parts[1],
                        "used_memory_mib": int(parts[2]),
                        "process_name": parts[3],
                    }
                )
            except ValueError:
                continue
    return {
        "passed": run.returncode == 0,
        "returncode": run.returncode,
        "entries": entries,
    }


def local_keepalive_evidence() -> dict[str, Any]:
    workers: list[dict[str, Any]] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        script_positions = [
            index
            for index, value in enumerate(fields)
            if Path(value).name == "smart_gpu_keepalive_cuda.py"
        ]
        if not script_positions:
            continue
        script_index = script_positions[0]
        subcommand = fields[script_index + 1] if script_index + 1 < len(fields) else ""
        if subcommand not in {"worker", "launch"}:
            continue
        role = "worker" if subcommand == "worker" else "launcher"
        gpu = None
        if "--gpu" in fields:
            try:
                gpu = int(fields[fields.index("--gpu") + 1])
            except (ValueError, IndexError):
                gpu = None
        workers.append({"pid": int(entry.name), "role": role, "gpu": gpu})
    return {
        "present": bool(workers),
        "worker_count": sum(item["role"] == "worker" for item in workers),
        "workers": sorted(workers, key=lambda item: item["pid"]),
    }


def local_gpu_process_evidence() -> dict[str, Any]:
    """Detect locally visible GPU-runtime launchers, including pre-allocation races."""
    processes = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = [
                value.decode("utf-8", errors="replace")
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        marker = None
        if "vllm.entrypoints.cli.main" in fields:
            marker = "vllm.entrypoints"
        elif any(Path(value).name == "torchrun" for value in fields):
            marker = "torchrun"
        elif "torch.distributed.run" in fields:
            marker = "torch.distributed.run"
        elif fields and fields[0].startswith("ray::"):
            marker = "ray::"
        if not marker:
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            environment = []
        campaign_owned = any(
            value
            == f"BCP_CAMPAIGN_ROOT={CAMPAIGN_ROOT}".encode()
            for value in environment
        )
        processes.append(
            {
                "pid": int(entry.name),
                "marker": marker,
                "campaign_owned": campaign_owned,
            }
        )
    unexpected = [item for item in processes if not item["campaign_owned"]]
    return {
        "process_count": len(processes),
        "unexpected_count": len(unexpected),
        "processes": sorted(processes, key=lambda item: item["pid"]),
        "unexpected_processes": sorted(unexpected, key=lambda item: item["pid"]),
    }


def gpu_admission(config: dict[str, Any]) -> dict[str, Any]:
    inventory = gpu_inventory()
    compute = cuda_compute_inventory()
    threshold = int(config["gpu_admission_memory_mib"])
    keepalive = local_keepalive_evidence()
    local_gpu_processes = local_gpu_process_evidence()
    memory_clear = bool(inventory.get("passed")) and all(
        gpu["memory_used_mib"] <= threshold for gpu in inventory.get("gpus", [])
    )
    compute_entries = compute.get("entries", [])
    exact_keepalive_workers = bool(
        keepalive.get("worker_count") == 8
        and {item.get("gpu") for item in keepalive.get("workers", []) if item.get("role") == "worker"}
        == set(range(8))
    )
    # NVML reports host-namespace PIDs on this worker while /proc exposes the
    # container namespace, so PID equality is unavailable.  Admission instead
    # requires the locally proven one-worker-per-GPU keeper topology, no more
    # NVML contexts than those eight workers, and every context at the keeper
    # memory floor. Any additional or larger context blocks launch.
    compute_matches_keepalive_floor = bool(
        compute.get("passed")
        and exact_keepalive_workers
        and len(compute_entries) <= 8
        and all(int(item["used_memory_mib"]) <= threshold for item in compute_entries)
    )
    passed = bool(
        inventory.get("passed")
        and memory_clear
        and compute_matches_keepalive_floor
        and local_gpu_processes["unexpected_count"] == 0
    )
    return {
        "checked_at": now(),
        "passed": passed,
        "memory_threshold_mib": threshold,
        "memory_clear": memory_clear,
        "smart_keepalive_allowed_exception": keepalive,
        "local_gpu_runtime_processes": local_gpu_processes,
        "cuda_compute_inventory": compute,
        "compute_matches_keepalive_floor": compute_matches_keepalive_floor,
        "inventory": inventory,
        "reason": (
            "all_eight_gpus_at_or_below_keepalive_floor"
            if passed
            else (
                "unrelated_local_gpu_runtime_present"
                if local_gpu_processes["unexpected_count"]
                else "cuda_memory_above_admission_threshold_or_inventory_invalid"
            )
        ),
    }


def stable_gpu_admission(
    config: dict[str, Any], *, samples: int | None = None, interval: float | None = None
) -> dict[str, Any]:
    sample_count = int(samples or config["gpu_stable_samples"])
    sample_interval = float(
        config["gpu_recheck_seconds"] if interval is None else interval
    )
    evidence = []
    for index in range(sample_count):
        evidence.append(gpu_admission(config))
        if index + 1 < sample_count:
            time.sleep(sample_interval)
    return {
        "checked_at": now(),
        "passed": len(evidence) == sample_count
        and all(item.get("passed") for item in evidence),
        "sample_count": sample_count,
        "sample_interval_seconds": sample_interval,
        "samples": evidence,
    }


def prerequisite_failures_for_profile(
    audit: dict[str, Any], profile: str
) -> list[dict[str, Any]]:
    failures = []
    for check in audit.get("checks", []):
        if check.get("passed"):
            continue
        name = str(check.get("name", ""))
        if name.startswith("agent_") and not name.endswith(f":{profile}"):
            continue
        failures.append(check)
    return failures


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", int(port)))
        except OSError:
            return False
    return True


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def latency_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    timings = [float(value) for value in values]
    return {
        "count": len(timings),
        "mean_seconds": statistics.fmean(timings) if timings else None,
        "median_seconds": statistics.median(timings) if timings else None,
        "p95_seconds": percentile(timings, 0.95),
        "p99_seconds": percentile(timings, 0.99),
        "max_seconds": max(timings) if timings else None,
    }


def max_step_termination_evidence(
    payload: dict[str, Any], checks: dict[str, bool]
) -> dict[str, Any]:
    """Prove a healthy terminal step-cap trajectory without fabricating output.

    The ordinary validator intentionally requires a natural final answer.  This
    separate predicate is the only exception: it accepts an empty final only
    when every persisted model turn ended in a real tool call and all protocol,
    context, and tool-result evidence is intact.  It is deliberately strict so
    an infrastructure or malformed partial record cannot become completed just
    because its step counter reached 50.
    """
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    result = payload.get("result")
    raw_response = payload.get("raw_response")
    native_history = payload.get("native_message_history")
    response_evidence = metadata.get("native_response_evidence")
    calls = [
        item
        for item in result or []
        if isinstance(item, dict) and item.get("type") == "tool_call"
    ]
    failures: list[str] = []

    def require(condition: bool, reason: str) -> None:
        if not condition:
            failures.append(reason)

    status = payload.get("status")
    terminated_by = payload.get("terminated_by")
    require(status in {"incomplete", "completed"}, "status_not_terminal_candidate")
    if status == "completed":
        require(terminated_by == "max_steps_limit", "completed_without_max_step_marker")
    require(not str(payload.get("final_answer") or "").strip(), "final_answer_present")
    require(
        not any(
            item.get("type") == "output_text"
            and str(item.get("content") or item.get("output") or "").strip()
            for item in result or []
            if isinstance(item, dict)
        ),
        "output_text_present",
    )
    error_class = payload.get("error_class")
    require(
        error_class is None
        or (isinstance(error_class, str) and error_class in {"", "IncompleteTrajectory"}),
        "error_class_not_terminal_only",
    )
    require(payload.get("interaction_step_count") == MAX_INTERACTION_STEPS, "interaction_steps_not_exactly_50")
    require(payload.get("llm_call_count") == MAX_INTERACTION_STEPS, "llm_call_count_not_exactly_50")
    require(isinstance(result, list) and bool(result), "result_empty")
    require(isinstance(raw_response, list) and bool(raw_response), "raw_response_empty")
    require(isinstance(native_history, list) and bool(native_history), "native_history_empty")
    require(bool(calls), "tool_calls_empty")
    require(
        int(payload.get("executed_tool_call_count", -1) or -1) == len(calls),
        "executed_tool_call_count_mismatch",
    )
    require(
        isinstance(raw_response, list)
        and bool(raw_response)
        and isinstance(raw_response[-1], dict)
        and raw_response[-1].get("role") == "function",
        "raw_trajectory_does_not_end_with_tool_result",
    )
    require(
        isinstance(native_history, list)
        and bool(native_history)
        and isinstance(native_history[-1], dict)
        and native_history[-1].get("role") == "tool",
        "native_trajectory_does_not_end_with_tool_result",
    )
    require(isinstance(response_evidence, list), "native_response_evidence_missing")
    if isinstance(response_evidence, list):
        require(len(response_evidence) == MAX_INTERACTION_STEPS, "native_response_evidence_count_mismatch")
        for index, item in enumerate(response_evidence):
            require(isinstance(item, dict), f"native_response_evidence_{index}_malformed")
            if not isinstance(item, dict):
                continue
            require(
                item.get("finish_reason") == "tool_calls",
                f"native_response_evidence_{index}_not_tool_call_turn",
            )
            require(bool(item.get("tool_calls")), f"native_response_evidence_{index}_has_no_tool_call")

    for key in MAX_STEP_ZERO_ERROR_COUNTERS:
        value = metadata.get(key, 0)
        try:
            is_zero = int(value or 0) == 0
        except (TypeError, ValueError):
            is_zero = False
        require(is_zero, f"{key}_nonzero")
    mcp_gate = metadata.get("mcp_init_gate")
    mcp_gate = mcp_gate if isinstance(mcp_gate, dict) else {}
    require(
        mcp_gate.get("graceful_shutdown_error_class") in {None, ""},
        "mcp_graceful_shutdown_error",
    )
    require(metadata.get("complete_tool_returns_persisted") is True, "tool_returns_not_persisted")

    # All non-final checks must remain true.  The four exceptions are exactly
    # the old validator's expected rejection for an otherwise healthy cap stop.
    for name, passed in checks.items():
        if name not in MAX_STEP_LEGACY_FAILURES:
            require(bool(passed), f"validation_check_failed:{name}")

    return {
        "passed": not failures,
        "terminated_by": "max_steps_limit" if not failures else None,
        "reason": None if not failures else ";".join(sorted(set(failures))),
        "failed_requirements": sorted(set(failures)),
        "allowed_legacy_validation_failures": sorted(MAX_STEP_LEGACY_FAILURES),
    }


def validate_trajectory(
    path: Path,
    *,
    expected_query_id: str | None = None,
    expected_profile: str | None = None,
    expected_retriever_slug: str | None = None,
) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "path": str(path),
            "passed": False,
            "error_class": type(exc).__name__,
            "checks": {"valid_json": False},
        }
    invalidation_markers = []
    resolved_path = path.resolve()
    for parent in (resolved_path.parent, *resolved_path.parents):
        for marker_name in ("INVALIDATED_CONFIG_MISMATCH.json", "invalidated_config_mismatch.json"):
            marker = parent / marker_name
            if marker.is_file():
                invalidation_markers.append(str(marker))
    if invalidation_markers:
        return {
            "path": str(path),
            "passed": False,
            "error_class": "invalidated_config_mismatch",
            "checks": {
                "valid_json": True,
                "not_invalidated_config_mismatch": False,
            },
            "invalidation_markers": invalidation_markers,
        }
    required_top_level = {
        "metadata",
        "query_id",
        "status",
        "tool_call_counts",
        "result",
        "raw_response",
        "native_message_history",
        "interaction_step_count",
        "final_answer",
        "timing",
    }
    complete_schema = required_top_level.issubset(payload)
    counts = payload.get("tool_call_counts") or {}
    result = payload.get("result")
    calls = [item for item in result or [] if item.get("type") == "tool_call"]

    def nonempty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (dict, list)):
            return bool(value)
        return bool(str(value).strip())

    def parse_json(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return None
        return value

    def docids(value: Any) -> set[str]:
        value = parse_json(value)
        if isinstance(value, dict):
            if value.get("docid") is not None:
                return {str(value["docid"])}
            nested = value.get("documents") or value.get("results")
            return docids(nested) if nested is not None else set()
        if isinstance(value, list):
            found: set[str] = set()
            for item in value:
                found.update(docids(item))
            return found
        return set()

    search_docids: set[str] = set()
    requested_document_ids: list[str] = []
    returned_document_ids: list[str] = []
    tool_arguments_valid = True
    tool_schema = {
        "search": {"query": str},
        "get_document": {"docid": str},
    }
    for item in calls:
        arguments = parse_json(item.get("arguments"))
        if not isinstance(arguments, dict):
            tool_arguments_valid = False
            arguments = {}
        name = item.get("tool_name")
        expected_arguments = tool_schema.get(name)
        if expected_arguments is None or set(arguments) != set(expected_arguments):
            tool_arguments_valid = False
        elif any(
            not isinstance(arguments[key], value_type) or not str(arguments[key]).strip()
            for key, value_type in expected_arguments.items()
        ):
            tool_arguments_valid = False
        if name == "search":
            search_docids.update(docids(item.get("output")))
        elif name == "get_document":
            requested = arguments.get("docid")
            if requested is None:
                requested_values = arguments.get("docids") or []
                requested_values = (
                    requested_values if isinstance(requested_values, list) else [requested_values]
                )
            else:
                requested_values = [requested]
            requested_document_ids.extend(str(value) for value in requested_values)
            returned_document_ids.extend(sorted(docids(item.get("output"))))

    tool_outputs_nonempty = all(nonempty(item.get("output")) for item in calls)
    # Tool use remains entirely model-selected and may repeat in arbitrary
    # order.  Strict acceptance nevertheless requires every requested document
    # ID to be one the real search tool actually returned; this is an evidence
    # predicate, never a generation directive.
    document_returns_match = all(
        requested in returned_document_ids for requested in requested_document_ids
    )
    document_ids_referenced = all(
        requested in search_docids for requested in requested_document_ids
    )
    raw_response = payload.get("raw_response") or []
    raw_pairs_valid = True
    for index, message in enumerate(raw_response):
        if message.get("role") == "assistant" and isinstance(message.get("function_call"), dict):
            if index + 1 >= len(raw_response):
                raw_pairs_valid = False
                break
            following = raw_response[index + 1]
            raw_pairs_valid = bool(
                following.get("role") == "function"
                and following.get("name") == message["function_call"].get("name")
                and nonempty(following.get("content"))
            )
            if not raw_pairs_valid:
                break
    # Native OpenAI/vLLM history is authoritative when present.  It preserves
    # the assistant tool_calls array and each tool_call_id, so a multi-call
    # turn is checked without collapsing it into qwen-agent's legacy
    # one-function-call Message projection.
    native_history = payload.get("native_message_history")
    native_tool_pairs_valid = True
    native_tool_names_valid = True
    native_call_sequence: list[tuple[str, str]] = []
    if native_history is not None:
        if not isinstance(native_history, list):
            native_tool_pairs_valid = False
        else:
            pending_call_ids: list[str] = []
            seen_call_ids: set[str] = set()
            for message in native_history:
                if not isinstance(message, dict):
                    native_tool_pairs_valid = False
                    break
                role = message.get("role")
                if role == "assistant":
                    if pending_call_ids:
                        native_tool_pairs_valid = False
                        break
                    calls_in_message = message.get("tool_calls") or []
                    if not isinstance(calls_in_message, list):
                        native_tool_pairs_valid = False
                        break
                    for call in calls_in_message:
                        if not isinstance(call, dict):
                            native_tool_pairs_valid = False
                            break
                        call_id = str(call.get("id") or "")
                        function = call.get("function")
                        if (
                            not call_id
                            or call_id in seen_call_ids
                            or not isinstance(function, dict)
                            or not str(function.get("name") or "")
                            or "arguments" not in function
                        ):
                            native_tool_pairs_valid = False
                            break
                        seen_call_ids.add(call_id)
                        pending_call_ids.append(call_id)
                        name = str(function.get("name") or "")
                        native_call_sequence.append((name, call_id))
                        if name not in {"search-server-search", "search-server-get_document"}:
                            native_tool_names_valid = False
                    if not native_tool_pairs_valid:
                        break
                elif role == "tool":
                    call_id = str(message.get("tool_call_id") or "")
                    if not pending_call_ids or call_id != pending_call_ids.pop(0):
                        native_tool_pairs_valid = False
                        break
                    if "content" not in message:
                        native_tool_pairs_valid = False
                        break
                    if not nonempty(message.get("content")):
                        native_tool_pairs_valid = False
                        break
                elif pending_call_ids:
                    native_tool_pairs_valid = False
                    break
            native_tool_pairs_valid = native_tool_pairs_valid and not pending_call_ids
    else:
        native_tool_pairs_valid = False
    native_result_sequence = [
        ("search-server-" + str(item.get("tool_name")), "")
        for item in calls
    ]
    native_sequence_names_match = (
        len(native_call_sequence) == len(calls)
        and [name for name, _call_id in native_call_sequence]
        == [name for name, _call_id in native_result_sequence]
    )
    serialized_raw = json.dumps(raw_response, ensure_ascii=False, default=str)
    final_answer = str(payload.get("final_answer", ""))
    # DeepSeek commonly emits the required natural final sections as
    # Markdown-bold labels (for example, ``**Explanation:**``).  That is
    # still the model's final answer, not a synthetic wrapper; accept the
    # formatting while keeping the section names and colon mandatory.
    final_format_valid = all(
        re.search(
            rf"(?im)(?:^|\n)\s*(?:#+\s*)?(?:(?:\*\*|__)?{label}(?:\*\*|__)?)\s*:",
            final_answer,
        )
        for label in ("Explanation", "Exact Answer", "Confidence")
    )
    no_model_tool_markup = not any(
        marker in final_answer
        for marker in (
            "<｜DSML｜",
            "<|DSML|>",
            "<tool_call>",
            "</tool_call>",
            "<|python_tag|>",
            "<|channel|>",
            "<|recipient|>",
        )
    )
    no_double_template = (
        "QUERY_TEMPLATE" not in serialized_raw
        and "<|im_start|>system" not in serialized_raw
    )
    result_shape_valid = isinstance(result, list) and all(
        isinstance(item, dict)
        and item.get("type") in {"tool_call", "output_text", "reasoning"}
        for item in result
    )

    def observed_tool_calls_valid(tool: str) -> bool:
        matching = [item for item in calls if item.get("tool_name") == tool]
        return bool(
            int(counts.get(tool, 0)) == len(matching)
            and all(nonempty(item.get("output")) for item in matching)
        )

    metadata = payload.get("metadata") or {}
    advertised_tools = metadata.get("native_advertised_mcp_tools") or []
    advertised_by_name = {
        str(item.get("name")): item
        for item in advertised_tools
        if isinstance(item, dict)
    }
    offered_tools_schema_valid = True
    for name, aliases, required, properties in (
        ("search-server-search", {"search-server-search", "search"}, ["query"], ["query"]),
        (
            "search-server-get_document",
            {"search-server-get_document", "get_document"},
            ["docid"],
            ["docid"],
        ),
    ):
        # MCPManager exposes the real server names without the client-side
        # ``search-server-`` namespace.  Older records use the namespaced
        # form, so accept both audit representations without changing the
        # actual tool-call validation above.
        item = next((advertised_by_name.get(alias) for alias in aliases if alias in advertised_by_name), None)
        if not isinstance(item, dict):
            offered_tools_schema_valid = False
            continue
        if sorted(str(value) for value in item.get("required") or []) != required:
            offered_tools_schema_valid = False
        if sorted(str(value) for value in item.get("property_names") or []) != properties:
            offered_tools_schema_valid = False
    interaction_steps = int(payload.get("interaction_step_count", 0) or 0)
    checks = {
        "valid_json": True,
        "complete_trajectory_schema": complete_schema,
        "result_shape_valid": result_shape_valid,
        "status_completed": payload.get("status") == "completed",
        "query_id_matches": expected_query_id is None
        or str(payload.get("query_id")) == str(expected_query_id),
        "profile_matches": expected_profile is None
        or metadata.get("profile") == expected_profile,
        "retriever_matches": expected_retriever_slug is None
        or metadata.get("retriever_slug") == expected_retriever_slug,
        "real_mcp_tool_api": metadata.get("tool_api_mode") == "native_openai_real_mcp",
        "offered_tools_schema_valid": offered_tools_schema_valid,
        "explicit_trial_id": bool(str(metadata.get("trial_id") or "").strip()),
        "explicit_model_endpoint": bool(str(metadata.get("model_endpoint") or "").strip()),
        "explicit_mcp_endpoint": bool(str(metadata.get("mcp_endpoint") or "").strip()),
        "real_tool_call_present": bool(calls),
        "final_answer_nonempty": bool(str(payload.get("final_answer", "")).strip()),
        "natural_model_final": metadata.get("final_answer_source") == "model_assistant_natural"
        and metadata.get("synthetic_final") is False,
        "required_final_format": final_format_valid,
        "tool_call_count_consistent": all(
            int(counts.get(tool, 0)) == sum(item.get("tool_name") == tool for item in calls)
            for tool in set(counts) | {item.get("tool_name") for item in calls}
        ),
        "tool_arguments_valid": tool_arguments_valid,
        "tool_outputs_nonempty": tool_outputs_nonempty,
        "observed_tool_calls_valid": all(
            observed_tool_calls_valid(tool)
            for tool in {"search", "get_document"}
            if int(counts.get(tool, 0)) or any(
                item.get("tool_name") == tool for item in calls
            )
        ),
        "raw_tool_pairs_valid": raw_pairs_valid,
        "native_tool_pairs_valid": native_tool_pairs_valid,
        "native_tool_names_valid": native_tool_names_valid,
        "native_result_order_and_ids_match": native_sequence_names_match,
        "document_ids_referenced_by_search": document_ids_referenced,
        "document_returns_match": document_returns_match,
        "final_answer_last": bool(result) and result[-1].get("type") == "output_text",
        "no_model_tool_markup": no_model_tool_markup,
        "no_double_template": no_double_template,
        "interaction_steps_within_limit": 0 < interaction_steps <= MAX_INTERACTION_STEPS,
        "effective_limit_is_50": (
            int(metadata.get("max_interaction_steps", 0)) == MAX_INTERACTION_STEPS
        ),
        "output_budget_is_4096": int(metadata.get("max_output_tokens", 0)) == 4096,
    }
    max_step_evidence = max_step_termination_evidence(payload, checks)
    if max_step_evidence.get("passed"):
        termination_reason = "max_steps_limit"
    elif payload.get("status") == "completed" and str(payload.get("final_answer", "")).strip():
        termination_reason = "natural_final"
    elif interaction_steps >= MAX_INTERACTION_STEPS:
        termination_reason = "max_steps_truncation"
    elif payload.get("error_class"):
        termination_reason = "error"
    else:
        termination_reason = "incomplete"
    passed = bool(all(checks.values()) or max_step_evidence.get("passed"))
    return {
        "path": str(path),
        "passed": passed,
        "query_id": payload.get("query_id"),
        "tool_call_counts": counts,
        "interaction_step_count": payload.get("interaction_step_count"),
        "duration_seconds": payload.get("timing", {}).get("duration_seconds"),
        "termination_reason": termination_reason,
        "terminated_by": termination_reason if passed else None,
        "max_step_termination": max_step_evidence,
        "autonomous_action_diagnostics": {
            "search_count": int(counts.get("search", 0)),
            "get_document_count": int(counts.get("get_document", 0)),
            "total_tool_calls": len(calls),
            "search_executed": int(counts.get("search", 0)) > 0,
            "get_document_executed": int(counts.get("get_document", 0)) > 0,
            "search_calls_valid": observed_tool_calls_valid("search"),
            "get_document_calls_valid": observed_tool_calls_valid("get_document"),
            "search_get_document_match": document_returns_match,
            "document_ids_referenced_by_search": document_ids_referenced,
        },
        "checks": checks,
    }
