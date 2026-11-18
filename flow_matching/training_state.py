"""Strict Phase 1 training state, run metadata, and data fingerprints."""

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from common.checkpoint import CheckpointCompatibilityError


FORMAT_VERSION = 1
TRAINING_KEYS = (
    "model",
    "optim",
    "step",
    "grad_scaler",
    "rng_state",
    "run_contract",
    "resolved_config",
)
_CONFIG_RUNTIME_KEYS = {
    "resume",
    "init_from",
    "from_checkpoint",
    "log_dir",
    "train_steps",
}


def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_container(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def resolved_config_yaml(cfg):
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_yaml(cfg, resolve=True)
    return OmegaConf.to_yaml(OmegaConf.create(cfg), resolve=True)


def compatibility_config(cfg):
    resolved = _resolved_container(cfg)
    return {
        key: value
        for key, value in resolved.items()
        if key not in _CONFIG_RUNTIME_KEYS
    }


def compute_config_hash(cfg):
    return _sha256_bytes(_json_bytes(compatibility_config(cfg)))


def build_data_manifest(dataset_name, repo_root):
    """Fingerprint the actual FlowPlace dataset without hashing every shard."""
    repo_root = Path(repo_root).resolve()
    candidates = (
        repo_root / "datasets" / "graph" / dataset_name,
        repo_root / "data-gen" / "outputs" / dataset_name,
    )
    dataset_root = next((path for path in candidates if path.is_dir()), None)
    if dataset_root is None:
        searched = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"dataset {dataset_name!r} was not found; searched: {searched}"
        )

    files = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        entry = {
            "path": path.relative_to(dataset_root).as_posix(),
            "size": path.stat().st_size,
        }
        if path.name == "config.yaml":
            entry["sha256"] = sha256_file(path)
        files.append(entry)
    manifest = {
        "schema_version": 1,
        "dataset": dataset_name,
        "dataset_root": str(dataset_root),
        "fingerprint_basis": "relative path, byte size, and config.yaml SHA-256",
        "files": files,
    }
    manifest["fingerprint"] = _sha256_bytes(_json_bytes(manifest))
    return manifest


def build_in_memory_manifest(name, tensors):
    digest = hashlib.sha256()
    entries = []
    for tensor_name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        tensor_digest = hashlib.sha256(value.numpy().tobytes()).hexdigest()
        entries.append({
            "name": tensor_name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_digest,
        })
        digest.update(_json_bytes(entries[-1]))
    return {
        "schema_version": 1,
        "dataset": name,
        "dataset_root": None,
        "fingerprint_basis": "tensor name, shape, dtype, and byte content",
        "files": entries,
        "fingerprint": digest.hexdigest(),
    }


class RNGState:
    """Save all RNGs used by FlowPlace loading and iMF source/time sampling."""

    def save(self):
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    def load(self, state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        # Checkpoints loaded with map_location="cuda" also move the CPU RNG
        # state tensor. PyTorch requires RNG state tensors to live on CPU.
        torch.set_rng_state(state["torch_cpu"].cpu())
        if state.get("torch_cuda") is not None:
            if not torch.cuda.is_available():
                raise CheckpointCompatibilityError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["torch_cuda"]]
            )


class RunContract:
    """Validate family/config/data identity before restoring mutable state."""

    def __init__(self, family, config_hash, data_fingerprint):
        self.expected = {
            "format_version": FORMAT_VERSION,
            "family": family,
            "config_hash": config_hash,
            "data_fingerprint": data_fingerprint,
        }

    def save(self):
        return dict(self.expected)

    def validate(self, value):
        if value != self.expected:
            raise CheckpointCompatibilityError(
                "checkpoint run contract mismatch: "
                f"expected {self.expected}, found {value}"
            )

    def load(self, value):
        self.validate(value)


class StoredValue:
    """Serializable immutable-at-save value retained for checkpoint audit."""

    def __init__(self, value):
        self.value = value

    def save(self):
        return self.value

    def load(self, value):
        self.value = value


def register_training_state(
        checkpointer,
        model,
        optimizer,
        step,
        grad_scaler,
        family,
        config_hash,
        data_fingerprint,
        resolved_config,
):
    contract = RunContract(family, config_hash, data_fingerprint)
    checkpointer.register({
        "model": model,
        "optim": optimizer,
        "step": step,
        "grad_scaler": grad_scaler,
        "rng_state": RNGState(),
        "run_contract": contract,
        "resolved_config": StoredValue(resolved_config),
    })
    return contract


def _normalise_optional_path(value):
    if value is None or str(value).strip().lower() in ("", "none", "null"):
        return None
    return os.path.abspath(os.path.expanduser(str(value)))


def select_restore_mode(resume=None, init_from=None):
    resume = _normalise_optional_path(resume)
    init_from = _normalise_optional_path(init_from)
    if resume and init_from:
        raise ValueError("resume and init_from are mutually exclusive")
    return ("resume", resume) if resume else (
        ("init_from", init_from) if init_from else ("new", None)
    )


def resume_training(checkpointer, path, device="cpu"):
    if not checkpointer.load(
            path_override=path,
            strict=True,
            required_keys=TRAINING_KEYS,
            map_location=device,
    ):
        raise FileNotFoundError(path)


def initialize_model(checkpointer, path, family, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device)
    saved_contract = payload.get("run_contract")
    if family == "imf" and saved_contract is None:
        raise CheckpointCompatibilityError(
            "legacy FlowPlace checkpoint has no family metadata and cannot be "
            "silently loaded into IMFPlacementModel"
        )
    if saved_contract is not None and saved_contract.get("family") != family:
        raise CheckpointCompatibilityError(
            "checkpoint family mismatch: "
            f"expected {family!r}, found {saved_contract.get('family')!r}"
        )
    if not checkpointer.load(
            path_override=path,
            filter_keys=("model",),
            strict=True,
            required_keys=("model",),
            map_location=device,
    ):
        raise FileNotFoundError(path)


def restore_training_state(
        checkpointer,
        family,
        resume=None,
        init_from=None,
        legacy_from_checkpoint=None,
        device="cpu",
):
    mode, path = select_restore_mode(resume, init_from)
    legacy_path = _normalise_optional_path(legacy_from_checkpoint)
    if family == "imf" and legacy_path is not None:
        raise ValueError(
            "iMF does not use ambiguous from_checkpoint; select exactly one "
            "of resume or init_from"
        )
    if mode == "resume":
        resume_training(checkpointer, path, device=device)
    elif mode == "init_from":
        initialize_model(checkpointer, path, family=family, device=device)
    return mode, path


def _git_commit(repo_root):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_path = Path(repo_root) / ".git"
        if git_path.is_file():
            pointer = git_path.read_text(encoding="utf-8").strip()
            if not pointer.startswith("gitdir: "):
                raise RuntimeError(f"unsupported .git pointer: {pointer}")
            git_path = (Path(repo_root) / pointer[8:]).resolve()
        head = (git_path / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref_name = head[5:]
        loose_ref = git_path / ref_name
        if loose_ref.exists():
            return loose_ref.read_text(encoding="utf-8").strip()
        packed_refs = git_path / "packed-refs"
        if packed_refs.exists():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    commit, name = line.split(" ", 1)
                    if name == ref_name:
                        return commit
        raise RuntimeError(f"cannot resolve Git ref {ref_name!r}")


def _environment_text():
    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except ImportError:
        pyg_version = "unavailable"
    return "\n".join((
        f"python={sys.version.replace(os.linesep, ' ')}",
        f"platform={platform.platform()}",
        f"torch={torch.__version__}",
        f"torch_geometric={pyg_version}",
        f"cuda_available={torch.cuda.is_available()}",
        f"cuda_runtime={torch.version.cuda}",
        "",
    ))


def _write_json(path, value):
    with open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def initialize_run_artifacts(run_dir, cfg, data_manifest, repo_root):
    run_dir = Path(run_dir).resolve()
    repo_root = Path(repo_root).resolve()
    if run_dir == repo_root or repo_root in run_dir.parents:
        raise ValueError("runtime outputs must be outside the Git source repository")
    run_dir.mkdir(parents=True, exist_ok=True)

    family = str(_resolved_container(cfg)["family"])
    seed = int(_resolved_container(cfg)["seed"])
    cfg_hash = compute_config_hash(cfg)
    git_commit = _git_commit(repo_root)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(resolved_config_yaml(cfg))
    with open(run_dir / "git_commit.txt", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(git_commit + "\n")
    with open(run_dir / "environment.txt", "w", encoding="utf-8", newline="\n") as stream:
        stream.write(_environment_text())
    _write_json(run_dir / "data_manifest.json", data_manifest)
    if not (run_dir / "metrics.json").exists():
        _write_json(run_dir / "metrics.json", {})
    for log_name in ("stdout.log", "stderr.log"):
        (run_dir / log_name).touch(exist_ok=True)
    metadata = {
        "schema_version": 1,
        "family": family,
        "seed": seed,
        "git_commit": git_commit,
        "config_hash": cfg_hash,
        "data_fingerprint": data_manifest["fingerprint"],
        "checkpoint": None,
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    return metadata


def record_restore(run_dir, mode, path):
    metadata_path = Path(run_dir) / "run_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata["restore"] = {
        "mode": mode,
        "path": path,
        "sha256": sha256_file(path) if path is not None else None,
    }
    _write_json(metadata_path, metadata)


def save_training_checkpoint(checkpointer, checkpoint_path, run_dir):
    checkpoint_path = os.path.abspath(checkpoint_path)
    checkpointer.save(path_override=checkpoint_path, strict=True)
    checkpoint_record = {
        "path": checkpoint_path,
        "sha256": sha256_file(checkpoint_path),
    }
    metadata_path = Path(run_dir) / "run_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    metadata["checkpoint"] = checkpoint_record
    _write_json(metadata_path, metadata)
    return checkpoint_record


def write_metrics(run_dir, metrics):
    _write_json(Path(run_dir) / "metrics.json", metrics)
