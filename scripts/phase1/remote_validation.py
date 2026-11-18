#!/usr/bin/env python3
"""Phase 1 real-data smoke, tiny training, and raw evaluation diagnostics."""

import argparse
import copy
import json
import pathlib
import pickle
import re
import sys
import time

try:
    import resource
except ImportError:  # pragma: no cover - Linux PBS jobs always provide resource.
    resource = None

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from torch_geometric.data import Data


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FLOW_MATCHING_ROOT = REPO_ROOT / "flow_matching"
CONFIG_ROOT = FLOW_MATCHING_ROOT / "configs"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FLOW_MATCHING_ROOT))

import common  # noqa: E402
import evaluation_metrics  # noqa: E402
import guidance  # noqa: E402
import models  # noqa: E402
import training_state  # noqa: E402


def write_json(path, value):
    path = pathlib.Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def compose_config(args, run_dir):
    with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = compose(
            config_name="config_combined",
            overrides=["+experiment=imf_raw"],
        )
    with open_dict(cfg):
        cfg.task = args.task
        cfg.test_task = args.task
        cfg.seed = args.seed
        cfg.method = "phase1_prompt7"
        cfg.log_dir = str(run_dir)
        cfg.batch_size = args.batch_size
        cfg.val_batch_size = 1
        cfg.train_steps = args.steps
        cfg.print_every = max(args.steps, 1)
        cfg.eval_every = 0
        cfg.test_every = 0
        cfg.eval_samples = 0
        cfg.test_num_output_samples = 0
        cfg.train_data_limit = args.train_samples
        cfg.val_data_limit = args.val_samples
        cfg.logger.wandb = False
        cfg.model.imf_sampling_steps = 1
        cfg.model.use_hard_constraint_in_sampling = False
        cfg.legalization.mode = "none"
    if cfg.family != "imf":
        raise RuntimeError(f"expected family=imf, found {cfg.family!r}")
    if cfg.legalization.mode != "none":
        raise RuntimeError("raw Phase 1 validation requires legalization.mode=none")
    if cfg.model.use_hard_constraint_in_sampling:
        raise RuntimeError("raw Phase 1 validation forbids hard-constraint sampling")
    return cfg


class BoundedGraphLoader:
    """Match FlowPlace's shared-graph batch convention for bounded diagnostics."""

    def __init__(self, train_set, val_set, train_batch_size, val_batch_size, device):
        self.datasets = {"train": train_set, "val": val_set}
        self.batch_sizes = {"train": train_batch_size, "val": val_batch_size}
        self.device = device

    def get_batch(self, split):
        dataset = self.datasets[split]
        index = torch.randint(0, len(dataset), (1,)).item()
        target, condition = dataset[index]
        target = target.to(self.device).view(1, *target.shape)
        target = target.expand(self.batch_sizes[split], *target.shape[1:])
        return target, condition.to(self.device), index


def load_release_synthetic_data(task, train_limit, val_limit):
    """Load a bounded prefix using the verified FlowPlace release contract."""
    dataset_root = REPO_ROOT / "data-gen" / "outputs" / task
    config_path = dataset_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    dataset_config = OmegaConf.load(config_path)
    if "num_train_samples" not in dataset_config or "num_val_samples" not in dataset_config:
        raise ValueError("Prompt 7 requires a concrete synthetic dataset config")
    if train_limit > dataset_config.num_train_samples:
        raise ValueError("train sample limit exceeds dataset config")
    if val_limit > dataset_config.num_val_samples:
        raise ValueError("validation sample limit exceeds dataset config")
    shards = []
    for path in dataset_root.rglob("*.pickle"):
        match = re.search(r"\d+", path.name)
        if match:
            shards.append((int(match.group()), path))
    shards.sort(key=lambda item: item[0])
    if not shards:
        raise RuntimeError(f"no pickle shards found in {dataset_root}")
    bounded = []
    wanted = train_limit + val_limit
    for _, path in shards:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        for sample in payload:
            if not isinstance(sample, (tuple, list)) or len(sample) != 2:
                raise TypeError(f"invalid sample in {path}: expected (x, condition)")
            target, condition = sample
            if not isinstance(condition, Data):
                raise TypeError(f"invalid condition in {path}: expected PyG Data")
            target = torch.as_tensor(target, dtype=torch.float32)
            if target.ndim != 2 or target.shape[-1] != 2:
                raise ValueError(f"invalid placement shape in {path}: {tuple(target.shape)}")
            if condition.edge_index.shape[1] % 2 or condition.edge_attr.shape[0] % 2:
                raise ValueError(f"graph edges are not paired in {path}")
            for key in ("is_macros", "is_ports"):
                if key in condition and condition[key].ndim > 1:
                    condition[key] = condition[key].flatten().bool()
            bounded.append((target, condition))
            if len(bounded) == wanted:
                return bounded[:train_limit], bounded[train_limit:]
    raise RuntimeError(f"dataset yielded {len(bounded)} samples; required {wanted}")


def load_bounded_data(cfg, device):
    train_set, val_set = load_release_synthetic_data(
        cfg.task, cfg.train_data_limit, cfg.val_data_limit
    )
    if not train_set:
        raise RuntimeError("bounded training set is empty")
    if cfg.val_data_limit and not val_set:
        raise RuntimeError("bounded validation set is empty")
    loader = BoundedGraphLoader(
        train_set,
        val_set,
        cfg.batch_size,
        cfg.val_batch_size,
        device,
    )
    return train_set, val_set, loader


def build_model(cfg, sample_shape, device):
    with open_dict(cfg):
        cfg.model.update({
            "num_classes": cfg.num_classes,
            "input_shape": tuple(sample_shape),
            "device": str(device),
        })
    model = models.get_model_class(cfg.family)(**cfg.model).to(device)
    return model


def initialize_artifacts(cfg, run_dir):
    manifest = training_state.build_data_manifest(cfg.task, REPO_ROOT)
    metadata = training_state.initialize_run_artifacts(
        run_dir, cfg, manifest, REPO_ROOT
    )
    return manifest, metadata


def new_training_state(cfg, sample_shape, device, metadata, checkpoint_path):
    model = build_model(cfg, sample_shape, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step = common.Counter()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpointer = common.Checkpointer(str(checkpoint_path))
    training_state.register_training_state(
        checkpointer,
        model,
        optimizer,
        step,
        scaler,
        cfg.family,
        metadata["config_hash"],
        metadata["data_fingerprint"],
        training_state.resolved_config_yaml(cfg),
    )
    return model, optimizer, step, scaler, checkpointer


def fixed_condition(model, target, condition):
    result = copy.copy(condition)
    mask = model._fixed_mask(target, condition)
    if mask is not None:
        result[model.fixed_xy_key] = torch.where(
            mask, target, torch.zeros_like(target)
        )
    return result


def gradient_metrics(model):
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite = bool(gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gradients
    )
    nonzero = any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
    norm = (
        torch.sqrt(sum(
            gradient.detach().float().square().sum() for gradient in gradients
        )).item()
        if gradients else 0.0
    )
    return {
        "finite": finite,
        "nonzero": nonzero,
        "global_l2_norm": norm,
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_metrics(device):
    result = {
        "max_rss_kib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if resource is not None else None
        ),
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_max_allocated_bytes": None,
        "cuda_max_reserved_bytes": None,
    }
    if device.type == "cuda":
        result.update({
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        })
    return result


def raw_overlap_metrics(position, sizes, fixed_mask):
    movable = ~fixed_mask.bool().view(-1)
    position = position[movable]
    sizes = sizes[movable]
    count = position.shape[0]
    possible_pairs = count * (count - 1) // 2
    if possible_pairs == 0:
        return {
            "raw_overlap_pair_count": 0,
            "raw_overlap_pair_fraction": 0.0,
            "raw_overlap_area_sum": 0.0,
        }
    lower = position - sizes / 2.0
    upper = position + sizes / 2.0
    intersection = torch.minimum(upper[:, None, :], upper[None, :, :]) - torch.maximum(
        lower[:, None, :], lower[None, :, :]
    )
    overlap_area = intersection.clamp_min(0).prod(dim=-1)
    upper_triangle = torch.triu(
        torch.ones_like(overlap_area, dtype=torch.bool), diagonal=1
    )
    pair_areas = overlap_area[upper_triangle]
    pair_count = torch.count_nonzero(pair_areas > 0).item()
    return {
        "raw_overlap_pair_count": pair_count,
        "raw_overlap_pair_fraction": pair_count / possible_pairs,
        "raw_overlap_area_sum": pair_areas.sum().item(),
    }


def normalized_hpwl(position, condition):
    pin_map, pin_offsets, pin_edge_index = guidance.compute_pin_map(condition)
    hpwl = guidance.HPWL().to(position.device)(
        position.unsqueeze(0), pin_map, pin_offsets, pin_edge_index
    )
    return hpwl.item()


def smoke(args, cfg, run_dir, device):
    train_set, _, loader = load_bounded_data(cfg, device)
    _, metadata = initialize_artifacts(cfg, run_dir)
    checkpoint_path = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=False)
    model, optimizer, step, _, checkpointer = new_training_state(
        cfg, train_set[0][0].shape, device, metadata, checkpoint_path
    )
    target, condition, _ = loader.get_batch("train")
    condition_for_model = fixed_condition(model, target, condition)
    batch_size = target.shape[0]
    source = 2.0 * torch.rand_like(target) - 1.0
    mask = model._fixed_mask(target, condition_for_model)
    if mask is not None:
        source = torch.where(mask, target, source)
    r = torch.full((batch_size,), 0.25, device=device, dtype=torch.float32)
    t = torch.full((batch_size,), 0.75, device=device, dtype=torch.float32)
    boundary = model(source, t, t, condition_for_model)
    forward = model(source, r, t, condition_for_model)
    jvp_primal, jvp_value, jvp_backend = model.jvp(
        source, r, t, condition_for_model
    )
    optimizer.zero_grad(set_to_none=True)
    loss, details = model.training_objective(target, condition)
    loss.backward()
    gradients = gradient_metrics(model)
    if not torch.isfinite(loss) or not gradients["finite"] or not gradients["nonzero"]:
        raise RuntimeError("smoke loss/gradient check failed")
    optimizer.step()
    step.increment()
    sample_one, _ = model.reverse_samples(
        batch_size, target, condition, source=source, num_timesteps=1
    )
    nfe_one = model.last_sampling_nfe
    sample_two, _ = model.reverse_samples(
        batch_size, target, condition, source=source, num_timesteps=2
    )
    nfe_two = model.last_sampling_nfe
    checkpoint = training_state.save_training_checkpoint(
        checkpointer, str(checkpoint_path), run_dir
    )
    reloaded = new_training_state(
        cfg, train_set[0][0].shape, device, metadata, checkpoint_path
    )
    training_state.resume_training(reloaded[-1], str(checkpoint_path), device=device)
    reload_exact = all(
        torch.equal(value, reloaded[0].state_dict()[key])
        for key, value in model.state_dict().items()
    )
    checks = {
        "data_load": True,
        "imf_forward_finite": torch.isfinite(forward).all().item(),
        "boundary_forward_finite": torch.isfinite(boundary).all().item(),
        "real_backbone_jvp_finite": torch.isfinite(jvp_value).all().item(),
        "jvp_primal_finite": torch.isfinite(jvp_primal).all().item(),
        "jvp_backend": jvp_backend,
        "loss_finite": torch.isfinite(loss).item(),
        "loss": loss.detach().item(),
        "loss_backward": True,
        "gradients": gradients,
        "one_step_sample_finite": torch.isfinite(sample_one).all().item(),
        "one_step_nfe": nfe_one,
        "two_step_sample_finite": torch.isfinite(sample_two).all().item(),
        "two_step_nfe": nfe_two,
        "checkpoint": checkpoint,
        "checkpoint_reload_exact": reload_exact,
        "nodes": target.shape[1],
        "batch_size": batch_size,
        "jvp_norm": details["total_derivative"].norm().item(),
    }
    required = [value for key, value in checks.items() if key.endswith("_finite")]
    if not all(required) or not reload_exact or nfe_one != 1 or nfe_two != 2:
        raise RuntimeError("one or more smoke checks failed")
    training_state.write_metrics(run_dir, checks)
    print(json.dumps(checks, indent=2, sort_keys=True))
    print("PHASE1_REAL_DATA_SMOKE_COMPLETE")


def tiny_train(args, cfg, run_dir, device):
    train_set, _, loader = load_bounded_data(cfg, device)
    _, metadata = initialize_artifacts(cfg, run_dir)
    checkpoint_path = run_dir / "checkpoints" / "latest.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=False)
    model, optimizer, step, _, checkpointer = new_training_state(
        cfg, train_set[0][0].shape, device, metadata, checkpoint_path
    )
    model.train()
    losses = []
    step_times = []
    run_started = time.perf_counter()
    for _ in range(args.steps):
        target, condition, _ = loader.get_batch("train")
        synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss, details = model.training_objective(target, condition)
        if not torch.isfinite(loss):
            raise RuntimeError("tiny training produced non-finite loss")
        loss.backward()
        gradients = gradient_metrics(model)
        if not gradients["finite"] or not gradients["nonzero"]:
            raise RuntimeError("tiny training produced invalid gradients")
        optimizer.step()
        step.increment()
        synchronize(device)
        step_times.append(time.perf_counter() - step_started)
        losses.append({
            "step": int(step),
            "loss": loss.detach().item(),
            "jvp_norm": details["total_derivative"].norm().item(),
            "gradient_l2_norm": gradients["global_l2_norm"],
        })
    total_seconds = time.perf_counter() - run_started
    model.eval()
    raw_target, raw_condition, _ = loader.get_batch("val")
    raw_sample, _ = model.reverse_samples(
        raw_target.shape[0], raw_target, raw_condition, num_timesteps=1
    )
    raw_sample_path = run_dir / "raw_sample.pt"
    torch.save({
        "sample": raw_sample.detach().cpu(),
        "target": raw_target.detach().cpu(),
        "legalization_mode": "none",
        "nfe": model.last_sampling_nfe,
    }, raw_sample_path)
    checkpoint = training_state.save_training_checkpoint(
        checkpointer, str(checkpoint_path), run_dir
    )
    timing = {
        "total_train_seconds": total_seconds,
        "step_seconds": step_times,
        "mean_step_seconds": sum(step_times) / len(step_times),
    }
    memory = memory_metrics(device)
    write_json(run_dir / "losses.json", losses)
    write_json(run_dir / "timing.json", timing)
    write_json(run_dir / "memory.json", memory)
    metrics = {
        "diagnostic_only": True,
        "steps": int(step),
        "losses_file": str(run_dir / "losses.json"),
        "loss_finite": all(torch.isfinite(torch.tensor(item["loss"])).item() for item in losses),
        "raw_sample": str(raw_sample_path),
        "raw_sample_finite": torch.isfinite(raw_sample).all().item(),
        "raw_sample_nfe": model.last_sampling_nfe,
        "timing_file": str(run_dir / "timing.json"),
        "memory_file": str(run_dir / "memory.json"),
        "checkpoint": checkpoint,
    }
    training_state.write_metrics(run_dir, metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("PHASE1_TINY_TRAIN_COMPLETE")


def raw_eval(args, cfg, run_dir, device):
    _, val_set, _ = load_bounded_data(cfg, device)
    if args.val_index >= len(val_set):
        raise IndexError(
            f"val-index {args.val_index} is outside the bounded validation set "
            f"of size {len(val_set)}"
        )
    _, metadata = initialize_artifacts(cfg, run_dir)
    checkpoint_path = pathlib.Path(args.checkpoint).resolve()
    model = build_model(cfg, val_set[args.val_index][0].shape, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step = common.Counter()
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    checkpointer = common.Checkpointer(str(checkpoint_path))
    training_state.register_training_state(
        checkpointer,
        model,
        optimizer,
        step,
        scaler,
        cfg.family,
        metadata["config_hash"],
        metadata["data_fingerprint"],
        training_state.resolved_config_yaml(cfg),
    )
    training_state.initialize_model(
        checkpointer, str(checkpoint_path), family="imf", device=device
    )
    training_state.record_restore(run_dir, "init_from", str(checkpoint_path))
    model.eval()
    target, condition = val_set[args.val_index]
    target = target.to(device).unsqueeze(0)
    condition = condition.to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    source = 2.0 * torch.rand(
        target.shape, device=device, dtype=torch.float32, generator=generator
    ) - 1.0
    mask = model._fixed_mask(target, condition)
    if mask is not None:
        source = torch.where(mask, target, source)
    results = {}
    samples = {}
    disallowed_calls = {"jvp": 0, "legalizer": 0}
    backbone_forward_calls = 0

    def count_backbone_forward(_module, _inputs):
        nonlocal backbone_forward_calls
        backbone_forward_calls += 1

    def reject_disallowed_call(frame, event, _arg):
        if event != "call":
            return
        filename = pathlib.Path(frame.f_code.co_filename).name
        if filename == "models.py" and frame.f_code.co_name == "jvp":
            disallowed_calls["jvp"] += 1
            raise RuntimeError("raw iMF inference attempted to call JVP")
        if filename in ("legalization.py", "hard_constraint_legalizer.py"):
            disallowed_calls["legalizer"] += 1
            raise RuntimeError("raw iMF inference attempted to call a legalizer")

    forward_hook = model._reverse_model.register_forward_pre_hook(
        count_backbone_forward
    )
    for sample_steps in (1, 2):
        synchronize(device)
        started = time.perf_counter()
        forwards_before = backbone_forward_calls
        sys.setprofile(reject_disallowed_call)
        try:
            sample, _ = model.reverse_samples(
                1,
                target,
                condition,
                source=source,
                num_timesteps=sample_steps,
            )
        finally:
            sys.setprofile(None)
        synchronize(device)
        runtime = time.perf_counter() - started
        forward_calls = backbone_forward_calls - forwards_before
        sample_2d = sample[0]
        fixed_mask = condition.is_ports if "is_ports" in condition else torch.zeros(
            sample_2d.shape[0], dtype=torch.bool, device=device
        )
        hpwl = normalized_hpwl(sample_2d, condition)
        boundary = evaluation_metrics.boundary_violation_metrics(
            sample_2d, condition.x
        )
        overlap = raw_overlap_metrics(sample_2d, condition.x, fixed_mask)
        raw_legal = (
            boundary["raw_boundary_violation_max"] == 0.0
            and overlap["raw_overlap_pair_count"] == 0
        )
        fixed_mask_full = model._fixed_mask(sample, condition)
        fixed_count = 0
        fixed_nodes_ok = True
        fixed_node_max_abs_error = 0.0
        if fixed_mask_full is not None and torch.any(fixed_mask_full):
            fixed_count = int(fixed_mask_full[..., 0].sum().item())
            if model.fixed_xy_key in condition:
                fixed_reference = model._fixed_xy(
                    sample, condition, fixed_mask_full
                )
            else:
                fixed_reference = target.expand_as(sample)
            fixed_actual = sample.masked_select(fixed_mask_full)
            fixed_expected = fixed_reference.masked_select(fixed_mask_full)
            fixed_nodes_ok = torch.equal(fixed_actual, fixed_expected)
            fixed_node_max_abs_error = (
                fixed_actual - fixed_expected
            ).abs().max().item()
        nan_count = int(torch.isnan(sample).sum().item())
        inf_count = int(torch.isinf(sample).sum().item())
        half_sizes = condition.x / 2.0
        boundary_violation_count = int((
            ((sample_2d - half_sizes) < -1.0)
            | ((sample_2d + half_sizes) > 1.0)
        ).any(dim=-1).sum().item())
        results[f"{sample_steps}_step"] = {
            "legalization_mode": "none",
            "hard_constraint_sampling": False,
            "nfe": model.last_sampling_nfe,
            "backbone_forward_calls": forward_calls,
            "time_grid": [1.0, 0.0] if sample_steps == 1 else [1.0, 0.5, 0.0],
            "runtime_seconds": runtime,
            "output_shape": list(sample.shape),
            "x_min": sample_2d[:, 0].min().item(),
            "x_max": sample_2d[:, 0].max().item(),
            "y_min": sample_2d[:, 1].min().item(),
            "y_max": sample_2d[:, 1].max().item(),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "boundary_violation_count": boundary_violation_count,
            "fixed_node_count": fixed_count,
            "fixed_nodes_ok": fixed_nodes_ok,
            "fixed_node_max_abs_error": fixed_node_max_abs_error,
            "hpwl_normalized": hpwl,
            "raw_legality_binary": raw_legal,
            "output_finite": torch.isfinite(sample).all().item(),
            **boundary,
            **overlap,
        }
        samples[f"sample_{sample_steps}_step"] = sample.detach().cpu()
    forward_hook.remove()
    sample_path = run_dir / args.raw_output_name
    torch.save({
        **samples,
        "target": target.detach().cpu(),
        "source": source.detach().cpu(),
        "legalization_mode": "none",
    }, sample_path)
    results["checkpoint"] = str(checkpoint_path)
    results["checkpoint_sha256"] = training_state.sha256_file(checkpoint_path)
    results["git_sha"] = training_state._git_commit(REPO_ROOT)
    results["problem"] = f"{args.task}/bounded_validation_{args.val_index}"
    results["seed"] = args.seed
    results["jvp_calls"] = disallowed_calls["jvp"]
    results["legalizer_calls"] = disallowed_calls["legalizer"]
    results["raw_samples"] = str(sample_path)
    results["memory"] = memory_metrics(device)
    if results["1_step"]["nfe"] != 1 or results["2_step"]["nfe"] != 2:
        raise RuntimeError("raw evaluation NFE mismatch")
    if not results["1_step"]["output_finite"] or not results["2_step"]["output_finite"]:
        raise RuntimeError("raw evaluation produced non-finite output")
    for key, expected_calls in (("1_step", 1), ("2_step", 2)):
        if results[key]["backbone_forward_calls"] != expected_calls:
            raise RuntimeError(f"{key} model call count mismatch")
        if not results[key]["fixed_nodes_ok"]:
            raise RuntimeError(f"{key} moved fixed nodes")
        if results[key]["nan_count"] or results[key]["inf_count"]:
            raise RuntimeError(f"{key} produced NaN or Inf")
    if results["jvp_calls"] or results["legalizer_calls"]:
        raise RuntimeError("raw evaluation called a forbidden operation")
    training_state.write_metrics(run_dir, results)
    run_info = [
        f"git_sha={results['git_sha']}",
        f"checkpoint_path={results['checkpoint']}",
        f"checkpoint_sha256={results['checkpoint_sha256']}",
        f"problem={results['problem']}",
        f"seed={results['seed']}",
        f"raw_output={sample_path}",
        f"jvp_calls={results['jvp_calls']}",
        f"legalizer_calls={results['legalizer_calls']}",
    ]
    for key in ("1_step", "2_step"):
        item = results[key]
        run_info.extend([
            f"{key}.nfe={item['nfe']}",
            f"{key}.time_grid={item['time_grid']}",
            f"{key}.output_shape={item['output_shape']}",
            f"{key}.runtime_seconds={item['runtime_seconds']}",
            f"{key}.nan_count={item['nan_count']}",
            f"{key}.inf_count={item['inf_count']}",
            f"{key}.fixed_nodes_ok={item['fixed_nodes_ok']}",
            f"{key}.backbone_forward_calls={item['backbone_forward_calls']}",
        ])
    run_info_path = run_dir / "run_info.txt"
    if run_info_path.exists():
        raise FileExistsError(f"refusing to overwrite {run_info_path}")
    run_info_path.write_text("\n".join(run_info) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("PHASE1_RAW_EVAL_COMPLETE")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "tiny-train", "raw-eval"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task", default="mask.61")
    parser.add_argument("--checkpoint")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=2)
    parser.add_argument("--val-samples", type=int, default=1)
    parser.add_argument("--val-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=61)
    parser.add_argument("--raw-output-name", default="raw_samples.pt")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("steps and batch-size must be positive")
    if args.val_index < 0 or args.val_index >= args.val_samples:
        parser.error("val-index must be non-negative and less than val-samples")
    if args.mode == "raw-eval" and not args.checkpoint:
        parser.error("raw-eval requires --checkpoint")
    if pathlib.Path(args.raw_output_name).name != args.raw_output_name:
        parser.error("raw-output-name must be a filename")
    return args


def main():
    args = parse_args()
    run_dir = pathlib.Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    cfg = compose_config(args, run_dir)
    print(f"mode={args.mode}")
    print(f"device={device}")
    print(f"run_dir={run_dir}")
    if args.mode == "smoke":
        smoke(args, cfg, run_dir, device)
    elif args.mode == "tiny-train":
        tiny_train(args, cfg, run_dir, device)
    else:
        raw_eval(args, cfg, run_dir, device)


if __name__ == "__main__":
    main()
