#!/usr/bin/env python3
"""Strictly continue a saved iMF G/P training state to a larger target step."""

import argparse
import hashlib
import json
import pathlib
import sys
import time

import torch
from omegaconf import OmegaConf, open_dict

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "flow_matching"), str(ROOT / "scripts/phase1"), str(ROOT)]
import remote_validation as phase1  # noqa: E402
import utils as flow_utils  # noqa: E402
from graph_batch import accumulate_graphs  # noqa: E402


def write_json(path, value):
    path = pathlib.Path(path)
    temporary = pathlib.Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--expected-start-step", type=int, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--graph-batch", type=int, required=True)
    parser.add_argument("--path-batch", type=int, required=True)
    parser.add_argument("--checkpoint-steps", required=True)
    parser.add_argument("--seed", type=int, default=61)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    resume_path = pathlib.Path(args.resume).resolve()
    config_path = pathlib.Path(args.training_config).resolve()
    checkpoint_steps = tuple(int(x) for x in args.checkpoint_steps.replace(":", ",").split(",") if x)
    if not run_dir.is_dir() or (run_dir / "resolved_config.yaml").exists():
        raise RuntimeError("run directory is missing or already used")
    if not resume_path.is_file() or not config_path.is_file():
        raise FileNotFoundError((resume_path, config_path))
    if not checkpoint_steps or tuple(sorted(set(checkpoint_steps))) != checkpoint_steps:
        raise ValueError("checkpoint steps must be nonempty, unique, and increasing")
    if checkpoint_steps[-1] != args.target_step:
        raise ValueError("the final checkpoint must equal target-step")
    if checkpoint_steps[0] <= args.expected_start_step:
        raise ValueError("all checkpoints must be after the resume step")
    if (args.graph_batch, args.path_batch) not in ((1, 2), (1, 16), (4, 4)):
        raise ValueError("this continuation runner supports only G1/P2, G1/P16, and G4/P4")
    if args.expected_start_step >= args.target_step:
        raise ValueError("target-step must be larger than start step")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a PBS GPU allocation")

    cfg = OmegaConf.load(config_path)
    if cfg.family != "imf" or cfg.task != "mask.61" or int(cfg.seed) != args.seed:
        raise RuntimeError("source configuration identity mismatch")
    if int(cfg.batch_size) != args.path_batch:
        raise RuntimeError(f"source batch_size={cfg.batch_size}, expected P={args.path_batch}")
    if "graph_batch" in cfg and int(cfg.graph_batch) != args.graph_batch:
        raise RuntimeError("source graph_batch mismatch")
    if "path_batch" in cfg and int(cfg.path_batch) != args.path_batch:
        raise RuntimeError("source path_batch mismatch")
    batch_mode = "legacy" if args.path_batch == 2 else "accumulate"
    if "batch_mode" in cfg and str(cfg.batch_mode) != batch_mode:
        raise RuntimeError("source batch_mode mismatch")
    with open_dict(cfg):
        cfg.log_dir = str(run_dir)
        cfg.train_steps = args.target_step
        cfg.resume = str(resume_path)
        cfg.init_from = "none"
        cfg.from_checkpoint = "none"
        cfg.eval_every = 0
        cfg.test_every = 0

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    train_set, val_set = flow_utils.load_graph_data(
        cfg.task,
        augment=cfg.augment,
        train_data_limit=cfg.train_data_limit,
        val_data_limit=cfg.val_data_limit,
    )
    if (len(train_set), len(val_set)) != (40000, 2000):
        raise RuntimeError(f"unexpected split: {len(train_set)}/{len(val_set)}")
    loader = flow_utils.GraphDataLoader(
        train_set, val_set, args.path_batch, cfg.val_batch_size, train_device=device
    )
    _, metadata = phase1.initialize_artifacts(cfg, run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    latest_path = checkpoint_dir / "latest.ckpt"
    model, optimizer, step, _, checkpointer = phase1.new_training_state(
        cfg, train_set[0][0].shape, device, metadata, latest_path
    )
    phase1.training_state.resume_training(checkpointer, str(resume_path), device=device)
    phase1.training_state.record_restore(run_dir, "resume", str(resume_path))
    start_step = int(step)
    if start_step != args.expected_start_step:
        raise RuntimeError(f"expected start {args.expected_start_step}, found {start_step}")
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not learning_rates or any(lr != float(cfg.lr) for lr in learning_rates):
        raise RuntimeError(f"restored learning rates {learning_rates} do not match {cfg.lr}")

    resume_record = {
        "event": "resume_confirmed",
        "resume_checkpoint": str(resume_path),
        "resume_sha256": sha256(resume_path),
        "source_config": str(config_path),
        "source_config_sha256": sha256(config_path),
        "start_step": start_step,
        "target_step": args.target_step,
        "checkpoint_steps": list(checkpoint_steps),
        "graph_batch": args.graph_batch,
        "path_batch": args.path_batch,
        "batch_mode": batch_mode,
        "learning_rate": learning_rates[0],
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
    }
    write_json(run_dir / "resume_state.json", resume_record)
    print(json.dumps(resume_record, sort_keys=True), flush=True)

    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    interval_started = started
    loss_window = []
    final_loss = None
    with (run_dir / "training_history.jsonl").open("x", buffering=1) as history:
        while int(step) < args.target_step:
            record = accumulate_graphs(
                model, optimizer, loader, graph_batch=args.graph_batch, mode=batch_mode
            )
            optimizer.step()
            step.increment()
            current_step = int(step)
            final_loss = float(record["train_loss"])
            loss_window.append(final_loss)
            if current_step % 100 == 0 or current_step in checkpoint_steps:
                torch.cuda.synchronize()
                now = time.perf_counter()
                progress = {
                    **record,
                    "step": current_step,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "loss_mean_last_100": sum(loss_window[-100:]) / len(loss_window[-100:]),
                    "seconds_since_last_log": now - interval_started,
                    "wall_seconds": now - started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                }
                if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                    raise FloatingPointError("non-finite model parameter")
                history.write(json.dumps(progress) + "\n")
                write_json(run_dir / "training_progress.json", progress)
                print(json.dumps(progress), flush=True)
                interval_started = now
            if current_step in checkpoint_steps:
                phase1.training_state.save_training_checkpoint(
                    checkpointer, str(checkpoint_dir / f"step_{current_step}.ckpt"), run_dir
                )
                phase1.training_state.save_training_checkpoint(
                    checkpointer, str(latest_path), run_dir
                )

    torch.cuda.synchronize()
    metrics = {
        **resume_record,
        "event": "training_complete",
        "final_step": int(step),
        "final_train_loss": final_loss,
        "loss_mean_last_100": sum(loss_window[-100:]) / len(loss_window[-100:]),
        "wall_seconds": time.perf_counter() - started,
        "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "nonfinite_detected": False,
        "legalizer_called": False,
        "sampling_called": False,
        "latest_checkpoint": str(latest_path),
    }
    for checkpoint_step in checkpoint_steps:
        checkpoint_path = checkpoint_dir / f"step_{checkpoint_step}.ckpt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"missing checkpoint {checkpoint_path}")
        metrics[f"step_{checkpoint_step}_checkpoint"] = str(checkpoint_path)
    if not latest_path.is_file():
        raise RuntimeError(f"missing checkpoint {latest_path}")
    write_json(run_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print("IMF_GP_CONTINUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
