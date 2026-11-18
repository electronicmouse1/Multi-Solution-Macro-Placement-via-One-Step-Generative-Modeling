"""Opt-in graph accumulation; the legacy GraphDataLoader is unchanged."""

import math

import torch


def gradient_stats(model):
    gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not gradients:
        raise RuntimeError("no gradients")
    norm = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g.float()) for g in gradients
    ])).item()
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    return norm


def accumulate_graphs(model, optimizer, loader, graph_batch=1, mode="accumulate"):
    """Mean of separately normalized graph losses, one optimizer step outside.

    Graph selection retains legacy replacement sampling across updates. Within
    an update, duplicate indices are rejected so G means G distinct graphs.
    For G=1 no extra RNG draw is made. No source/time sampler is changed.
    """
    if graph_batch < 1 or graph_batch > len(loader.train_set):
        raise ValueError("graph_batch outside dataset size")
    if mode not in ("legacy", "accumulate") or (mode == "legacy" and graph_batch != 1):
        raise ValueError("legacy mode requires graph_batch=1")
    optimizer.zero_grad(set_to_none=True)
    indices, losses, backends = [], [], []
    rejected = 0
    while len(indices) < graph_batch:
        target, condition, index = loader.get_batch("train")
        if index in indices:
            condition.to("cpu")
            rejected += 1
            continue
        try:
            loss, details = model.training_objective(target, condition)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("non-finite graph loss")
            if mode == "legacy":
                loss.backward()
            else:
                (loss / graph_batch).backward()
            losses.append(float(loss.detach().item()))
            indices.append(index)
            backends.append(str(details.get("jvp_backend", "unreported")))
            del loss, details, target
        finally:
            condition.to("cpu")
    return {
        "train_loss": sum(losses) / graph_batch,
        "graph_losses": losses,
        "graph_indices": indices,
        "duplicate_draws_rejected": rejected,
        "grad_norm": gradient_stats(model),
        "jvp_backends": backends,
    }
