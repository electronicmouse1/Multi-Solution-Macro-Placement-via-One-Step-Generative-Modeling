"""Optional terminal macro legalization using the installed DREAMPlace operator.

B calls the official global MacroLegalize; C repairs a bounded conflict
neighbourhood with all other macros
as fixed obstacles. The cascade retries B from the original Y when C fails.
No model or legacy soft/hard policy is changed. Every operator call runs in its
own scratch directory and subprocess because the native LP code may abort.

Metadata (JSON file or mapping) is explicit, never inferred from a design name:
  bounds_dbu: [xl, yl, xh, yh] of the normalized canvas / rectangular ROW region
  node_sizes_dbu: N x 2 oriented macro dimensions in graph order
  node_roles: N entries, each movable, fixed, or port
  site_width_dbu, row_height_dbu: positive integer physical grid pitches
  fixed_ll_dbu: N x 2 physical lower-left positions, required if any role=fixed
Optional provenance fields include design, dbu_per_micron and node_names.
Ports are non-obstructing terminal_NIs and retain their exact input coordinates.
Only positive-area movable macros with height > 5*row_height are supported.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from collections.abc import Mapping

import torch


SCRATCH_ROOT = os.environ.get("FP_WORK_ROOT", "./work")
DEFAULT_DP_ROOT = SCRATCH_ROOT + "/tools/dreamplace-4.1.0-torch221-cpu"
DEFAULT_PYTHON = SCRATCH_ROOT + "/envs/flowplace-py310-torch221/bin/python"
MODES = {
    "B": "dreamplace", "C": "dreamplace_local", "cascade": "dreamplace_cascade",
    "dreamplace": "dreamplace", "dreamplace_local": "dreamplace_local",
    "dreamplace_cascade": "dreamplace_cascade",
}


def _metadata(metadata, cond, count, size_tolerance_dbu=1.0):
    if isinstance(metadata, (str, os.PathLike)):
        with open(metadata, encoding="utf-8") as stream:
            metadata = json.load(stream)
    if not isinstance(metadata, Mapping):
        raise ValueError("Explicit physical metadata mapping or JSON path is required")
    required = ("bounds_dbu", "node_sizes_dbu", "node_roles", "site_width_dbu", "row_height_dbu")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError("Missing physical metadata: " + ", ".join(missing))
    bounds = torch.as_tensor(metadata["bounds_dbu"], dtype=torch.float64)
    sizes = torch.as_tensor(metadata["node_sizes_dbu"], dtype=torch.float64)
    roles = list(metadata["node_roles"])
    pitches = torch.tensor([metadata["site_width_dbu"], metadata["row_height_dbu"]], dtype=torch.float64)
    if bounds.shape != (4,) or sizes.shape != (count, 2) or len(roles) != count:
        raise ValueError("Metadata bounds/sizes/roles shape does not match graph node order")
    if not all(role in ("movable", "fixed", "port") for role in roles):
        raise ValueError("node_roles must explicitly contain movable, fixed, or port")
    if not torch.isfinite(bounds).all() or not torch.isfinite(sizes).all() or not torch.isfinite(pitches).all():
        raise ValueError("Physical metadata must be finite")
    span = bounds[2:] - bounds[:2]
    if not (span > 0).all() or not (pitches >= 1).all() or not (sizes >= 0).all():
        raise ValueError("Invalid physical bounds, nonnegative dimensions, or DBU pitches")
    for value in (bounds, sizes, pitches):
        if not torch.allclose(value, value.round(), rtol=0, atol=1e-6):
            raise ValueError("DREAMPlace LP requires physical integer DBU geometry; normalized geometry is unsupported")
    movable = torch.tensor([role == "movable" for role in roles], dtype=torch.bool)
    fixed = torch.tensor([role == "fixed" for role in roles], dtype=torch.bool)
    ports = torch.tensor([role == "port" for role in roles], dtype=torch.bool)
    if not (sizes[~ports] > 0).all():
        raise ValueError("Movable macros and fixed obstacles require positive width and height")
    if ((sizes[movable] > span).any()):
        raise ValueError("A movable macro is larger than the rectangular placement canvas")
    if not (sizes[movable, 1] > 5 * pitches[1]).all():
        raise ValueError("DREAMPlace would skip a movable macro: height must exceed 5*row_height")
    graph_sizes = torch.as_tensor(cond.x).detach().cpu().to(torch.float64)
    if graph_sizes.shape != (count, 2) or not torch.isfinite(graph_sizes).all():
        raise ValueError("cond.x must contain finite N x 2 normalized full dimensions")
    if not torch.allclose(graph_sizes * span / 2, sizes, rtol=0, atol=size_tolerance_dbu):
        raise ValueError("Physical node sizes do not match cond.x / normalization canvas")
    cond_ports = getattr(cond, "is_ports", None)
    if cond_ports is not None:
        cond_ports = torch.as_tensor(cond_ports).detach().cpu().bool().reshape(-1)
        if cond_ports.shape != ports.shape or (cond_ports & movable).any():
            raise ValueError("Explicit movable roles contradict the existing frozen-node mask")
    anchors = None
    if fixed.any():
        if "fixed_ll_dbu" not in metadata:
            raise ValueError("Fixed obstacles require explicit fixed_ll_dbu anchors in node order")
        anchors = torch.as_tensor(metadata["fixed_ll_dbu"], dtype=torch.float64)
        if anchors.shape != (count, 2) or not torch.isfinite(anchors[fixed]).all():
            raise ValueError("fixed_ll_dbu must be N x 2 with finite fixed-obstacle anchors")
    return {"bounds": bounds, "sizes": sizes, "pitches": pitches, "movable": movable,
            "fixed": fixed, "ports": ports, "anchors": anchors, "roles": roles,
            "design": str(metadata.get("design", "unspecified"))}


def _to_ll(x, data):
    bounds = data["bounds"]
    return bounds[:2] + (x.to(torch.float64) + 1) * (bounds[2:] - bounds[:2]) / 2 - data["sizes"] / 2


def _from_ll(ll, data):
    bounds = data["bounds"]
    return 2 * (ll + data["sizes"] / 2 - bounds[:2]) / (bounds[2:] - bounds[:2]) - 1


def _check(ll, reference_ll, data, tolerance_dbu, grid_tolerance_dbu):
    """Independent AABB checks; geometry and grid results remain separate."""
    finite = bool(torch.isfinite(ll).all())
    result = {"finite": finite, "geometry_success": False, "grid_success": False,
              "success": False, "overlap_pairs": [], "boundary_nodes": [],
              "offgrid_nodes": [], "fixed_moved_nodes": []}
    if not finite:
        result["reason"] = "nonfinite_coordinates"
        return result
    sizes, bounds = data["sizes"], data["bounds"]
    physical = ~data["ports"]
    lower_violation = (bounds[:2] - ll).clamp_min(0)
    upper_violation = (ll + sizes - bounds[2:]).clamp_min(0)
    boundary = torch.maximum(lower_violation, upper_violation).max(dim=-1).values
    boundary_nodes = torch.nonzero(physical & (boundary > tolerance_dbu)).flatten().tolist()
    overlap = torch.minimum((ll + sizes)[:, None], (ll + sizes)[None]) - torch.maximum(ll[:, None], ll[None])
    overlap_mask = (overlap > tolerance_dbu).all(dim=-1) & physical[:, None] & physical[None]
    pairs = torch.nonzero(torch.triu(overlap_mask, diagonal=1)).tolist()
    frozen = ~data["movable"]
    delta = (ll - reference_ll).abs().max(dim=-1).values
    changed = torch.nonzero(frozen & (delta > tolerance_dbu)).flatten().tolist()
    if data["anchors"] is not None:
        anchor_delta = (ll - data["anchors"]).abs().max(dim=-1).values
        changed = sorted(set(changed + torch.nonzero(data["fixed"] & (anchor_delta > tolerance_dbu)).flatten().tolist()))
    grid_delta = ((ll - bounds[:2]) / data["pitches"] - ((ll - bounds[:2]) / data["pitches"]).round()).abs() * data["pitches"]
    offgrid = torch.nonzero(data["movable"] & (grid_delta.max(dim=-1).values > grid_tolerance_dbu)).flatten().tolist()
    max_boundary = float(boundary[physical].max()) if physical.any() else 0.0
    max_grid = float(grid_delta[data["movable"]].max()) if data["movable"].any() else 0.0
    max_overlap = max((float(overlap[a, b].min()) for a, b in pairs), default=0.0)
    geometry = not pairs and not boundary_nodes and not changed
    result.update(geometry_success=geometry, grid_success=not offgrid,
                  success=geometry and not offgrid, overlap_pairs=pairs,
                  boundary_nodes=boundary_nodes, offgrid_nodes=offgrid,
                  fixed_moved_nodes=changed, max_boundary_dbu=max_boundary,
                  max_grid_residual_dbu=max_grid, max_overlap_depth_dbu=max_overlap,
                  reason="ok" if geometry and not offgrid else "geometry_or_grid_failed")
    return result


def validate_placement(x, cond, metadata, reference=None, tolerance_dbu=1.0,
                       grid_tolerance_dbu=1.0, size_tolerance_dbu=1.0):
    """Return one check dict per candidate, also usable for raw/legacy outputs.

    geometry_success intentionally does not include site/row alignment. Pass the
    original Y as reference to check frozen-node preservation across a strategy.
    """
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[-1] != 2:
        raise ValueError("Placement must have shape (N,2) or (B,N,2)")
    data = _metadata(metadata, cond, x.shape[1], size_tolerance_dbu)
    values = x.detach().cpu().to(torch.float64)
    if reference is None:
        reference = values
    elif reference.ndim == 2:
        reference = reference.unsqueeze(0)
    reference = reference.detach().cpu().to(torch.float64)
    if reference.shape != values.shape:
        raise ValueError("Reference shape must match candidate shape")
    return [_check(_to_ll(v, data), _to_ll(r, data), data, tolerance_dbu, grid_tolerance_dbu)
            for v, r in zip(values, reference)]


def _run_operator(reference_ll, current_ll, active, data, options):
    active = sorted(active)
    inactive = [i for i in range(len(data["roles"])) if i not in active and not data["ports"][i]]
    ports = torch.nonzero(data["ports"]).flatten().tolist()
    order = active + inactive + ports
    folder = Path(tempfile.mkdtemp(prefix="macro-", dir=options["scratch_root"]))
    # Native fixed-obstacle checks read init_pos, so inactive anchors must be
    # current positions after earlier C components; active objectives stay Y.
    initial_ll = current_ll.clone()
    initial_ll[active] = reference_ll[active]
    payload = {"init_ll": initial_ll[order].tolist(), "ll": current_ll[order].tolist(),
               "sizes": data["sizes"][order].tolist(), "bounds": data["bounds"].tolist(),
               "pitches": data["pitches"].tolist(), "num_movable": len(active),
               "num_terminal_NIs": len(ports), "num_bins": options["num_bins"]}
    input_path, output_path = folder / "input.json", folder / "output.json"
    input_path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    env = os.environ.copy()
    dp_root = Path(options["dreamplace_root"])
    env["PYTHONPATH"] = os.pathsep.join([str(dp_root / "pydeps"), str(dp_root / "install")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    command = [options["dreamplace_python"], str(Path(__file__).resolve()),
               "--worker-input", str(input_path), "--worker-output", str(output_path)]
    record = {"work_dir": str(folder), "active_nodes": active, "num_active": len(active)}
    start = time.perf_counter()
    try:
        with (folder / "stdout.log").open("w") as stdout, (folder / "stderr.log").open("w") as stderr:
            process = subprocess.run(command, cwd=folder, env=env, stdout=stdout,
                                     stderr=stderr, timeout=options["timeout_seconds"], check=False)
        record["returncode"] = process.returncode
        if process.returncode != 0:
            record["reason"] = "worker_signal" if process.returncode < 0 else "worker_error"
            return None, record
        if not output_path.exists():
            record["reason"] = "missing_worker_output"
            return None, record
        result = json.loads(output_path.read_text(encoding="utf-8"))
        ordered_ll = torch.as_tensor(result["ll"], dtype=torch.float64)
        if ordered_ll.shape != current_ll.shape:
            record["reason"] = "worker_output_shape"
            return None, record
        ll = current_ll.clone()
        ll[order] = ordered_ll
        record["reason"] = "operator_returned"
        return ll, record
    except subprocess.TimeoutExpired:
        record["reason"] = "worker_timeout"
        return None, record
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        record["reason"] = "worker_exception"
        record["error"] = str(error)
        return None, record
    finally:
        record["wall_seconds"] = time.perf_counter() - start


def _neighbours(active, ll, data, count):
    if not active or count == 0:
        return set(active)
    available = [i for i in torch.nonzero(data["movable"]).flatten().tolist() if i not in active]
    if not available:
        return set(active)
    seeds = sorted(active)
    # Edge-to-edge distance prioritizes immediate geometric neighbours.
    gap = torch.maximum(ll[available, None] - (ll + data["sizes"])[seeds],
                        ll[seeds] - (ll + data["sizes"])[available, None]).clamp_min(0)
    distance = gap.square().sum(dim=-1).min(dim=-1).values
    ranked = sorted(zip(distance.tolist(), available))
    return set(active) | {i for _, i in ranked[:count]}


def _accepted(check, active, require_full_grid):
    if not check["geometry_success"]:
        return False
    return check["grid_success"] if require_full_grid else not (set(check["offgrid_nodes"]) & set(active))


def _conflict_components(check, movable):
    """Overlap-connected movable components plus isolated boundary violators."""
    adjacency = {}
    for pair in check["overlap_pairs"]:
        endpoints = set(pair) & movable
        for node in endpoints:
            adjacency.setdefault(node, set()).update(endpoints - {node})
    for node in set(check["boundary_nodes"]) & movable:
        adjacency.setdefault(node, set())
    components = []
    unseen = set(adjacency)
    while unseen:
        pending, component = [min(unseen)], set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)
        unseen -= component
        components.append(component)
    return sorted(components, key=lambda nodes: (-len(nodes), min(nodes)))


def _local_candidate_ok(check, active):
    return (check["finite"] and not check["fixed_moved_nodes"]
            and not (set(check["boundary_nodes"]) & active)
            and not (set(check["offgrid_nodes"]) & active)
            and not any(set(pair) & active for pair in check["overlap_pairs"]))


def _solve(original_ll, data, mode, options):
    tolerance, grid_tolerance = options["tolerance_dbu"], options["grid_tolerance_dbu"]
    initial_check = _check(original_ll, original_ll, data, tolerance, grid_tolerance)
    record = {"strategy": mode, "attempts": [], "fallback": False, "initial_check": initial_check,
              "require_full_grid": options["require_full_grid"], "final_active_nodes": []}
    if not initial_check["finite"]:
        return None, dict(record, success=False, reason="nonfinite_input")
    if initial_check["fixed_moved_nodes"]:
        return None, dict(record, success=False, reason="input_fixed_anchor_mismatch")
    movable = set(torch.nonzero(data["movable"]).flatten().tolist())
    if initial_check["success"]:
        return original_ll.clone(), dict(record, success=True, reason="already_legal", check=initial_check)
    if not movable:
        return None, dict(record, success=False, reason="no_movable_nodes", check=initial_check)
    if mode in ("dreamplace_local", "dreamplace_cascade"):
        current = original_ll.clone()
        ever_active = set()
        max_active = max(1, math.floor(len(movable) * options["local_max_fraction"]))
        local_reason = "local_iteration_budget_exceeded"
        for component_step in range(len(movable) + 1):
            check = _check(current, original_ll, data, tolerance, grid_tolerance)
            if _accepted(check, ever_active, options["require_full_grid"]):
                record["final_active_nodes"] = sorted(ever_active)
                return current, dict(record, success=True,
                                     reason="ok" if ever_active else "already_geometry_legal", check=check)
            components = _conflict_components(check, movable)
            if not components:
                local_reason = "inactive_offgrid" if check["geometry_success"] else "unrepairable_fixed_geometry"
                break
            if component_step == len(movable):
                break
            active = set(components[0])
            component_repaired = False
            for expansion in range(options["local_max_expansions"] + 1):
                if len(active) > max_active:
                    local_reason = "local_active_budget_exceeded"
                    break
                active = _neighbours(active, current, data,
                                     min(options["local_neighbors"], max_active - len(active)))
                candidate, attempt = _run_operator(original_ll, current, active, data, options)
                attempt.update(stage="C", component_step=component_step, expansion=expansion,
                               seed_nodes=sorted(components[0]), num_components=len(components),
                               active_fraction=len(active) / len(movable))
                record["attempts"].append(attempt)
                if candidate is not None:
                    candidate_check = _check(candidate, original_ll, data, tolerance, grid_tolerance)
                    inactive = torch.tensor([i not in active for i in range(len(data["roles"]))])
                    preserved = bool(torch.allclose(candidate[inactive], current[inactive], rtol=0, atol=tolerance))
                    attempt.update(check=candidate_check, inactive_preserved=preserved)
                    if _local_candidate_ok(candidate_check, active) and preserved:
                        # Exact preservation avoids accumulating native round-off.
                        candidate[inactive] = current[inactive]
                        current = candidate
                        ever_active |= active
                        component_repaired = True
                        attempt["component_success"] = True
                        break
                    local_reason = "local_geometry_failed" if preserved else "inactive_obstacle_moved"
                    if preserved and candidate_check["geometry_success"] and set(candidate_check["offgrid_nodes"]) & active:
                        local_reason = "active_offgrid"
                else:
                    local_reason = attempt["reason"]
                attempt["component_success"] = False
                grown = _neighbours(active, current, data, max(1, options["local_neighbors"]))
                if grown == active:
                    break
                active = grown
            record["local_final_active_count"] = len(active)
            if not component_repaired:
                break
        record["local_failure_reason"] = local_reason
        record["local_repaired_nodes"] = sorted(ever_active)
        if mode == "dreamplace_local":
            return None, dict(record, success=False, reason=local_reason)
        record["fallback"] = True
    candidate, attempt = _run_operator(original_ll, original_ll, movable, data, options)
    attempt["stage"] = "B"
    record["attempts"].append(attempt)
    record["final_active_nodes"] = sorted(movable)
    if candidate is None:
        return None, dict(record, success=False, reason=attempt["reason"])
    check = _check(candidate, original_ll, data, tolerance, grid_tolerance)
    attempt["check"] = check
    return (candidate if check["success"] else None), dict(record, success=check["success"],
                                                          reason=check["reason"], check=check)


@torch.no_grad()
def legalize(x, cond, metadata=None, mode="dreamplace", *,
             scratch_root=SCRATCH_ROOT + "/runs/legalization", dreamplace_root=DEFAULT_DP_ROOT,
             dreamplace_python=DEFAULT_PYTHON, timeout_seconds=120, num_bins=8,
             local_neighbors=2, local_max_expansions=2, local_max_fraction=0.75,
             tolerance_dbu=1.0, grid_tolerance_dbu=1.0, size_tolerance_dbu=1.0,
             require_full_grid=False, **unused_config):
    """Terminal adapter with the existing (X, metrics, metrics_special) contract.

    Failed candidates return their exact original Y and a failure record. No
    clamping, replacement with target placements, or hidden fallback is applied.
    Input tensors / graph condition are never modified. B and C are opt-in.
    """
    unknown = set(unused_config) - {"softmax_min", "softmax_max"}
    if unknown:
        raise TypeError("Unknown DREAMPlace settings: " + ", ".join(sorted(unknown)))
    if mode not in MODES:
        raise ValueError("Unsupported DREAMPlace legalizer mode: " + str(mode))
    mode = MODES[mode]
    squeeze = x.ndim == 2
    samples = x.unsqueeze(0) if squeeze else x
    if samples.ndim != 3 or samples.shape[-1] != 2 or not samples.is_floating_point():
        raise ValueError("x must be a floating tensor with shape (N,2) or (B,N,2)")
    if not 0 < local_max_fraction <= 1 or min(local_neighbors, local_max_expansions) < 0:
        raise ValueError("Invalid local neighbourhood budget")
    if timeout_seconds <= 0 or num_bins < 1 or min(tolerance_dbu, grid_tolerance_dbu, size_tolerance_dbu) < 0:
        raise ValueError("Invalid operator timeout, bins, or geometry tolerances")
    data = _metadata(metadata, cond, samples.shape[1], size_tolerance_dbu)
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    options = dict(scratch_root=str(scratch_root), dreamplace_root=str(dreamplace_root),
                   dreamplace_python=str(dreamplace_python), timeout_seconds=timeout_seconds,
                   num_bins=int(num_bins), local_neighbors=int(local_neighbors),
                   local_max_expansions=int(local_max_expansions), local_max_fraction=local_max_fraction,
                   tolerance_dbu=tolerance_dbu, grid_tolerance_dbu=grid_tolerance_dbu,
                   require_full_grid=bool(require_full_grid))
    output = samples.detach().clone()
    originals = samples.detach().cpu().to(torch.float64)
    records = []
    for index, original in enumerate(originals):
        start = time.perf_counter()
        original_ll = _to_ll(original, data)
        result, record = _solve(original_ll, data, mode, options)
        if result is not None:
            normalized = _from_ll(result, data).to(device=x.device, dtype=x.dtype)
            # Preserve all frozen coordinates exactly, including zero-area ports.
            frozen = torch.tensor([i not in record["final_active_nodes"] for i in range(samples.shape[1])],
                                  dtype=torch.bool, device=x.device)
            normalized[frozen] = samples[index, frozen]
            final_check = _check(_to_ll(normalized.detach().cpu(), data), original_ll,
                                 data, tolerance_dbu, grid_tolerance_dbu)
            record["returned_check"] = final_check
            if _accepted(final_check, record["final_active_nodes"], require_full_grid):
                output[index] = normalized
            else:
                record.update(success=False, reason="dtype_roundtrip_failed")
        record.update(candidate=index, design=data["design"], wall_seconds=time.perf_counter() - start)
        displacement = (_to_ll(output[index].detach().cpu(), data) - original_ll).norm(dim=-1)[data["movable"]]
        defined = bool(torch.isfinite(displacement).all())
        record["mean_displacement_dbu"] = (float(displacement.mean()) if displacement.numel() else 0.0) if defined else None
        record["max_displacement_dbu"] = (float(displacement.max()) if displacement.numel() else 0.0) if defined else None
        records.append(record)
    total = max(1, len(records))
    metrics = {"legalization_success_rate": sum(r["success"] for r in records) / total,
               "legalization_fallback_rate": sum(r["fallback"] for r in records) / total,
               "legalization_time": sum(r["wall_seconds"] for r in records),
               "legalization_mean_displacement_dbu": sum(r["mean_displacement_dbu"] or 0.0 for r in records) / total}
    return (output[0] if squeeze else output), metrics, {"legalization_records": records}


def _worker(input_path, output_path):
    # Only this subprocess imports the compiled official operator.
    from dreamplace.ops.macro_legalize.macro_legalize import MacroLegalize
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    sizes = torch.tensor(payload["sizes"], dtype=torch.float64)
    initial = torch.tensor(payload["init_ll"], dtype=torch.float64).t().contiguous().view(-1)
    current = torch.tensor(payload["ll"], dtype=torch.float64).t().contiguous().view(-1)
    bounds, pitches = payload["bounds"], payload["pitches"]
    operator = MacroLegalize(
        node_size_x=sizes[:, 0].contiguous(), node_size_y=sizes[:, 1].contiguous(),
        node_weights=torch.ones(sizes.shape[0], dtype=torch.float64),
        flat_region_boxes=torch.empty(0, dtype=torch.float64),
        flat_region_boxes_start=torch.zeros(1, dtype=torch.int32),
        node2fence_region_map=torch.empty(0, dtype=torch.int32),
        xl=bounds[0], yl=bounds[1], xh=bounds[2], yh=bounds[3],
        site_width=pitches[0], row_height=pitches[1],
        num_bins_x=payload["num_bins"], num_bins_y=payload["num_bins"],
        num_movable_nodes=payload["num_movable"],
        num_terminal_NIs=payload["num_terminal_NIs"], num_filler_nodes=0,
    )
    result = operator(initial, current).reshape(2, -1).t().contiguous()
    Path(output_path).write_text(json.dumps({"ll": result.tolist()}, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated DREAMPlace macro legalizer worker")
    parser.add_argument("--worker-input", required=True)
    parser.add_argument("--worker-output", required=True)
    args = parser.parse_args()
    _worker(args.worker_input, args.worker_output)
