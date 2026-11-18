"""Optional B_grid / hard_grid endpoints; legacy policies remain unchanged.

The native B operator floors continuous coordinates onto site/row pitches at the
end. For non-pitch-multiple macro heights this can create overlaps after its LP
legality check. This module keeps the native candidate, snaps to the real grid,
and repairs only remaining conflicts by deterministic single-node relocation.

For each conflicting node, candidate grid coordinates come from the base target,
canvas edges and obstacle edges. Their Cartesian product includes a nearest
feasible lattice point for that node with all other rectangles fixed. Accepted
moves introduce no overlaps and remove at least one current conflict. This is a
bounded local repair, not a complete multi-macro packing solver.

If no single-node slot exists, one explicitly recorded optional DREAMPlace retry
uses conservative grid envelopes. Pitch-multiple envelopes preserve separating
inequalities under common-origin floor snapping; fixed envelopes enclose the
real fixed rectangles. Returned placements always use the ORIGINAL dimensions
and exact frozen coordinates and undergo independent full geometry/grid checks.
No HPWL measurements or evaluation targets influence repair choices.
"""
import copy
import math
from pathlib import Path
import time

import torch

try:
    from flow_matching import dreamplace_legalization as dp
except ImportError:  # support the existing flow_matching script entry point
    import dreamplace_legalization as dp

MODES = ("B_grid", "hard_grid")


def _snap(ll, data):
    result = ll.clone()
    origin, high, pitches = data["bounds"][:2], data["bounds"][2:], data["pitches"]
    for node in torch.nonzero(data["movable"]).flatten().tolist():
        maximum = torch.floor((high - origin - data["sizes"][node]) / pitches)
        if (maximum < 0).any():
            return None
        grid = ((ll[node] - origin) / pitches).round().clamp_min(0)
        result[node] = origin + torch.minimum(grid, maximum) * pitches
    return result


def _nearest_slot(node, current, target, data):
    """Find the nearest legal grid slot with every other node held fixed.

    Obstacle edges partition each axis into intervals where all overlap predicates
    are constant. On each resulting feasible rectangle, the squared-distance
    minimum is at the target projection or an interval endpoint; the candidate
    sets enumerate exactly these lattice event coordinates, without scanning
    every site in the design. Ties use x then y for determinism.
    """
    origin, high, pitches = data["bounds"][:2], data["bounds"][2:], data["pitches"]
    size = data["sizes"][node]
    obstacles = [i for i in range(len(data["roles"])) if i != node and not data["ports"][i]]
    axes = []
    for axis in range(2):
        low, pitch = float(origin[axis]), float(pitches[axis])
        limit = math.floor(float((high[axis] - origin[axis] - size[axis]) / pitches[axis]))
        if limit < 0:
            return None
        relative_target = float((target[node, axis] - origin[axis]) / pitches[axis])
        events = {0, limit, math.floor(relative_target), math.ceil(relative_target)}
        for obstacle in obstacles:
            # Place entirely before or after this fixed rectangle on this axis.
            before = float((current[obstacle, axis] - size[axis] - origin[axis]) / pitches[axis])
            after = float((current[obstacle, axis] + data["sizes"][obstacle, axis] - origin[axis]) / pitches[axis])
            events.add(math.floor(before))
            events.add(math.ceil(after))
        axes.append(torch.tensor([low + value * pitch for value in sorted(events) if 0 <= value <= limit], dtype=torch.float64))
    xx, yy = torch.meshgrid(axes[0], axes[1], indexing="ij")
    positions = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    if obstacles:
        lower = current[obstacles]
        upper = lower + data["sizes"][obstacles]
        overlap = torch.minimum((positions + size)[:, None, :], upper[None, :, :]) - torch.maximum(positions[:, None, :], lower[None, :, :])
        legal = ~((overlap > 1e-7).all(dim=-1).any(dim=-1))
        positions = positions[legal]
    if not positions.numel():
        return None
    squared = (positions - target[node]).square().sum(dim=-1)
    index = int(torch.argmin(squared))  # flattened coordinates have stable x/y order
    return positions[index], float(torch.sqrt(squared[index]))


def _repair_check(ll, reference_ll, data, tolerance_dbu, grid_tolerance_dbu):
    """Do not manufacture tolerated overlap slivers at the 1-DBU cutoff.

    Boundary, grid and frozen-coordinate checks keep their original physical
    tolerances. Movable-incident overlaps use a numerical epsilon only, so a
    1-DBU real overlap must be repaired before a normalized round trip. Fixed-
    fixed pairs retain the public check's tolerance: they cannot be repaired and
    their normalized input coordinates may contain permitted floating error.
    """
    check = dp._check(ll, reference_ll, data, tolerance_dbu, grid_tolerance_dbu)
    if not check["finite"]:
        return check
    physical = ~data["ports"]
    overlap = torch.minimum((ll + data["sizes"])[:, None], (ll + data["sizes"])[None]) - torch.maximum(ll[:, None], ll[None])
    incident = data["movable"][:, None] | data["movable"][None]
    strict_mask = (overlap > 1e-7).all(dim=-1) & physical[:, None] & physical[None] & incident
    pairs = {tuple(pair) for pair in check["overlap_pairs"]}
    pairs.update(tuple(pair) for pair in torch.nonzero(torch.triu(strict_mask, diagonal=1)).tolist())
    check["overlap_pairs"] = [list(pair) for pair in sorted(pairs)]
    check["max_overlap_depth_dbu"] = max((float(overlap[a, b].min()) for a, b in pairs), default=0.0)
    check["geometry_success"] = not pairs and not check["boundary_nodes"] and not check["fixed_moved_nodes"]
    check["success"] = check["geometry_success"] and check["grid_success"]
    check["movable_overlap_epsilon_dbu"] = 1e-7
    if pairs:
        check["reason"] = "strict_movable_overlap"
    return check


def _repair(base_ll, reference_ll, data, max_moves, tolerance_dbu, grid_tolerance_dbu, target_ll=None):
    target_ll = base_ll if target_ll is None else target_ll
    record = {"moves": [], "success": False}
    if not torch.isfinite(base_ll).all():
        return None, dict(record, reason="nonfinite_base")
    current = _snap(base_ll, data)
    if current is None:
        return None, dict(record, reason="macro_does_not_fit_grid")
    record["snap_max_displacement_dbu"] = float((current - base_ll).norm(dim=-1).max()) if current.numel() else 0.0
    for iteration in range(max_moves + 1):
        check = _repair_check(current, reference_ll, data, tolerance_dbu, grid_tolerance_dbu)
        if check["success"]:
            return current, dict(record, success=True, reason="ok", check=check)
        if not check["finite"] or check["fixed_moved_nodes"]:
            return None, dict(record, reason="fixed_or_finite_invariant", check=check)
        if iteration == max_moves:
            return None, dict(record, reason="repair_move_budget_exceeded", check=check)
        conflicts = set(check["boundary_nodes"])
        for pair in check["overlap_pairs"]:
            conflicts.update(pair)
        conflicts &= set(torch.nonzero(data["movable"]).flatten().tolist())
        if not conflicts:
            return None, dict(record, reason="unrepairable_fixed_geometry", check=check)
        alternatives = []
        for node in sorted(conflicts):
            slot = _nearest_slot(node, current, target_ll, data)
            if slot is None:
                continue
            position, base_distance = slot
            previous_distance = float((current[node] - target_ll[node]).norm())
            alternatives.append((base_distance - previous_distance, node, position, base_distance))
        if not alternatives:
            return None, dict(record, reason="no_single_node_grid_slot", check=check)
        _, node, position, base_distance = min(alternatives, key=lambda item: (item[0], item[1], float(item[2][0]), float(item[2][1])))
        previous = current[node].clone()
        current[node] = position
        after = _repair_check(current, reference_ll, data, tolerance_dbu, grid_tolerance_dbu)
        # A valid move clears this node and creates no new conflict anywhere.
        before_pairs = {tuple(pair) for pair in check["overlap_pairs"]}
        after_pairs = {tuple(pair) for pair in after["overlap_pairs"]}
        if not after_pairs <= before_pairs or any(node in pair for pair in after_pairs):
            return None, dict(record, reason="repair_progress_invariant", check=after)
        record["moves"].append({"iteration": iteration, "node": node,
                                "from_ll_dbu": previous.tolist(), "to_ll_dbu": position.tolist(),
                                "move_dbu": float((position - previous).norm()),
                                "base_distance_dbu": base_distance,
                                "overlap_pairs_before": len(before_pairs), "overlap_pairs_after": len(after_pairs)})
    raise AssertionError("bounded repair loop did not return")


def _envelope_problem(current, reference, data):
    """Conservatively enclose geometry in the common physical lattice."""
    envelope = dict(data)
    sizes = data["sizes"].clone()
    initial, seed = reference.clone(), current.clone()
    pitches, origin = data["pitches"], data["bounds"][:2]
    movable = data["movable"]
    sizes[movable] = torch.ceil(sizes[movable] / pitches) * pitches
    for node in torch.nonzero(data["fixed"]).flatten().tolist():
        lo = origin + torch.floor((current[node] - origin) / pitches) * pitches
        hi = origin + torch.ceil((current[node] + sizes[node] - origin) / pitches) * pitches
        sizes[node] = hi - lo
        seed[node] = lo
        initial[node] = lo
    envelope["sizes"] = sizes
    envelope["anchors"] = seed.clone() if data["fixed"].any() else None
    bounds = data["bounds"].clone()
    bounds[2:] = origin + torch.floor((bounds[2:] - origin) / pitches) * pitches
    envelope["bounds"] = bounds
    if (sizes[movable] > bounds[2:] - bounds[:2]).any():
        return None
    return seed, initial, envelope


def _base_candidate(sample, reference_ll, cond, data, mode, options):
    if mode == "B_grid":
        active = set(torch.nonzero(data["movable"]).flatten().tolist())
        if not active:
            return reference_ll.clone(), {"stage": "B", "reason": "no_movable_nodes"}
        ll, record = dp._run_operator(reference_ll, reference_ll, active, data, options)
        record["stage"] = "B"
        if ll is not None:
            record["check"] = dp._check(ll, reference_ll, data, options["tolerance_dbu"], options["grid_tolerance_dbu"])
        return ll, record
    try:
        from flow_matching.hard_constraint_legalizer import Legalizer
    except ImportError:
        from hard_constraint_legalizer import Legalizer
    hard_cond = copy.deepcopy(cond)
    hard_cond.x = cond.x.detach().cpu()
    hard_cond.is_ports = (~data["movable"]).clone()
    projector = Legalizer(cond=hard_cond, device="cpu", dtype=sample.dtype,
                          grid_res=options["hard_grid_res"], padding=options["hard_padding"],
                          scores={"w_reg": 0.1, "w_dist": 1.0})
    result = projector.run(sample.detach().cpu().unsqueeze(0).clone())[0]
    ll = dp._to_ll(result.to(torch.float64), data)
    return ll, {"stage": "hard", "reason": "operator_returned",
                "grid_res": options["hard_grid_res"], "padding": options["hard_padding"],
                "scores": {"w_reg": 0.1, "w_dist": 1.0},
                "check": dp._check(ll, reference_ll, data, options["tolerance_dbu"], options["grid_tolerance_dbu"])}


@torch.no_grad()
def legalize(x, cond, metadata=None, mode="B_grid", *, base=None,
             scratch_root=dp.SCRATCH_ROOT + "/runs/grid-legalization",
             dreamplace_root=dp.DEFAULT_DP_ROOT, dreamplace_python=dp.DEFAULT_PYTHON,
             timeout_seconds=120, num_bins=8, max_repair_moves=None,
             allow_envelope_fallback=True, hard_grid_res=128, hard_padding=0.001,
             tolerance_dbu=1.0, grid_tolerance_dbu=1.0, size_tolerance_dbu=1.0,
             **unused_config):
    """Run the full raw -> base -> grid repair policy, returning the usual tuple.

    Failure returns the exact original Y. All timings include fresh base execution
    and any failed attempts, never a cached B/hard output. Default base parameters
    are the unchanged parameters used in the existing paired comparison.
    """
    if mode == "grid_repair":
        mode = {"B": "B_grid", "hard": "hard_grid"}.get(base, "B_grid" if base is None else base)
    if mode not in MODES:
        raise ValueError("Unsupported grid repair mode: " + str(mode))
    if base is not None and base != {"B_grid": "B", "hard_grid": "hard"}[mode]:
        raise ValueError("base conflicts with the explicit grid repair mode")
    unknown = set(unused_config) - {"softmax_min", "softmax_max"}
    if unknown:
        raise TypeError("Unknown grid repair settings: " + ", ".join(sorted(unknown)))
    squeeze = x.ndim == 2
    samples = x.unsqueeze(0) if squeeze else x
    if samples.ndim != 3 or samples.shape[-1] != 2 or not samples.is_floating_point():
        raise ValueError("x must be a floating tensor of shape (N,2) or (B,N,2)")
    if timeout_seconds <= 0 or num_bins < 1 or hard_grid_res < 2 or hard_padding < 0:
        raise ValueError("Invalid timeout, bins, or hard projection parameters")
    if min(tolerance_dbu, grid_tolerance_dbu, size_tolerance_dbu) < 0:
        raise ValueError("Geometry tolerances must be nonnegative")
    data = dp._metadata(metadata, cond, samples.shape[1], size_tolerance_dbu)
    if max_repair_moves is None:
        max_repair_moves = int(data["movable"].sum())
    if int(max_repair_moves) != max_repair_moves or max_repair_moves < 0:
        raise ValueError("max_repair_moves must be a nonnegative integer")
    Path(scratch_root).mkdir(parents=True, exist_ok=True)
    options = dict(scratch_root=str(scratch_root), dreamplace_root=str(dreamplace_root),
                   dreamplace_python=str(dreamplace_python), timeout_seconds=timeout_seconds,
                   num_bins=int(num_bins), tolerance_dbu=tolerance_dbu, grid_tolerance_dbu=grid_tolerance_dbu,
                   hard_grid_res=int(hard_grid_res), hard_padding=float(hard_padding))
    output = samples.detach().clone()
    records = []
    for index, sample in enumerate(samples):
        started = time.perf_counter()
        reference = dp._to_ll(sample.detach().cpu().to(torch.float64), data)
        initial = _repair_check(reference, reference, data, tolerance_dbu, grid_tolerance_dbu)
        record = dict(candidate=index, design=data["design"], strategy=mode, success=False,
                      fallback=False, attempts=[], initial_check=initial, base_seconds=0.0,
                      repair_seconds=0.0, envelope_seconds=0.0, reason="uninitialized")
        candidate, base_ll = None, None
        if not initial["finite"] or initial["fixed_moved_nodes"]:
            record["reason"] = "nonfinite_input" if not initial["finite"] else "input_fixed_anchor_mismatch"
        elif initial["success"]:
            candidate = reference.clone()
            base_ll = reference.clone()
            record["reason"] = "already_legal"
        else:
            base_started = time.perf_counter()
            try:
                base_ll, attempt = _base_candidate(sample, reference, cond, data, mode, options)
            except Exception as error:
                base_ll, attempt = None, {"stage": "base", "reason": "base_exception", "error": str(error)}
            record["base_seconds"] = time.perf_counter() - base_started
            record["attempts"].append(attempt)
            if base_ll is not None:
                repair_started = time.perf_counter()
                candidate, repair = _repair(base_ll, reference, data, int(max_repair_moves), tolerance_dbu, grid_tolerance_dbu)
                record["repair_seconds"] += time.perf_counter() - repair_started
                record["attempts"].append(dict(repair, stage="grid_repair"))
                record["reason"] = repair["reason"]
            else:
                record["reason"] = attempt["reason"]
            if candidate is None and allow_envelope_fallback:
                record["fallback"] = True
                envelope_started = time.perf_counter()
                seed = base_ll if base_ll is not None and torch.isfinite(base_ll).all() else reference
                problem = _envelope_problem(seed, seed, data)
                if problem is None:
                    record["attempts"].append({"stage": "grid_envelope", "reason": "envelope_does_not_fit"})
                    record["reason"] = "envelope_does_not_fit"
                else:
                    envelope_seed, envelope_reference, envelope_data = problem
                    proposed, envelope_attempt = dp._run_operator(envelope_reference, envelope_seed,
                                                                 set(torch.nonzero(data["movable"]).flatten().tolist()),
                                                                 envelope_data, options)
                    envelope_attempt["stage"] = "grid_envelope"
                    envelope_attempt["inflated_sizes_dbu"] = envelope_data["sizes"].tolist()
                    record["attempts"].append(envelope_attempt)
                    record["envelope_seconds"] = time.perf_counter() - envelope_started
                    if proposed is not None:
                        envelope_attempt["envelope_check"] = dp._check(proposed, envelope_seed, envelope_data,
                                                                      tolerance_dbu, grid_tolerance_dbu)
                        proposed[~data["movable"]] = reference[~data["movable"]]
                        repair_started = time.perf_counter()
                        candidate, repair = _repair(proposed, reference, data, int(max_repair_moves), tolerance_dbu, grid_tolerance_dbu,
                                                    target_ll=base_ll if base_ll is not None and torch.isfinite(base_ll).all() else reference)
                        record["repair_seconds"] += time.perf_counter() - repair_started
                        record["attempts"].append(dict(repair, stage="grid_repair_after_envelope"))
                        record["reason"] = repair["reason"]
                    else:
                        record["reason"] = envelope_attempt["reason"]
                if record["envelope_seconds"] == 0.0:
                    record["envelope_seconds"] = time.perf_counter() - envelope_started
        if candidate is not None:
            normalized = sample.detach().clone() if initial["success"] else dp._from_ll(candidate, data).to(device=x.device, dtype=x.dtype)
            frozen = (~data["movable"]).to(x.device)
            normalized[frozen] = sample[frozen]
            returned = dp._check(dp._to_ll(normalized.detach().cpu(), data), reference, data, tolerance_dbu, grid_tolerance_dbu)
            record["returned_check"] = returned
            if returned["success"]:
                output[index] = normalized
                record["success"] = True
            else:
                record["reason"] = "dtype_roundtrip_failed"
        returned_ll = dp._to_ll(output[index].detach().cpu(), data)
        displacement = (returned_ll - reference).norm(dim=-1)[data["movable"]]
        finite = bool(torch.isfinite(displacement).all())
        record["mean_displacement_dbu"] = (float(displacement.mean()) if displacement.numel() else 0.0) if finite else None
        record["max_displacement_dbu"] = (float(displacement.max()) if displacement.numel() else 0.0) if finite else None
        if record["success"] and base_ll is not None and torch.isfinite(base_ll).all():
            base_displacement = (returned_ll - base_ll).norm(dim=-1)[data["movable"]]
            record["mean_base_displacement_dbu"] = float(base_displacement.mean()) if base_displacement.numel() else 0.0
            record["max_base_displacement_dbu"] = float(base_displacement.max()) if base_displacement.numel() else 0.0
        record["wall_seconds"] = time.perf_counter() - started
        records.append(record)
    count = max(1, len(records))
    metrics = {"legalization_success_rate": sum(item["success"] for item in records) / count,
               "legalization_fallback_rate": sum(item["fallback"] for item in records) / count,
               "legalization_time": sum(item["wall_seconds"] for item in records),
               "legalization_mean_displacement_dbu": sum(item["mean_displacement_dbu"] or 0 for item in records) / count}
    return (output[0] if squeeze else output), metrics, {"legalization_records": records}
