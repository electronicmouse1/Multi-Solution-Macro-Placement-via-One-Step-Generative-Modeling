import torch


def boundary_violation_metrics(pos, sizes, min_bound=-1.0, max_bound=1.0):
    """Measure raw normalized overhang without changing the placement."""
    if pos.numel() == 0:
        return {
            "raw_boundary_violation_mean": 0.0,
            "raw_boundary_violation_max": 0.0,
            "raw_boundary_violation_fraction": 0.0,
        }
    half_sizes = sizes / 2.0
    lower_overhang = torch.relu(min_bound - (pos - half_sizes))
    upper_overhang = torch.relu((pos + half_sizes) - max_bound)
    coordinate_overhang = lower_overhang + upper_overhang
    node_overhang = coordinate_overhang.sum(dim=-1)
    return {
        "raw_boundary_violation_mean": node_overhang.mean().item(),
        "raw_boundary_violation_max": coordinate_overhang.max().item(),
        "raw_boundary_violation_fraction": (
            node_overhang > 0
        ).to(torch.float32).mean().item(),
    }
