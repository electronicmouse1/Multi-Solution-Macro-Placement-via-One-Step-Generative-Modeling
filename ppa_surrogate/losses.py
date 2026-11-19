"""Predictor losses: per-design normalized regression + within-design pairwise ranking + optional maps."""
import torch
import torch.nn.functional as F


def scalar_loss(pred, z, head_mask, delta=1.0):
    """pred, z: (B,T); head_mask: (T,) bool. Huber on per-design z-scores."""
    if head_mask.sum() == 0:
        return pred.sum() * 0.0
    return F.huber_loss(pred[:, head_mask], z[:, head_mask], delta=delta)


def rank_loss(pred, z, head_mask, tau=0.1, cap=2.0):
    """Pairwise logistic ranking within the batch (all samples are one design).
    Pairs closer than tau (in robust z units, ~label noise floor) are ignored; weights grow with the gap."""
    if head_mask.sum() == 0 or pred.shape[0] < 2:
        return pred.sum() * 0.0
    p, t = pred[:, head_mask], z[:, head_mask]                     # (B,T')
    dp = p.unsqueeze(1) - p.unsqueeze(0)                            # (B,B,T')
    dt = t.unsqueeze(1) - t.unsqueeze(0)
    w = (dt.abs() > tau).float() * dt.abs().clamp(max=cap)
    iu = torch.triu(torch.ones(p.shape[0], p.shape[0], device=p.device, dtype=torch.bool), 1)
    w = w * iu.unsqueeze(-1)
    loss = F.softplus(-torch.sign(dt) * dp) * w
    return loss.sum() / w.sum().clamp_min(1.0)


def map_loss(pred_maps, true_maps, use_cong):
    """pred/true: (B,3,G,G) = log1p(cong_H), log1p(cong_V), cell density. Congestion channels only where the
    design has an overflow signal."""
    if pred_maps is None or true_maps is None:
        return None
    lc = F.mse_loss(pred_maps[:, :2], true_maps[:, :2]) if use_cong else pred_maps.sum() * 0.0
    ld = F.mse_loss(pred_maps[:, 2:], true_maps[:, 2:])
    return lc + ld
