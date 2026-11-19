"""Differentiable features of a macro placement Y (B,N,2), normalized canvas [-1,1]^2.

Everything here is a smooth function of Y, so d(prediction)/dY exists for guidance.
Rasterization uses separable soft boxes / Gaussians: map = A_y^T diag(w) A_x, i.e. one batched
matmul per channel, (B,K,G) x (B,K,G) -> (B,G,G), with K boxes/points and G bins per axis.
"""
import torch
import torch.nn.functional as F

GRID = 64


def bin_centers(G=GRID, device=None):
    return (torch.arange(G, device=device, dtype=torch.float32) + 0.5) * (2.0 / G) - 1.0


def soft_interval(lo, hi, centers, sharp):
    """(..., K) intervals -> (..., K, G) soft indicator of bin centers inside [lo, hi]."""
    c = centers.view(*([1] * lo.dim()), -1)
    return torch.sigmoid(sharp * (c - lo.unsqueeze(-1))) - torch.sigmoid(sharp * (c - hi.unsqueeze(-1)))


def gaussian_1d(x, centers, sigma):
    c = centers.view(*([1] * x.dim()), -1)
    return torch.exp(-0.5 * ((c - x.unsqueeze(-1)) / sigma) ** 2)


def raster_boxes(lo, hi, weight, G=GRID, sharp=None):
    """lo, hi: (B,K,2); weight: (B,K) -> (B,G,G) indexed [y][x]."""
    centers = bin_centers(G, lo.device)
    sharp = sharp or G  # transition width ~ one bin
    ax = soft_interval(lo[..., 0], hi[..., 0], centers, sharp)        # (B,K,G)
    ay = soft_interval(lo[..., 1], hi[..., 1], centers, sharp)
    return torch.einsum("bky,bk,bkx->byx", ay, weight, ax)


def raster_points(p, weight, G=GRID, sigma_bins=1.5):
    """p: (B,K,2), weight (B,K) -> (B,G,G) Gaussian splat (mass-preserving per point)."""
    centers = bin_centers(G, p.device)
    s = sigma_bins * 2.0 / G
    gx = gaussian_1d(p[..., 0], centers, s)
    gy = gaussian_1d(p[..., 1], centers, s)
    gx = gx / gx.sum(-1, keepdim=True).clamp_min(1e-8)
    gy = gy / gy.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bky,bk,bkx->byx", gy, weight, gx)


def segment_lse(values, seg, n_seg, gamma):
    """Smooth max per segment: gamma * log sum exp(v / gamma). values: (B,P), seg: (P,) -> (B,n_seg)."""
    B = values.shape[0]
    vmax = torch.full((B, n_seg), -1e9, device=values.device).scatter_reduce(
        1, seg.expand(B, -1), values.detach(), reduce="amax", include_self=True)
    ex = torch.exp((values - vmax[:, seg]) / gamma)
    s = torch.zeros(B, n_seg, device=values.device).index_add_(1, seg, ex)
    return vmax + gamma * torch.log(s.clamp_min(1e-30))


def pin_positions(Y, design):
    """Absolute pin positions (B,P,2)."""
    return Y[:, design.pin_macro, :] + design.pin_off.unsqueeze(0)


def net_bboxes(Y, design, gamma=0.01):
    """Smooth per-net bbox (B,M,2) lo / hi."""
    P = pin_positions(Y, design)
    M = design.net_degree.shape[0]
    hi = torch.stack([segment_lse(P[..., k], design.pin_net, M, gamma) for k in range(2)], -1)
    lo = -torch.stack([segment_lse(-P[..., k], design.pin_net, M, gamma) for k in range(2)], -1)
    return lo, hi


def spatial_maps(Y, design, G=GRID, smooth=False):
    """(B,7,G,G): macro occupancy, overlap excess, macro-pin density, RUDY, IO density, x-coord, y-coord.
    smooth=True replaces the ReLUs by softplus so the maps are C1 in Y (gradient-based refinement)."""
    act = (lambda v: F.softplus(v, beta=20.0)) if smooth else F.relu
    B, N, _ = Y.shape
    half = design.sizes.unsqueeze(0) / 2
    occ = raster_boxes(Y - half, Y + half, torch.ones(B, N, device=Y.device), G)
    overlap = act(occ - 1.0)
    P = pin_positions(Y, design)
    pin_den = raster_points(P, torch.ones(P.shape[:2], device=Y.device), G) * (G * G / max(P.shape[1], 1))
    lo, hi = net_bboxes(Y, design)
    min_ext = 2.0 / G
    w = (hi[..., 0] - lo[..., 0]).clamp_min(min_ext)
    h = (hi[..., 1] - lo[..., 1]).clamp_min(min_ext)
    rudy_w = (w + h) / (w * h) * (2.0 / G)        # RUDY demand per bin area
    rudy = raster_boxes(lo, hi, rudy_w, G)
    if design.io_xy.shape[0]:
        io = raster_points(design.io_xy.unsqueeze(0), torch.ones(1, design.io_xy.shape[0], device=Y.device), G)
        io = (io * (G * G / design.io_xy.shape[0])).expand(B, -1, -1)
    else:
        io = torch.zeros(B, G, G, device=Y.device)
    c = bin_centers(G, Y.device)
    xs = c.view(1, 1, G).expand(B, G, G)
    ys = c.view(1, G, 1).expand(B, G, G)
    maps = torch.stack([occ, overlap, pin_den, rudy, io, xs, ys], 1)
    scale = torch.tensor([1.0, 1.0, 0.2, 0.05, 0.2, 1.0, 1.0], device=Y.device).view(1, -1, 1, 1)
    return torch.cat([torch.log1p(act(maps[:, :5]) * scale[:, :5]), maps[:, 5:]], 1)


def node_features(Y, design, io_gamma=0.05):
    """(B,N,F) dynamic per-macro features: position, distance to the 4 canvas edges, soft distance to IO."""
    B, N, _ = Y.shape
    half = design.sizes.unsqueeze(0) / 2
    left = Y[..., 0] - half[..., 0] + 1.0
    right = 1.0 - (Y[..., 0] + half[..., 0])
    bottom = Y[..., 1] - half[..., 1] + 1.0
    top = 1.0 - (Y[..., 1] + half[..., 1])
    feats = [Y[..., 0], Y[..., 1], left, right, bottom, top]
    if design.io_xy.shape[0]:
        d = torch.cdist(Y, design.io_xy.unsqueeze(0).expand(B, -1, -1))              # (B,N,Q)
        feats.append(-io_gamma * torch.logsumexp(-d / io_gamma, dim=-1))             # soft min distance
    else:
        feats.append(torch.ones(B, N, device=Y.device))
    dyn = torch.stack(feats, -1)
    return torch.cat([dyn, design.macro_static.unsqueeze(0).expand(B, -1, -1)], -1)


def pin_features(Y, design):
    """(B,P,4): absolute pin position and offset."""
    P = pin_positions(Y, design)
    return torch.cat([P, design.pin_off.unsqueeze(0).expand(Y.shape[0], -1, -1)], -1)
