"""Differentiable std-cell proxy: where would the std cells go for a given macro layout?

Offline (per design, CPU, from scripts/ppa/dump_netlist.py's <design>.netlist.npz):
  1. graph of std cells from the full netlist: nets with <= CLIQUE_MAX endpoints -> clique (w = 1/(d-1)),
     larger nets -> star from the driver (w = 1), nets with > GLOBAL_DEG endpoints or clock/reset-like
     names -> dropped (clock, reset, scan enable);
  2. multilevel heavy-edge matching (area-capped) coarsens ~10^5 cells into K <= K_TARGET clusters;
  3. quadratic placement of the clusters with macro pins and IO pins as anchors (this is OpenROAD's own
     "InitialPlace" model, without density spreading). Its solution is LINEAR in the macro centers:
         X_clusters = A_M @ Y_macros + c        (A_M: K x N, c: K x 2, both precomputed)
  4. contracted nets (endpoints = clusters / macro pins / IO pins, deduplicated with multiplicity weights)
     for full-netlist HPWL and RUDY.
Online (torch, any batch of macro layouts Y (B,N,2), normalized canvas [-1,1]^2), all smooth in Y:
  cluster positions = quadratic placement + SimPL-style spreading (growing anchors to density-flow targets) in which macros are density
  blockages (StdCellProxy.spread), then
  maps    (B,4,G,G): cluster area density, flip-flop density, full-netlist RUDY, macro-net RUDY
  macro   (B,N,3)  : log total connection weight to std cells, weighted mean / soft-max L1 distance to them
  scalars (B,7)    : log HPWL (all nets / nets touching macros / IO nets), macro-cell pull, soft-max macro-cell
                     distance, residual density overfill, log HPWL of the unspread quadratic placement
Scalars are normalized per design with median/IQR over reference layouts (calibrate(); label free).
features_at(Y, Xc) evaluates the same features at GIVEN cluster positions (e.g. the real OpenROAD cluster
centroids of stdcell_real.py, or a learned correction of the proxy positions).
"""
import argparse
import contextlib
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from . import features as feat

CLIQUE_MAX = 32
GLOBAL_DEG = 2000
K_TARGET = 4096
MAX_RUDY_NETS = 40000


# ----------------------------------------------------------------------------------------------- build
def _endpoints(z):
    """Per net entry: endpoint id in [cells | macro pins | io pins] and the derived tables."""
    kind, ref = z["node_kind"], z["node_ref"]
    start, flat = z["net_start"].astype(np.int64), z["net_pins"].astype(np.int64)
    pnode = z["pin_node"][flat]
    knd, rf = kind[pnode], ref[pnode].astype(np.int64)
    C = int((kind == 0).sum())
    net = np.repeat(np.arange(len(start) - 1), np.diff(start))
    is_mp = knd == 1
    Pm = int(is_mp.sum())
    ep = np.full(len(flat), -1, np.int64)
    ep[knd == 0] = rf[knd == 0]
    ep[is_mp] = C + np.arange(Pm)
    ep[knd == 2] = C + Pm + rf[knd == 2]
    mp_macro = rf[is_mp]
    mp_off = z["pin_off"][flat[is_mp]].astype(np.float32)
    pdir = z["pin_dir"][flat]
    return net, ep, knd, pdir, C, Pm, mp_macro, mp_off


def _dedupe_entries(net, ep, knd, pdir):
    """Drop ignored nodes and repeated (net, cell) entries (multi-pin cells)."""
    keep = ep >= 0
    net, ep, knd, pdir = net[keep], ep[keep], knd[keep], pdir[keep]
    order = np.lexsort((-(pdir == 1).astype(np.int8), ep, net))     # outputs first within (net, ep)
    net, ep, knd, pdir = net[order], ep[order], knd[order], pdir[order]
    first = np.r_[True, (net[1:] != net[:-1]) | (ep[1:] != ep[:-1])]
    return net[first], ep[first], knd[first], pdir[first]


def _pairs(net, ep, knd, pdir, n_nets, net_drop):
    """Clique / star decomposition -> (u, v, w) endpoint pairs."""
    deg = np.bincount(net, minlength=n_nets)
    ok = ~net_drop[net] & (deg[net] >= 2) & (deg[net] <= GLOBAL_DEG)
    net, ep, knd, pdir = net[ok], ep[ok], knd[ok], pdir[ok]
    d = deg[net]
    us, vs, ws = [], [], []
    # cliques, grouped by degree
    for k in range(2, CLIQUE_MAX + 1):
        m = d == k
        if not m.any():
            continue
        e = ep[m].reshape(-1, k)
        iu, ju = np.triu_indices(k, 1)
        us.append(e[:, iu].ravel()); vs.append(e[:, ju].ravel())
        ws.append(np.full(e.shape[0] * len(iu), 1.0 / (k - 1)))
    # stars from the driver (first output pin, else first entry)
    m = d > CLIQUE_MAX
    if m.any():
        n2, e2, o2 = net[m], ep[m], (pdir[m] == 1)
        first = np.r_[True, n2[1:] != n2[:-1]]
        seg = np.cumsum(first) - 1
        drv_pos = np.full(seg.max() + 1, -1)
        cand = np.where(o2)[0]
        drv_pos[seg[cand[::-1]]] = cand[::-1]                       # first output per net
        none = drv_pos < 0
        drv_pos[none] = np.where(first)[0][none]
        drv = e2[drv_pos[seg]]
        sel = e2 != drv
        us.append(drv[sel]); vs.append(e2[sel]); ws.append(np.ones(sel.sum()))
    return np.concatenate(us), np.concatenate(vs), np.concatenate(ws)


def _coarsen(A, area, k_target, log):
    """Area-capped heavy-edge mutual matching until <= k_target nodes. Returns assignment (n,) -> [0,K)."""
    n = A.shape[0]
    assign = np.arange(n)
    cur, cur_area = A.tocsr(), area.astype(np.float64).copy()
    cap = 4.0 * area.sum() / max(k_target, 1)
    level = 0
    while level < 80:
        # only nodes with edges can still merge: stop when (mergeable + frozen) <= k_target, or the mergeable part
        # alone is small (anchored-only cells / finished components are frozen and must not drive over-coarsening)
        active = np.diff(cur.indptr) > 0
        n_act, n_frozen = int(active.sum()), int((~active).sum())
        if n_act + n_frozen <= k_target or n_act <= max(k_target - n_frozen, k_target // 4):
            break
        level += 1
        coo = cur.tocoo()
        r, c, w = coo.row, coo.col, coo.data
        m = (r != c) & (cur_area[r] + cur_area[c] <= cap)
        r, c, w = r[m], c[m], w[m]
        if len(r) == 0:
            if cap > 64 * area.sum() / max(k_target, 1):
                break
            cap *= 2
            continue
        score = w / (cur_area[r] + cur_area[c] + 1e-12)
        order = np.lexsort((-score, r))
        r, c = r[order], c[order]
        first = np.r_[True, r[1:] != r[:-1]]
        nb = cur.shape[0]
        best = np.full(nb, -1)
        best[r[first]] = c[first]
        i = np.arange(nb)
        mutual = (best >= 0) & (best[np.maximum(best, 0)] == i) & (i < best)
        n_pairs = int(mutual.sum())
        if n_pairs < 0.02 * n_act:
            if cap > 64 * area.sum() / max(k_target, 1):
                break
            cap *= 2                                                   # matching stalled on the area cap
            continue
        target = np.arange(nb)
        target[best[mutual]] = i[mutual]                               # partner -> representative
        uniq, new_id = np.unique(target, return_inverse=True)
        P = sp.csr_matrix((np.ones(nb), (np.arange(nb), new_id)), shape=(nb, len(uniq)))
        cur = (P.T @ cur @ P).tocsr()
        cur.setdiag(0); cur.eliminate_zeros()
        cur_area = P.T @ cur_area
        assign = new_id[assign]
        log(f"  level {level}: {nb} -> {len(uniq)} (cap {cap:.2e})")
    return assign


def build(npz_path, k_target=K_TARGET, eps_rel=1e-3, max_rudy_nets=MAX_RUDY_NETS, log=print, return_assign=False):
    """return_assign: also return out["assign"] (C,) netlist cell -> cluster (-1 = isolated cell); not part of
    the saved <design>.stdcell.pt (the build is deterministic, so it can be recomputed)."""
    t0 = time.time()
    z = dict(np.load(npz_path, allow_pickle=False))
    N = len(z["macro_names"])
    io_xy = z["io_xy"].astype(np.float32).reshape(-1, 2)
    Q = len(io_xy)
    net, ep, knd, pdir, C, Pm, mp_macro, mp_off = _endpoints(z)
    n_nets = len(z["net_start"]) - 1
    net, ep, knd, pdir = _dedupe_entries(net, ep, knd, pdir)
    deg = np.bincount(net, minlength=n_nets)
    net_drop = (deg > GLOBAL_DEG) | (z["net_clk"] & (deg > 64))
    u, v, w = _pairs(net, ep, knd, pdir, n_nets, net_drop)
    is_cell = lambda e: e < C
    is_mp = lambda e: (e >= C) & (e < C + Pm)
    is_io = lambda e: e >= C + Pm
    # orient pairs so that u is a cell whenever one end is a cell
    swap = ~is_cell(u) & is_cell(v)
    u, v = np.where(swap, v, u), np.where(swap, u, v)
    cc = is_cell(u) & is_cell(v)
    cm = is_cell(u) & is_mp(v)
    cq = is_cell(u) & is_io(v)
    A = sp.coo_matrix((np.r_[w[cc], w[cc]], (np.r_[u[cc], v[cc]], np.r_[v[cc], u[cc]])), shape=(C, C)).tocsr()
    area = z["cell_area"].astype(np.float64)
    connected = np.zeros(C, bool)
    connected[u[cc]] = True; connected[v[cc]] = True; connected[u[cm | cq]] = True
    log(f"{Path(npz_path).name}: cells {C} (connected {connected.sum()}), macro pins {Pm}, io {Q}, "
        f"nets {n_nets} (dropped {net_drop.sum()}), pairs {len(u)}")
    # coarsen connected cells; isolated cells (no signal nets left) only add a uniform background
    idx = np.where(connected)[0]
    A_sub = A[idx][:, idx]
    assign_sub = _coarsen(A_sub, area[idx], k_target, log)
    K = int(assign_sub.max()) + 1
    assign = np.full(C, -1)
    assign[idx] = assign_sub
    P = sp.csr_matrix((np.ones(len(idx)), (idx, assign_sub)), shape=(C, K))
    Ak = (P.T @ A @ P).tocsr()
    Ak.setdiag(0); Ak.eliminate_zeros()
    cl_area = P.T @ area
    cl_seq = P.T @ (area * z["cell_seq"].astype(np.float64))
    # anchors
    cu, cmac, cw = assign[u[cm]], mp_macro[v[cm] - C], w[cm]
    Bk = sp.coo_matrix((cw, (cu, cmac)), shape=(K, N)).toarray()
    off = mp_off[v[cm] - C]
    Ok = np.stack([np.bincount(cu, weights=cw * off[:, k], minlength=K) for k in range(2)], 1)
    qu, qv, qw = assign[u[cq]], v[cq] - C - Pm, w[cq]
    Rq = np.stack([np.bincount(qu, weights=qw * io_xy[qv, k], minlength=K) for k in range(2)], 1)
    wq = np.bincount(qu, weights=qw, minlength=K)
    D = np.asarray(Ak.sum(1)).ravel() + Bk.sum(1) + wq
    eps = eps_rel * float(np.median(D[D > 0])) if (D > 0).any() else 1.0
    L = (sp.diags(D + eps) - Ak).tocsc()
    sol = spla.splu(L).solve(np.ascontiguousarray(np.concatenate([Bk, Ok + Rq], 1)))
    A_M, c = sol[:, :N], sol[:, N:]
    if K > 2 * k_target:
        log(f"WARNING: coarsening stalled at K={K} (target {k_target})")
    # contracted nets for HPWL / RUDY: endpoints in [clusters K | macro pins Pm | io Q]
    keep = ~net_drop[net] & (deg[net] >= 2)
    n2, e2 = net[keep], ep[keep]
    cell = e2 < C
    e2 = np.where(cell, assign[np.minimum(e2, C - 1)], e2 - C + K)
    ok = ~(cell & (assign[np.minimum(ep[keep], C - 1)] < 0))
    n2, e2 = n2[ok], e2[ok]
    order = np.lexsort((e2, n2))
    n2, e2 = n2[order], e2[order]
    first = np.r_[True, (n2[1:] != n2[:-1]) | (e2[1:] != e2[:-1])]
    n2, e2 = n2[first], e2[first]
    starts = np.r_[0, np.where(n2[1:] != n2[:-1])[0] + 1, len(n2)]
    groups = {}
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a < 2:
            continue
        key = e2[a:b].tobytes()
        groups[key] = groups.get(key, 0) + 1
    ptr, eps_list, wts = [0], [], []
    for key, mult in groups.items():
        e = np.frombuffer(key, dtype=np.int64)
        eps_list.append(e); ptr.append(ptr[-1] + len(e)); wts.append(mult)
    net_ep = np.concatenate(eps_list) if eps_list else np.zeros(0, np.int64)
    net_ptr = np.asarray(ptr, np.int64)
    net_w = np.asarray(wts, np.float32)
    seg = np.repeat(np.arange(len(net_w)), np.diff(net_ptr))
    has_macro = np.bincount(seg, weights=((net_ep >= K) & (net_ep < K + Pm)).astype(float), minlength=len(net_w)) > 0
    has_io = np.bincount(seg, weights=(net_ep >= K + Pm).astype(float), minlength=len(net_w)) > 0
    # RUDY subset: every net touching a macro (the part macro moves change most) + a uniform random sample of
    # the rest; inverse-inclusion weights keep the RUDY map an unbiased estimate of the full-netlist one.
    rng = np.random.default_rng(0)
    p_incl = np.ones(len(net_w))
    others = np.where(~has_macro)[0]
    room = max_rudy_nets - int(has_macro.sum())
    if room < len(others):
        keep = rng.permutation(others)[:max(room, 0)]
        p_incl[others] = max(room, 0) / max(len(others), 1)
        sel = np.sort(np.concatenate([np.where(has_macro)[0], keep]))
    else:
        sel = np.arange(len(net_w))
    rudy_w = (net_w[sel] / p_incl[sel]).astype(np.float32)
    rudy_scale = float(rudy_w.sum() / max(net_w[sel].sum(), 1e-9))
    Lc = L.tocoo()
    out = {"K": K, "N": N, "A_M": torch.tensor(A_M, dtype=torch.float32), "c": torch.tensor(c, dtype=torch.float32),
           "L_idx": torch.tensor(np.stack([Lc.row, Lc.col]), dtype=torch.long),
           "L_val": torch.tensor(Lc.data, dtype=torch.float64),
           "r": torch.tensor(Ok + Rq, dtype=torch.float32),
           "area": torch.tensor(cl_area, dtype=torch.float32), "seq_area": torch.tensor(cl_seq, dtype=torch.float32),
           "Bk": torch.tensor(Bk, dtype=torch.float32), "mp_macro": torch.tensor(mp_macro, dtype=torch.long),
           "mp_off": torch.tensor(mp_off, dtype=torch.float32), "io_xy": torch.tensor(io_xy, dtype=torch.float32),
           "net_ptr": torch.tensor(net_ptr), "net_ep": torch.tensor(net_ep), "net_w": torch.tensor(net_w),
           "net_has_macro": torch.tensor(has_macro), "net_has_io": torch.tensor(has_io),
           "rudy_sel": torch.tensor(sel, dtype=torch.long), "rudy_scale": rudy_scale,
           "rudy_w": torch.tensor(rudy_w),
           "iso_area": float(area[~connected].sum()), "total_area": float(area.sum())}
    out["stats"] = {"cells": C, "connected": int(connected.sum()), "K": K, "macro_pins": Pm, "io": Q,
                    "nets_in": int(n_nets), "nets_dropped": int(net_drop.sum()), "contracted_nets": int(len(net_w)),
                    "contracted_endpoints": int(len(net_ep)), "rudy_nets": int(len(sel)), "rudy_scale": rudy_scale,
                    "A_M_rowsum_mean": float(A_M.sum(1).mean()), "eps": eps, "seconds": round(time.time() - t0, 1)}
    if return_assign:
        out["assign"] = assign
    return out


# ----------------------------------------------------------------------------------------------- online
def poisson_force(rho):
    """Electrostatic spreading field of a density excess map rho (B,G,G) [y][x] (Neumann boundary, via the
    mirrored FFT = DCT): solve lap(phi) = -rho, return E = -grad(phi) as (B,2,G,G) in bin units (x, y)."""
    B, G, _ = rho.shape
    rho = rho - rho.mean((1, 2), keepdim=True)
    ext = torch.cat([torch.cat([rho, rho.flip(-1)], -1), torch.cat([rho.flip(-2), rho.flip(-2).flip(-1)], -1)], -2)
    F = torch.fft.rfft2(ext)
    ky = torch.fft.fftfreq(2 * G, device=rho.device) * 2 * math.pi
    kx = torch.fft.rfftfreq(2 * G, device=rho.device) * 2 * math.pi
    k2 = ky.view(-1, 1) ** 2 + kx.view(1, -1) ** 2
    k2[0, 0] = 1.0
    Fp = F / k2
    Fp[..., 0, 0] = 0.0
    phi = torch.fft.irfft2(Fp, s=(2 * G, 2 * G))[:, :G, :G]
    gy, gx = torch.gradient(phi, dim=(1, 2))
    return -torch.stack([gx, gy], 1)


def splat(p, w, sigma, G):
    """Gaussian splat with per-point sigma: p (B,K,2), w (K,), sigma (K,) normalized units -> (B,G,G), mass w."""
    c = feat.bin_centers(G, p.device)
    gx = torch.exp(-0.5 * ((c.view(1, 1, -1) - p[..., 0:1]) / sigma.view(1, -1, 1)) ** 2)
    gy = torch.exp(-0.5 * ((c.view(1, 1, -1) - p[..., 1:2]) / sigma.view(1, -1, 1)) ** 2)
    gx = gx / gx.sum(-1, keepdim=True).clamp_min(1e-8)
    gy = gy / gy.sum(-1, keepdim=True).clamp_min(1e-8)
    return torch.einsum("bky,k,bkx->byx", gy, w, gx)


class StdCellProxy:
    """Holds one design's proxy tensors; features(Y) is differentiable in Y.

    Cluster positions: quadratic placement X0 = A_M Y + c (linear in the macro centers), then SimPL-style
    spreading with the same netlist springs L (anchor rhs B Y + r) and growing pseudo-anchors:
        Xhat_t  = X_t + clamp(kappa * E(X_t))                                  (density-flow look-ahead)
        X_{t+1} = argmin_X  springs(X) + alpha_t ||X - Xhat_t||^2_D  = (L + alpha_t D)^{-1} (B Y + r + alpha_t D Xhat_t)
    with D = diag(L), alpha_t = alpha0 * rate^t, E the electrostatic field (bin units) of
    (cluster density + macro occupancy - uniform target); the per-step look-ahead move is smoothly clamped to
    max_bins. (L + alpha D)^{-1} for any alpha comes from one eigendecomposition of D^-1/2 L D^-1/2.
    cfg td_rel > 0 replaces the uniform target by a density CAP (OpenROAD-like target density): only the excess
    over td = fill + (1 - fill) * td_rel pushes (fill = average utilization incl. macros), so cells are not drawn
    into empty space; sigma_scale scales the cluster footprint. Defaults reproduce the original proxy.
    Macros are density blockages, which is what makes the std-cell layout (and wirelength) depend on where
    macros sit beyond their pin pulls: OpenROAD's own quadratic initial placement alone tracks GRT_WL at
    tau ~0.1, its spread global placement at ~0.7.
    """

    TENSORS = ("A_M", "c", "r", "area", "seq_area", "Bk", "mp_macro", "mp_off", "io_xy", "net_ptr", "net_ep", "net_w",
               "net_has_macro", "net_has_io", "rudy_sel", "rudy_w", "seg", "wsum", "scal_med", "scal_scale", "sizes", "L_idx",
               "L_val", "sigma")
    N_MAPS, N_MACRO, N_SCALARS = 4, 3, 7
    SCALARS = ("hpwl_all", "hpwl_macro", "hpwl_io", "pull", "smax", "overflow", "hpwl_quad")
    DEFAULT_CFG = {"iters": 20, "alpha0": 0.01, "rate": 1.35, "kappa": 1.0, "max_bins": 4.0, "td_rel": 0.0,
                   "sigma_scale": 1.0}

    def __init__(self, d, grid=feat.GRID, **cfg):
        self.__dict__.update({k: v for k, v in d.items() if k != "stats"})
        self.stats = d.get("stats", {})
        self.seg = torch.repeat_interleave(torch.arange(len(self.net_w)), torch.diff(self.net_ptr))
        self.wsum = self.Bk.sum(0)
        self.scal_med = torch.zeros(self.N_SCALARS)
        self.scal_scale = torch.ones(self.N_SCALARS)
        self.sizes = None
        unknown = set(cfg) - set(self.DEFAULT_CFG)
        assert not unknown, f"unknown proxy cfg keys {unknown}"
        self.cfg = {**self.DEFAULT_CFG, **cfg, "grid": grid}
        # cluster footprint radius at ~70% local utilization, at least one bin
        self.sigma = torch.maximum(torch.sqrt(self.area / 0.7) / 2, torch.full_like(self.area, 2.0 / grid))
        self.target_cell = float(self.area.sum()) / 4.0
        self._eig = None
        self.crit = {}                    # variant -> (M,) static-criticality weight per contracted net (attach_crit)
        self.crit_med, self.crit_scale = {}, {}
        self._last_hp = None

    def attach_crit(self, d):
        """Static-criticality weights of the contracted nets (crit_proxy.py, <design>.crit.pt), one (M,) vector per
        variant: wc1 = sum of crit_q, wc4 = sum of crit_q^4, wc90 = number of original nets with crit_q >= 0.9 merged
        into the contracted net (crit_q = within-design quantile criticality of the pre-placement STA, design_cache_v0)."""
        assert int(d["M"]) == len(self.net_w), "crit weights do not match the contracted nets"
        dev = self.net_w.device
        self.crit = {k: v.float().to(dev) for k, v in d["weights"].items()}
        return self

    def _crit_raw(self, hp, variant):
        return torch.log((self.crit[variant] * hp).sum(1) + 1e-6)

    def crit_scalar(self, variant):
        """(B,1) normalized log criticality-weighted HPWL of the layout of the LAST raw_features() call."""
        raw = self._crit_raw(self._last_hp, variant)
        return ((raw - self.crit_med[variant]) / self.crit_scale[variant]).unsqueeze(1)

    def attach(self, sizes):
        """Macro sizes (N,2) in normalized units (full width/height, canvas = 2)."""
        self.sizes = sizes.float().to(self.A_M.device)
        return self

    def to(self, device):
        for k in self.TENSORS:
            v = getattr(self, k)
            if v is not None:
                setattr(self, k, v.to(device))
        self.crit = {k: v.to(device) for k, v in self.crit.items()}
        self.crit_med = {k: v.to(device) for k, v in self.crit_med.items()}
        self.crit_scale = {k: v.to(device) for k, v in self.crit_scale.items()}
        self._eig = None
        return self

    def _eigen(self):
        if self._eig is None:
            K = self.A_M.shape[0]
            L = torch.sparse_coo_tensor(self.L_idx, self.L_val, (K, K)).to_dense().double()
            dg = torch.diagonal(L).clone()
            s = dg.rsqrt()
            ev, U = torch.linalg.eigh(s.view(-1, 1) * L * s.view(1, -1))
            self._eig = (U.float(), ev.float(), s.float(), dg.float())
        return self._eig

    def _solve(self, rhs, alpha):
        """(L + alpha D)^{-1} rhs for rhs (B,K,2)."""
        U, ev, s, _ = self._eigen()
        w = torch.einsum("jk,bjd->bkd", U, rhs * s.view(1, -1, 1)) / (ev + alpha).view(1, -1, 1)
        return torch.einsum("kj,bjd->bkd", U, w) * s.view(1, -1, 1)

    def cluster_xy(self, Y):
        return torch.einsum("kn,bnd->bkd", self.A_M, Y) + self.c

    def macro_occupancy(self, Y, G):
        half = self.sizes.unsqueeze(0) / 2
        occ = feat.raster_boxes(Y - half, Y + half, torch.ones(Y.shape[:2], device=Y.device), G)
        return occ - torch.nn.functional.softplus(occ - 1.0, beta=20.0)     # smooth min(occ, 1): abutting macros

    def _sig(self):
        return self.sigma * self.cfg.get("sigma_scale", 1.0)

    def spread(self, Y, return_all=False):
        """Cluster positions after quadratic placement + spreading (B,K,2); with return_all also the
        quadratic positions, the final cluster density map and the macro occupancy map.
        Within cached(), repeated calls on the same tensor Y reuse one computation."""
        c = getattr(self, "_cache", None)
        if c is not None and c.get("spread_Y") is Y:
            out = c["spread"]
            return out if return_all else out[0]
        G, it = self.cfg["grid"], int(self.cfg["iters"])
        x = self.cluster_xy(Y)
        x0 = x
        occ = self.macro_occupancy(Y, G)
        bin_area = (2.0 / G) ** 2
        sig = self._sig()
        if it > 0:
            dg = self._eigen()[3].view(1, -1, 1)
            b = torch.einsum("kn,bnd->bkd", self.Bk, Y) + self.r
            fill = self.target_cell + occ.mean((1, 2), keepdim=True)
            td_rel = self.cfg.get("td_rel", 0.0)
            cap = isinstance(td_rel, torch.Tensor) or td_rel > 0
            td = fill + (1.0 - fill).clamp_min(0.0) * td_rel if cap else None
            kappa, mb = self.cfg["kappa"], self.cfg["max_bins"]
            for t in range(it):
                dens = splat(x, self.area, sig, G) / bin_area
                if cap:                                             # only the excess over the density cap pushes
                    E = poisson_force(torch.nn.functional.softplus((dens + occ - td) * 20.0) / 20.0)
                else:
                    E = poisson_force(dens + occ - fill)
                Ex = torch.nn.functional.grid_sample(E, x.unsqueeze(1), mode="bicubic", padding_mode="border",
                                                     align_corners=False)[:, :, 0].transpose(1, 2)   # (B,K,2) bins
                mv = kappa * Ex
                mv = mv / torch.sqrt(1.0 + (mv.norm(dim=-1, keepdim=True) / mb) ** 2)            # smooth clamp
                xhat = (x + mv * (2.0 / G)).clamp(-0.999, 0.999)
                alpha = self.cfg["alpha0"] * self.cfg["rate"] ** t
                x = self._solve(b + alpha * dg * xhat, alpha).clamp(-0.999, 0.999)
        dens = splat(x, self.area, sig, G) / bin_area
        out = (x, x0, dens, occ)
        if c is not None:
            c["spread_Y"], c["spread"] = Y, out
        return out if return_all else x

    def _net_boxes(self, Y, Xc, gamma=0.01):
        B = Y.shape[0]
        mp = Y[:, self.mp_macro] + self.mp_off
        table = torch.cat([Xc, mp, self.io_xy.unsqueeze(0).expand(B, -1, -1)], 1)
        P = table[:, self.net_ep]                                                    # (B,E,2)
        M = len(self.net_w)
        hi = torch.stack([feat.segment_lse(P[..., k], self.seg, M, gamma) for k in range(2)], -1)
        lo = -torch.stack([feat.segment_lse(-P[..., k], self.seg, M, gamma) for k in range(2)], -1)
        return lo, hi

    def raw_features(self, Y, Xc=None):
        """Features of layout Y with the clusters at the proxy's spread positions, or at the given Xc (B,K,2)."""
        G = self.cfg["grid"]
        B = Y.shape[0]
        bin_area = (2.0 / G) ** 2
        if Xc is None:
            Xc, X0, dens, occ = self.spread(Y, return_all=True)
        else:
            X0, occ = self.cluster_xy(Y), self.macro_occupancy(Y, G)
            dens = splat(Xc, self.area, self._sig(), G) / bin_area
        ff = splat(Xc, self.seq_area, self._sig(), G) / bin_area
        lo, hi = self._net_boxes(Y, Xc)
        ext = (hi - lo).clamp_min(1e-6)
        hp = ext.sum(-1)                                                             # (B,M)
        self._last_hp = hp                                                           # for crit_scalar()
        w = self.net_w
        min_ext = 2.0 / G
        sel = self.rudy_sel
        ws, hs = ext[:, sel, 0].clamp_min(min_ext), ext[:, sel, 1].clamp_min(min_ext)
        rwt = self.rudy_w if getattr(self, "rudy_w", None) is not None else w[sel] * self.rudy_scale
        rw = rwt * (ws + hs) / (ws * hs) * (2.0 / G)
        centers = feat.bin_centers(G, Y.device)
        ax = feat.soft_interval(lo[:, sel, 0], hi[:, sel, 0], centers, G)          # (B,R,G)
        ay = feat.soft_interval(lo[:, sel, 1], hi[:, sel, 1], centers, G)
        rudy = torch.einsum("bky,bk,bkx->byx", ay, rw, ax)
        rudy_m = torch.einsum("bky,bk,bkx->byx", ay, rw * self.net_has_macro[sel].float(), ax)
        maps = torch.stack([torch.log1p(dens), torch.log1p(ff * 5.0), torch.log1p(rudy * 0.05),
                            torch.log1p(rudy_m * 0.05)], 1)
        # macro <-> std-cell attachment
        dist = (Xc.unsqueeze(2) - Y.unsqueeze(1)).abs().sum(-1)                     # (B,K,N)
        wsum = self.wsum.clamp_min(1e-9)
        d_mean = (self.Bk.unsqueeze(0) * dist).sum(1) / wsum                         # (B,N)
        g = 0.05
        # unconnected clusters must not enter the weighted soft-max: -1e4 (finite, so wsum == 0 rows stay NaN-free)
        logw = torch.where(self.Bk > 0, torch.log(self.Bk.clamp_min(1e-38)), torch.full_like(self.Bk, -1e4)).unsqueeze(0)
        d_soft = g * torch.logsumexp(dist / g + logw, 1) - g * torch.log(wsum)       # weighted soft-max
        has = (self.wsum > 0).float()
        macro = torch.stack([torch.log1p(self.wsum).expand(B, -1), d_mean * has, d_soft * has], -1)
        tot = (w * hp).sum(1)
        hm = (w * hp * self.net_has_macro.float()).sum(1)
        hio = (w * hp * self.net_has_io.float()).sum(1)
        pull = (self.wsum * d_mean).sum(1)
        smax = g * torch.logsumexp(d_soft / g + torch.log(has.clamp_min(1e-30)), 1)
        over = torch.nn.functional.softplus((dens + occ - 1.0) * 10.0).mean((1, 2)) / 10.0   # residual overfill
        lo0, hi0 = self._net_boxes(Y, X0)
        hq = (w * (hi0 - lo0).clamp_min(1e-6).sum(-1)).sum(1)
        scal = torch.stack([torch.log(tot + 1e-6), torch.log(hm + 1e-6), torch.log(hio + 1e-6),
                            torch.log(pull + 1e-6), smax, over, torch.log(hq + 1e-6)], -1)
        return maps, macro, scal

    @contextlib.contextmanager
    def cached(self):
        """Within this context, repeated features(Y) calls on the SAME tensor object Y (e.g. every member of an
        ensemble scoring one layout) reuse one computation; the graph is shared, so gradients still accumulate.
        The cache holds a reference to Y, so an id can never be recycled for a different tensor."""
        self._cache = {}
        try:
            yield self
        finally:
            self._cache = None

    def features(self, Y):
        c = getattr(self, "_cache", None)
        if c is not None and c.get("Y") is Y:
            return c["out"]
        maps, macro, scal = self.raw_features(Y)
        out = (maps, macro, (scal - self.scal_med) / self.scal_scale)
        if c is not None:
            c["Y"], c["out"] = Y, out
        return out

    def features_at(self, Y, Xc):
        """Normalized features (maps, macro, scalars) with the clusters at Xc (B,K,2) instead of the proxy's."""
        maps, macro, scal = self.raw_features(Y, Xc)
        return maps, macro, (scal - self.scal_med) / self.scal_scale

    @torch.no_grad()
    def calibrate(self, Y_ref, chunk=16):
        """Per-design robust normalization of the scalar features over reference layouts (no labels)."""
        parts, cparts = [], []
        for i in range(0, len(Y_ref), chunk):
            parts.append(self.raw_features(Y_ref[i:i + chunk])[2])
            if self.crit:
                cparts.append({v: self._crit_raw(self._last_hp, v) for v in self.crit})
        s = torch.cat(parts)
        for v in self.crit:
            cv = torch.cat([c[v] for c in cparts])
            m_ = cv.median()
            q_ = (torch.quantile(cv, 0.75) - torch.quantile(cv, 0.25)) / 1.349
            self.crit_med[v], self.crit_scale[v] = m_, torch.where(q_ > 1e-6, q_, torch.ones_like(q_))
        med = s.median(0).values
        iqr = (torch.quantile(s, 0.75, 0) - torch.quantile(s, 0.25, 0)) / 1.349
        self.scal_med = med
        self.scal_scale = torch.where(iqr > 1e-6 * med.abs().clamp_min(1.0), iqr, torch.ones_like(iqr))
        return s


def load_proxy(path, **cfg):
    return StdCellProxy(torch.load(path, weights_only=False), **cfg)


def main():
    ap = argparse.ArgumentParser(description="build <design>.stdcell.pt from <design>.netlist.npz")
    ap.add_argument("--netlist", required=True, help="dir with <design>.netlist.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--designs", default="")
    ap.add_argument("--k", type=int, default=K_TARGET)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.netlist).glob("*.netlist.npz"))
    want = set(filter(None, a.designs.split(",")))
    failed = []
    for p in paths:
        design = p.name[: -len(".netlist.npz")]
        if want and design not in want:
            continue
        try:
            d = build(p, a.k)
        except Exception as exc:                       # keep building the other designs, fail at the end
            print(f"BUILD_FAILED {design}: {exc!r}", flush=True)
            failed.append(design)
            continue
        tmp = out / f"{design}.stdcell.pt.tmp"
        torch.save(d, tmp)
        tmp.replace(out / f"{design}.stdcell.pt")
        print(design, json.dumps(d["stats"]), flush=True)
    if failed:
        raise SystemExit(f"std-cell proxy build failed for {failed}")


if __name__ == "__main__":
    main()
