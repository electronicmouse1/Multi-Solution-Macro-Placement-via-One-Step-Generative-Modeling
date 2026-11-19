"""Train a Selection, Refine, Refine-student, macro-only Selection or ridge predictor on LEGAL layouts (final_xy).

python -m ppa_surrogate.train2 --model refine|student|selection|ridge|macro --data D/dataset --static D/design_static \
    --stdcell D/stdcell --out RUN [--val-designs vga_lcd] [--within-frac 0.25] [--steps 6000] [--seed 0]
    [--real D/stdcell_real]     (student: privileged real std-cell targets, training only)

Training designs default to the 14 training circuits minus --val-designs; the six test designs are never
used for training. Losses (per-design z-scored costs, batches within one design):
  Huber regression + pairwise ranking (pairs closer than the label-noise threshold tau_h are ignored;
  tau_h = sqrt(2) * flow-noise sigma_h estimated from jitter pairs, --tau auto)
  + family loss on legal-space perturbations (parent vs child difference, level >= --fam-min-level um)
  + [refine, macro] stage-A supervision: placement-stage scalars (--aux-w) and maps (--map-w)
  + [selection] optional congestion-map auxiliary (--map-w)
  + [student] real std-cell supervision (--real): per contracted net, the HPWL of the corrected clusters vs the
    real placement (--net-w; local structure, invariant to where a connected group sits), corrected cluster
    positions vs the real centroids (--pos-w, small: B1 showed absolute cluster positions are close to
    unpredictable from the macro layout, fitting them collapses the layout), oracle proxy scalars = proxy features
    at the real centroids (--oracle-w), corrected macro-to-cell distances (--macro-w), real cell RUDY (--rudy-w).
Model selection: mean within-design Kendall tau over the primary heads (wl, wns, tns, ovf) on the
validation samples (iMF seeds of --val-designs, or held-out iMF seeds with --within-frac).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import (PLACE_NAMES, Z_CLIP_LOSS, DesignBatchSampler, attach_real, estimate_noise, families, load_all,
                   robust_z, soft_clip, subset, within_split)
from .stdcell import StdCellProxy
from .losses import map_loss
from .metrics import ranking_metrics
from .models2 import RidgePredictor, build_model, make_obs

PRIMARY = ("wl", "wns", "tns", "ovf")
DEFAULT_TARGETS = ("wl", "wns", "tns", "ovf", "gcell", "pwr")
ALL_TRAIN = ("ariane81", "bp68", "bp_be12", "bp_fe38", "bp_multi57", "bp_multi", "swerv_wrapper43", "vga_lcd",
             "dft68", "or1200", "mor1kx", "ethernet", "VeriGPU", "isa_npu")
TEST = ("ariane133", "ariane136", "bp", "bp_be", "bp_fe", "swerv_wrapper")


def xy_of(d, input_mode):
    return d.raw_xy if input_mode == "raw" else d.final_xy


def forward(model, d, idx, input_mode, device, want_maps=False):
    it = torch.as_tensor(idx, dtype=torch.long, device=d.final_xy.device)
    X = xy_of(d, input_mode)[it]
    obs = make_obs(d, idx, device) if model.kind in ("selection", "ridge") else None
    return model(X, d, obs=obs, want_maps=want_maps)


@torch.no_grad()
def predict(model, d, input_mode="final", chunk=32):
    device = d.final_xy.device
    outs = [forward(model, d, list(range(i, min(i + chunk, d.n_samples))), input_mode, device)["out"]
            for i in range(0, d.n_samples, chunk)]
    return torch.cat(outs).cpu()


STAGE_A_FEATS = ("dp_hpwl", "gp_final_hpwl", "pl_wns", "pl_tns")
STUDENT_DIAG = ("C:hpwl", "C:pos_proxy_bins", "C:pos_student_bins")


@torch.no_grad()
def student_diag(model, d, sel, chunk=16):
    """Student: Kendall of its full-netlist HPWL estimate vs GRT_WL, and (with real data) the cluster position
    error (area-weighted RMS, bins) of the proxy and of the student's corrected clusters."""
    hh, ep, es, ww = [], [], [], []
    real = "real_has" in d.extra
    G = d.proxy.cfg["grid"]
    for i in range(0, d.n_samples, chunk):
        r = model(d.final_xy[i:i + chunk], d)
        hh.append(r["hpwl_hat"].cpu())
        if real and "xc" in r:
            xr = d.extra["real_xc"][i:i + chunk].to(r["xc"].device)
            w = d.proxy.area.view(1, -1) * d.extra["real_valid"][i:i + chunk].to(r["xc"].device).float()
            ep.append((((r["xc_proxy"] - xr) * G / 2) ** 2).sum(-1).mul(w).sum(1).cpu())
            es.append((((r["xc"] - xr) * G / 2) ** 2).sum(-1).mul(w).sum(1).cpu())
            ww.append(w.sum(1).cpu())
    out = {}
    j = 0                                                       # targets[0] is wl in every training config
    m = ranking_metrics(torch.cat(hh).numpy()[sel], d.y[:, j].cpu().numpy()[sel])
    if m:
        out["C:hpwl"] = m["kendall"]
    if ep:
        has = d.extra["real_has"].numpy() & sel
        w = torch.cat(ww).numpy()[has]
        if w.sum() > 0:
            out["C:pos_proxy_bins"] = float(np.sqrt(torch.cat(ep).numpy()[has].sum() / w.sum()))
            out["C:pos_student_bins"] = float(np.sqrt(torch.cat(es).numpy()[has].sum() / w.sum()))
    return out


@torch.no_grad()
def predict_place(model, d, chunk=32):
    outs = [model(d.final_xy[i:i + chunk], d)["place"] for i in range(0, d.n_samples, chunk)]
    return torch.cat(outs).cpu()


def evaluate(model, designs, input_mode, targets, only_imf=True):
    model.eval()
    res = {}
    for name, d in designs.items():
        p = predict(model, d, input_mode)
        sel = np.array([s == "imf" for s in d.source]) if only_imf else ~d.noise_only.cpu().numpy()
        res[name] = {}
        for j, t in enumerate(targets):
            if bool(d.mask[j]):
                res[name][t] = ranking_metrics(p[sel, j].numpy(), d.y[sel, j].cpu().numpy())
        if getattr(model, "kind", "") == "student":
            res[name].update(student_diag(model, d, sel))
        if getattr(model, "kind", "") in ("refine", "macro", "student") and d.place_z is not None:
            # stage A: can the model (no std-cell placement input) predict what OpenROAD's placement shows?
            pp = predict_place(model, d).numpy()
            valid = d.place_valid.cpu().numpy() & d.place_mask.cpu().numpy()[None]
            zt = d.place_z.cpu().numpy()
            for f in STAGE_A_FEATS:
                k = PLACE_NAMES.index(f)
                m = sel & valid[:, k]
                if m.sum() >= 8:
                    res[name]["A:" + f] = ranking_metrics(pp[m, k], zt[m, k])
    model.train()
    return res


def summarize(res, heads=PRIMARY):
    out = {}
    for f in STAGE_A_FEATS:                      # stage-A diagnostics (refine models), not part of the score
        v = [r["A:" + f]["kendall"] for r in res.values() if "A:" + f in r and r["A:" + f]]
        if v:
            out["A:" + f] = float(np.mean(v))
    for k in STUDENT_DIAG:                       # student diagnostics, not part of the score
        v = [r[k] for r in res.values() if r.get(k) is not None]
        if v:
            out[k] = float(np.mean(v))
    for h in heads:
        v = [r[h]["kendall"] for r in res.values() if h in r and r[h]]
        out[h] = float(np.mean(v)) if v else float("nan")
    vals = [out[h] for h in heads if np.isfinite(out[h])]
    out["score"] = float(np.mean(vals)) if vals else float("nan")
    return out


# ------------------------------------------------------------------------------------------------- losses
def sample_mask(d, it):
    return d.mask.unsqueeze(0) & ~torch.isnan(d.y[it])


def huber(pred, z, m):
    return F.huber_loss(pred[m], z[m], delta=1.0) if m.any() else pred.sum() * 0.0


def rank_loss(pred, z, m, tau, cap=2.0, uniform=False, per_head=False):
    """Within-batch pairwise logistic ranking per head; pairs with |dz| <= tau_h are treated as ties.
    uniform: a_ij = 1 instead of |dz| (capped); per_head: mean over the pairs of each head, then over the heads."""
    if pred.shape[0] < 2 or not m.any():
        return pred.sum() * 0.0
    dp = pred.unsqueeze(1) - pred.unsqueeze(0)
    dt = z.unsqueeze(1) - z.unsqueeze(0)
    mm = m.unsqueeze(1) & m.unsqueeze(0)
    iu = torch.triu(torch.ones(pred.shape[0], pred.shape[0], device=pred.device, dtype=torch.bool), 1).unsqueeze(-1)
    w = ((dt.abs() > tau.view(1, 1, -1)) & mm & iu).float()
    if not uniform:
        w = w * dt.abs().clamp(max=cap)
    loss = F.softplus(-torch.sign(dt) * dp) * w
    if per_head:
        n = w.sum((0, 1))
        has = n > 0
        return (loss.sum((0, 1))[has] / n[has]).mean() if bool(has.any()) else pred.sum() * 0.0
    return loss.sum() / w.sum().clamp_min(1.0)


def family_loss(pred, z, m, pairs):
    """Huber on (pred_child - pred_parent) - (z_child - z_parent) for (child, parent) batch positions."""
    if not pairs:
        return pred.sum() * 0.0
    c = torch.tensor([p[0] for p in pairs], device=pred.device)
    p = torch.tensor([p[1] for p in pairs], device=pred.device)
    mm = m[c] & m[p]
    if not mm.any():
        return pred.sum() * 0.0
    return F.huber_loss((pred[c] - pred[p])[mm], (z[c] - z[p])[mm], delta=1.0)


def place_aux_loss(pred_place, d, it):
    m = d.place_mask.unsqueeze(0) & d.place_valid[it]
    return huber(pred_place, d.place_z[it], m)


def map_supervision(res_maps, d, it, targets, kind, dev):
    """Stage-A / auxiliary map loss on the samples of the batch that have OpenROAD maps."""
    ok = d.extra["maps_ok"][it.cpu()]
    if not ok.any():
        return None
    sel = torch.nonzero(ok).flatten()
    mt = d.extra["maps"][it.cpu()[sel]].to(dev).float()
    pm = res_maps[sel.to(dev)][:, :3]                 # student: channel 3 is the real RUDY (student_losses)
    use_cong = bool(d.mask[targets.index("ovf")]) if "ovf" in targets else False
    if kind == "selection":          # density is an input there: supervise congestion only
        return F.mse_loss(pm[:, :2], mt[:, :2]) if use_cong else None
    return map_loss(pm, mt, use_cong)


def student_losses(res, d, it, dev):
    """Privileged real std-cell supervision of the student on the batch samples that have real data."""
    zero = res["out"].sum() * 0.0
    keys = ("pos", "oracle", "macro", "rudy", "net")
    if "real_has" not in d.extra:
        return dict.fromkeys(keys, zero)
    itc = it.cpu()
    has = d.extra["real_has"][itc]
    if not has.any():
        return dict.fromkeys(keys, zero)
    sel = torch.nonzero(has).flatten().to(dev)
    osc = d.extra["real_oscal"][itc].to(dev)[sel]
    out = {"oracle": F.huber_loss(res["oracle"][sel], osc, delta=1.0),
           "rudy": F.mse_loss(res["maps"][sel, 3], d.extra["real_rudy"][itc].to(dev)[sel].float()),
           "pos": zero, "macro": zero, "net": zero}
    if "xc" in res:
        G = d.proxy.cfg["grid"]
        xr = d.extra["real_xc"][itc].to(dev)[sel]
        vr = d.extra["real_valid"][itc].to(dev)[sel]
        w = d.proxy.area.view(1, -1) * vr.float()
        # per-net HPWL of the corrected clusters vs the real ones (clusters without placed cells: proxy position)
        Y = d.final_xy[it][sel]
        xr_f = torch.where(vr.unsqueeze(-1), xr, res["xc_proxy"][sel].detach())
        lo_c, hi_c = d.proxy._net_boxes(Y, res["xc"][sel])
        with torch.no_grad():
            lo_r, hi_r = d.proxy._net_boxes(Y, xr_f)
        hc = (hi_c - lo_c).clamp_min(1e-4).sum(-1)
        hr = (hi_r - lo_r).clamp_min(1e-4).sum(-1)
        wn = d.proxy.net_w.view(1, -1)
        ln = F.huber_loss(torch.log(hc), torch.log(hr), delta=1.0, reduction="none")
        out["net"] = (ln * wn).sum() / (wn.sum() * len(sel))
        diff = (res["xc"][sel] - xr) * (G / 2.0)                                   # bins
        hub = F.huber_loss(diff, torch.zeros_like(diff), delta=1.0, reduction="none").sum(-1)
        out["pos"] = (hub * w).sum() / w.sum().clamp_min(1e-12)
        out["oracle"] = out["oracle"] + F.huber_loss(res["scal"][sel], osc, delta=1.0)
        hm = d.proxy.wsum > 0                                                      # macros with std-cell pins
        if bool(hm.any()):
            om = d.extra["real_omacro"][itc].to(dev)[sel]
            out["macro"] = F.huber_loss(10.0 * res["macro"][sel][:, hm, 1:], 10.0 * om[:, hm, 1:], delta=1.0)
    return out


# ------------------------------------------------------------------------------------------------- main
def fit_ridge(model, train, targets, lam):
    Xs, Zs, Ms = [], [], []
    for d in train.values():
        idx = [i for i in range(d.n_samples) if bool(d.place_ok[i])]
        if not idx:
            continue
        it = torch.tensor(idx, device=d.final_xy.device)
        obs = make_obs(d, idx, "cpu")
        Xs.append(obs["place"] * obs["mask"]); Zs.append(d.z[it].cpu()); Ms.append(sample_mask(d, it).cpu())
    model.fit(torch.cat(Xs), torch.cat(Zs), torch.cat(Ms), lam)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("refine", "student", "selection", "ridge", "macro"), required=True,
                    help="macro = Selection from the macro layout alone (no placement-stage inputs); "
                         "student = Refine student with real std-cell supervision (--real)")
    ap.add_argument("--real", default="", help="student: dir with <design>.real.pt (ppa_surrogate.stdcell_real)")
    ap.add_argument("--pos-w", type=float, default=0.1, help="student: corrected cluster positions vs real")
    ap.add_argument("--net-w", type=float, default=1.0, help="student: per-net HPWL of corrected vs real clusters")
    ap.add_argument("--oracle-w", type=float, default=0.5, help="student: proxy scalars at the real centroids")
    ap.add_argument("--macro-w", type=float, default=0.25, help="student: corrected macro-to-cell distances")
    ap.add_argument("--rudy-w", type=float, default=0.2, help="student: real cell-level RUDY map")
    ap.add_argument("--no-corr", action="store_true", help="student without the cluster-correction module")
    ap.add_argument("--corr-max", type=float, default=1.0, help="student: max correction (normalized units)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--stdcell", default="", help="dir with <design>.stdcell.pt ('' = no std-cell proxy)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-designs", default=",".join(ALL_TRAIN))
    ap.add_argument("--val-designs", default="")
    ap.add_argument("--sample-frac", type=float, default=1.0,
                    help="< 1: train on this random fraction of each design's sample groups (data-scaling check)")
    ap.add_argument("--within-frac", type=float, default=0.0,
                    help="> 0: also hold out this fraction of iMF seeds (+ their children) of every training design")
    ap.add_argument("--input", choices=("raw", "final"), default="final")
    ap.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--rank-w", type=float, default=1.0)
    ap.add_argument("--huber-w", type=float, default=1.0, help="weight of the Huber regression term")
    ap.add_argument("--rank-weight", choices=("abs", "uniform"), default="abs",
                    help="ranking pair weight: |dz| capped at 2 (legacy) or a_ij = 1")
    ap.add_argument("--rank-norm", choices=("joint", "per_head"), default="joint",
                    help="ranking loss averaged over all pairs of all heads (legacy) or per head, then over the heads")
    ap.add_argument("--mix-imf", type=float, default=0.0,
                    help="> 0: stratified batches, this fraction from iMF seeds and the rest from the other allowed sources")
    ap.add_argument("--exclude-sources", default="", help="comma list of sample sources never used for training")
    ap.add_argument("--ovf-degenerate-zero-frac", type=float, default=0.0,
                    help="> 0: mask the ovf head of designs whose iMF pool has zero overflow in at least this fraction")
    ap.add_argument("--dump-preds", action="store_true", help="write RUN/preds_last.pt (validation designs)")
    ap.add_argument("--fam-w", type=float, default=0.5)
    ap.add_argument("--fam-min-level", type=float, default=1.0, help="um; family loss only on moves >= this")
    ap.add_argument("--aux-w", type=float, default=0.5, help="refine / macro: placement-stage scalar supervision")
    ap.add_argument("--map-w", type=float, default=0.2, help="refine / macro: placement maps; selection: congestion aux")
    ap.add_argument("--tau", default="auto", help="'auto' (from jitter pairs) or a float in z units")
    ap.add_argument("--z-clip", type=float, default=Z_CLIP_LOSS, help="targets enter the losses as c*tanh(z/c)")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--cnn-width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--allow-missing-proxy", action="store_true")
    ap.add_argument("--prior-fixed", action="store_true",
                    help="selection: fixed physical prior (placement-stage observation per head) + learned residual")
    ap.add_argument("--proxy-cfg", default="", help="std-cell proxy spreading, e.g. iters=20,alpha0=0.01,rate=1.35,"
                    "kappa=1,max_bins=4 (defaults: stdcell.StdCellProxy.DEFAULT_CFG)")
    ap.add_argument("--ridge-lambda", type=float, default=0.1, help="relative to the mean squared feature")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--family-frac", type=float, default=0.5)
    ap.add_argument("--prior-crit", default="", help="macro --prior-fixed: timing heads use the criticality-weighted "
                    "proxy HPWL (variant wc1 | wc4 | wc90 of <design>.crit.pt next to the proxy files) as prior")
    ap.add_argument("--resid-w", type=float, default=0.0,
                    help="macro --prior-fixed: L2 penalty on the learned residual (keeps the model near the proxy prior)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    targets = tuple(a.targets.split(","))
    val_names = [d for d in a.val_designs.split(",") if d]
    assert not set(val_names) & set(TEST), "test designs are for evaluate2 only (no model selection on them)"
    train_names = [d for d in a.train_designs.split(",") if d and d not in val_names and d not in TEST]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_proxy = a.model != "ridge" and not a.no_proxy and bool(a.stdcell)
    # ';' also accepted as separator: PBS `qsub -v` splits variable lists on ','
    proxy_cfg = {k: float(v) for k, v in (kv.split("=") for kv in a.proxy_cfg.replace(";", ",").split(",") if kv)}
    t_load = time.time()
    designs = load_all(a.data, a.static, train_names + val_names, targets,
                       stdcell_dir=a.stdcell if use_proxy else None, proxy_cfg=proxy_cfg)
    for d in designs.values():
        d.to(dev)
    degenerate = {}
    if a.ovf_degenerate_zero_frac > 0 and "ovf" in targets:
        j_ovf = targets.index("ovf")
        for n_, d in designs.items():
            imf_ = torch.tensor([s == "imf" for s in d.source], device=d.y.device)
            y_ = d.y[imf_, j_ovf]
            y_ = y_[~torch.isnan(y_)]
            frac = float((y_ == 0).float().mean()) if len(y_) else 0.0
            if frac >= a.ovf_degenerate_zero_frac and bool(d.mask[j_ovf]):
                d.mask[j_ovf] = False
                degenerate[n_] = round(frac, 3)
        print(json.dumps({"ovf_head_masked": degenerate}), flush=True)
    real_n = {}
    if a.model == "student":
        assert a.real, "--model student needs --real (privileged real std-cell targets)"
        real_n = {n: attach_real(d, a.real) for n, d in designs.items()}
        no_real = [n for n in train_names if not real_n.get(n)]
        if no_real:
            raise SystemExit(f"real std-cell data missing for training designs {no_real}")
    missing_proxy = [n for n, d in designs.items() if use_proxy and d.proxy is None]
    if missing_proxy and not a.allow_missing_proxy:
        raise SystemExit(f"std-cell proxy missing for {missing_proxy} (zero inputs would be fed); "
                         "build them or pass --allow-missing-proxy")
    proxy_files = {n: d.proxy.source for n, d in designs.items() if d.proxy is not None}
    train = {n: designs[n] for n in train_names}
    val = {n: designs[n] for n in val_names}
    if a.within_frac > 0:
        for n in train_names:
            tr, va = within_split(designs[n], a.within_frac, seed=a.seed)
            d0 = designs[n]
            # re-normalize with the TRAINING iMF seeds only (held-out labels must not set median / IQR / mask)
            tr_imf = [i for i in tr if d0.source[i] == "imf"]
            if len(tr_imf) >= 8:
                import copy
                d0 = copy.copy(d0)
                z, m, st_ = robust_z(d0.y.cpu(), d0.y.cpu()[tr_imf])
                d0.z, d0.mask = z.to(dev), m.to(dev)
                d0.stats = {t: v for t, v in zip(targets, st_) if v is not None}
                if n in degenerate:                       # keep the degenerate-head mask of --ovf-degenerate-zero-frac
                    d0.mask[targets.index("ovf")] = False
            train[n] = subset(d0, tr)
            val[n + "@heldout"] = subset(d0, va)
    if a.sample_frac < 1.0:
        # data-scaling diagnostic: keep a random fraction of each training design's sample groups (an iMF seed
        # together with all its perturbed children; every other sample is its own group); validation untouched
        rng = np.random.default_rng(1000 + a.seed)
        for n, d in list(train.items()):
            id2i = {sid: i for i, sid in enumerate(d.ids)}
            group = [id2i.get(p, i) if p is not None and p in id2i else i for i, p in enumerate(d.parent)]
            heads_ = sorted(set(group))
            keep = set(rng.choice(heads_, size=max(1, int(round(a.sample_frac * len(heads_)))), replace=False).tolist())
            train[n] = subset(d, [i for i, g in enumerate(group) if g in keep])
        print(json.dumps({"sample_frac": a.sample_frac, "train_samples": {n: d.n_samples for n, d in train.items()}}),
              flush=True)
    sigma, sigma_per = estimate_noise(train, targets)
    if a.tau == "auto":
        tau = torch.where(torch.isnan(sigma), torch.full_like(sigma, 0.1), (math.sqrt(2) * sigma).clamp(0.05, 1.0))
    else:
        tau = torch.full((len(targets),), float(a.tau))
    tau = tau.to(dev)
    model_cfg = dict(n_targets=len(targets), n_place=len(PLACE_NAMES), hidden=a.hidden, layers=a.layers,
                     cnn_width=a.cnn_width, use_proxy=use_proxy, use_graph=not a.no_graph, dropout=a.dropout,
                     targets=list(targets), prior_fixed=a.prior_fixed, prior_crit=a.prior_crit, corr=not a.no_corr,
                     corr_max=a.corr_max)
    model = build_model(a.model, model_cfg).to(dev)
    model_cfg = model.cfg
    cfg = {**vars(a), "targets": targets, "train_names": train_names, "val_names": val_names, "kind": a.model,
           "model_cfg": model_cfg, "proxy_cfg": {**StdCellProxy.DEFAULT_CFG, **proxy_cfg} if use_proxy else proxy_cfg,
           "proxy_files": proxy_files, "use_proxy": use_proxy, "missing_proxy": missing_proxy,
           "noise_sigma_z": sigma.tolist(), "noise_sigma_per_design": {k: v.tolist() for k, v in sigma_per.items()},
           "tau": tau.tolist(), "place_features": PLACE_NAMES, "device": str(dev), "real_samples": real_n,
           "ovf_head_masked": degenerate, "head_mask": {n: d.mask.cpu().tolist() for n, d in designs.items()},
           "n_params": sum(p.numel() for p in model.parameters()), "load_seconds": round(time.time() - t_load, 1)}
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    print(json.dumps({k: cfg[k] for k in ("kind", "train_names", "val_names", "tau", "use_proxy", "missing_proxy",
                                          "n_params", "load_seconds")}), flush=True)

    if a.model == "ridge":
        fit_ridge(model, train, targets, a.ridge_lambda)
        rec = {"step": 0}
        if val:
            vres = evaluate(model, val, a.input, targets)
            rec["val"] = summarize(vres); rec["val_detail"] = vres
        (out / "history.json").write_text(json.dumps([rec], indent=1))
        for name in ("best.pt", "last.pt"):
            torch.save({"model": model.state_dict(), "cfg": cfg, "step": 0}, out / name)
        print(json.dumps({k: v for k, v in rec.items() if k != "val_detail"}), flush=True)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    # --input raw: legal-space samples have no raw layout (raw_xy := final_xy) -> keep them out of raw training
    skip_src = {s for s in a.exclude_sources.split(",") if s}
    excl = (lambda d, i: not bool(d.has_raw[i])) if a.input == "raw" else None
    if skip_src:
        base = excl
        excl = lambda d, i: (base is not None and base(d, i)) or d.source[i] in skip_src      # noqa: E731
    sampler = DesignBatchSampler(train, a.batch, family_frac=a.family_frac, seed=a.seed, exclude=excl,
                                 mix_imf=a.mix_imf if a.mix_imf > 0 else None)
    fam_pairs = {n: {c: p for c, p in families(d, min_level=a.fam_min_level)} for n, d in train.items()}
    hist, best, t0 = [], -1e9, time.time()
    keys = ("loss", "scalar", "rank", "fam", "aux", "map", "pos", "oracle", "macro", "rudy", "net")
    run = {k: 0.0 for k in keys}; run["n"] = 0
    for step in range(1, a.steps + 1):
        name, idx = sampler.sample()
        d = train[name]
        it = torch.tensor(idx, device=dev)
        want_maps = a.map_w > 0 and d.extra.get("maps") is not None
        res = forward(model, d, idx, a.input, dev, want_maps=want_maps)
        pred, z = res["out"], soft_clip(d.z[it], a.z_clip)          # heavy-tailed robust z: bounded targets
        m = sample_mask(d, it)
        pos = {i: k for k, i in enumerate(idx)}
        pairs = [(pos[c], pos[p]) for c, p in fam_pairs[name].items() if c in pos and p in pos]
        ls, lf = huber(pred, z, m), family_loss(pred, z, m, pairs)
        lr_ = rank_loss(pred, z, m, tau, uniform=a.rank_weight == "uniform", per_head=a.rank_norm == "per_head")
        loss = a.huber_w * ls + a.rank_w * lr_ + a.fam_w * lf
        la = lm = pred.sum() * 0.0
        sl = dict.fromkeys(("pos", "oracle", "macro", "rudy", "net"), pred.sum() * 0.0)
        if a.model == "student":
            sl = student_losses(res, d, it, dev)
            loss = loss + a.pos_w * sl["pos"] + a.oracle_w * sl["oracle"] + a.macro_w * sl["macro"] + \
                a.rudy_w * sl["rudy"] + a.net_w * sl["net"]
        if a.model in ("refine", "macro", "student") and a.aux_w > 0:
            la = place_aux_loss(res["place"], d, it)
            loss = loss + a.aux_w * la
        if a.resid_w > 0 and res.get("resid") is not None:
            loss = loss + a.resid_w * (res["resid"] ** 2).mean()
        if want_maps and res.get("maps") is not None:
            lm_ = map_supervision(res["maps"], d, it, targets, a.model, dev)
            if lm_ is not None:
                lm = lm_
                loss = loss + a.map_w * lm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in zip(keys, (loss, ls, lr_, lf, la, lm, sl["pos"], sl["oracle"], sl["macro"], sl["rudy"], sl["net"])):
            run[k] += float(v.detach())
        run["n"] += 1
        if step % a.eval_every == 0 or step == a.steps:
            rec = {"step": step, "time_s": round(time.time() - t0, 1), **{k: run[k] / max(run["n"], 1) for k in keys}}
            run = {k: 0.0 for k in keys}; run["n"] = 0
            if val:
                vres = evaluate(model, val, a.input, targets)
                rec["val"] = summarize(vres)
                rec["val_detail"] = vres
                score = rec["val"]["score"]
            else:
                score = step
            if np.isfinite(score) and score > best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "best.pt")
            hist.append(rec)
            (out / "history.json").write_text(json.dumps(hist, indent=1))
            print(json.dumps({k: v for k, v in rec.items() if k != "val_detail"}), flush=True)
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": a.steps}, out / "last.pt")
    if a.dump_preds:
        model.eval()
        torch.save({n: {"ids": d.ids, "source": d.source, "targets": list(targets), "pred": predict(model, d, a.input),
                        "y": d.y.cpu(), "mask": d.mask.cpu()} for n, d in {**val}.items()}, out / "preds_last.pt")
    if not (out / "best.pt").exists():
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": a.steps}, out / "best.pt")


if __name__ == "__main__":
    main()
