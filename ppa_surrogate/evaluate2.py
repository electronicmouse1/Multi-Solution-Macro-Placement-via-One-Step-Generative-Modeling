"""Evaluate Selection / Refine / ridge predictors (a checkpoint ensemble) on held-out designs.

python -m ppa_surrogate.evaluate2 --ckpts RUN/full_seed*/last.pt --data D/dataset --static D/design_static \
    --stdcell D/stdcell --designs ariane133,ariane136,bp,bp_be,bp_fe,swerv_wrapper --out eval.json [--grad-check]

1. Ranking, per design and head on the candidate pool (iMF seeds by default): Kendall / Spearman /
   pairwise accuracy / top-5 recall / regret of the predicted best, for the predictor and single-signal
   baselines: corrected macro HPWL (macro_hpwl), std-cell proxy HPWL (proxy_hpwl), and the placement-stage
   observation of the same head (place: dp_hpwl -> wl, pl_wns -> wns, pl_tns -> tns, pl_power -> pwr,
   dp_hpwl -> ovf/gcell; needs an OpenROAD floorplan + place run per candidate, like the Selection predictor).
2. Selection ("diverse objectives + overall good solutions"). For each preference w (wl, timing, cong,
   wl_timing, wl_cong, timing_cong, balanced; weights over available heads, renormalized) a scorer picks the
   candidate with the lowest scored w.z. Reported against the TRUE costs:
     regret      true w.z of the pick - best true w.z in the pool      (0 = oracle)
     gain        mean true w.z of the pool - true w.z of the pick      (> 0 = better than a random pick)
     percentile  fraction of the pool strictly better than the pick
     deltas      pick vs pool median: GRT_WL %, WNS / TNS (ns, + = better), overflow (log1p)
   and for the SET of distinct picks: hypervolume in true (WL, timing, congestion) z-space divided by the
   hypervolume of the whole pool, vs the mean over random sets of the same size.
3. Local response (designs with legal-space perturbation families, i.e. training circuits in LOCO folds):
   sign accuracy / correlation of predicted vs true child - parent changes; stage-A Kendall for Refine models.
4. --grad-check (refine / student models): autograd vs central finite differences of w.z_hat along random moves.
5. Refine students: their full-netlist HPWL estimate as a single-signal scorer (student_hpwl) and, with --real,
   the cluster position error of the proxy vs the student's corrected clusters against the real placement.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .data import PLACE_NAMES, PLACE_PRIOR, Z_CLIP_EVAL, attach_real, families, load_all, soft_clip
from .metrics import mean_over, ranking_metrics
from .models2 import build_model, make_obs
from .train2 import TEST, predict, student_diag

PREFS = {"wl": {"wl": 1.0}, "timing": {"wns": 0.5, "tns": 0.5}, "cong": {"ovf": 0.5, "gcell": 0.5},
         "wl_timing": {"wl": 0.5, "wns": 0.25, "tns": 0.25}, "wl_cong": {"wl": 0.5, "ovf": 0.25, "gcell": 0.25},
         "timing_cong": {"wns": 0.25, "tns": 0.25, "ovf": 0.25, "gcell": 0.25},
         "balanced": {"wl": 1 / 3, "wns": 1 / 6, "tns": 1 / 6, "ovf": 1 / 6, "gcell": 1 / 6}}
OBJ_DIMS = {"WL": ("wl",), "timing": ("wns", "tns"), "cong": ("ovf", "gcell")}
PLACE_FOR_HEAD = PLACE_PRIOR
SOURCES = ("predictor", "student_hpwl", "place", "proxy_hpwl", "macro_hpwl")


@torch.no_grad()
def predict_hpwl_hat(model, d, chunk=16):
    return torch.cat([model(d.final_xy[i:i + chunk], d)["hpwl_hat"].cpu() for i in range(0, d.n_samples, chunk)])


def load_models(ckpts, device):
    models, cfgs = [], []
    for c in ckpts:
        ck = torch.load(c, map_location=device, weights_only=False)
        cfg = ck["cfg"]
        if cfg["kind"] != "ridge" and ck.get("step") != cfg.get("steps"):
            raise SystemExit(f"{c}: checkpoint at step {ck.get('step')} of {cfg.get('steps')} (unfinished run?); "
                             "use the last.pt of finished runs")
        m = build_model(cfg["kind"], cfg["model_cfg"]).to(device)
        m.load_state_dict(ck["model"]); m.eval()
        models.append(m); cfgs.append(cfg)
    kinds = {c["kind"] for c in cfgs}
    tg = {tuple(c["targets"]) for c in cfgs}
    assert len(kinds) == 1 and len(tg) == 1, f"ensemble must share kind and targets: {kinds} {tg}"
    return models, cfgs


def pref_weights(pref, targets, mask):
    w = np.zeros(len(targets))
    for h, v in PREFS[pref].items():
        if h in targets and mask[targets.index(h)]:
            w[targets.index(h)] = v
    return w / w.sum() if w.sum() > 0 else None


def obj_matrix(Z, targets, mask):
    cols, names = [], []
    for name, heads in OBJ_DIMS.items():
        idx = [targets.index(h) for h in heads if h in targets and mask[targets.index(h)]]
        if idx:
            cols.append(Z[:, idx].mean(1)); names.append(name)
    return (np.stack(cols, 1) if cols else np.zeros((len(Z), 0))), names


def hypervolume(P, ref):
    """Exact dominated hypervolume (minimization) by recursive slicing; fine for <= 3 objectives, n <= ~100."""
    P = P[(P < ref).all(1)]
    if len(P) == 0:
        return 0.0
    if P.shape[1] == 1:
        return float(ref[0] - P[:, 0].min())
    order = np.argsort(P[:, -1], kind="stable")
    P = P[order]
    vol = 0.0
    for i in range(len(P)):
        nxt = P[i + 1, -1] if i + 1 < len(P) else ref[-1]
        if nxt > P[i, -1]:
            vol += hypervolume(P[: i + 1, :-1], ref[:-1]) * (nxt - P[i, -1])
    return vol


def zscore_within(v, sel):
    """Robust z of a single-signal scorer within the pool; missing values stay NaN."""
    v = np.asarray(v, float)
    ref = v[sel & ~np.isnan(v)]
    if len(ref) < 4:
        return np.full_like(v, np.nan)
    s = (np.percentile(ref, 75) - np.percentile(ref, 25)) / 1.349 or (np.std(ref) or 1.0)
    return (v - np.median(ref)) / s


def selection_eval(scores, d, sel, targets, rng, n_random=200):
    """scores (S,T) lower = better (z-like). Pool = samples where sel. Returns per-pref + set metrics.
    The SCORER combines heads with label-free weights (every head of the model, as at deployment); the TRUE
    objective uses only heads that vary on this design (d.mask). Missing labels stay NaN; a candidate with a
    missing score is never picked. All deltas are signed '+ = better than the pool median'."""
    Y = d.y.cpu().numpy()
    Ztrue = soft_clip(d.z.cpu().numpy(), Z_CLIP_EVAL)              # bounded: robust z is heavy tailed
    Ztrue[np.isnan(Y)] = np.nan
    mask = d.mask.cpu().numpy()
    S_ = np.where(np.isnan(scores), 1e6, scores)
    pool = np.where(sel)[0]
    out, picks = {}, []
    for pref in PREFS:
        w_true = pref_weights(pref, targets, mask)
        w_score = pref_weights(pref, targets, np.ones(len(targets), bool))
        if w_true is None or w_score is None:
            continue
        cols = np.where(w_true > 0)[0]
        ok = pool[~np.isnan(Ztrue[pool][:, cols]).any(1)]
        if len(ok) < 4:
            continue
        true_obj = Ztrue[ok][:, cols] @ w_true[cols]
        sc_nan = np.where(w_score > 0, scores[ok], 0.0) @ w_score          # NaN where a weighted score is missing
        if np.isnan(sc_nan).all() or np.nanmax(sc_nan) - np.nanmin(sc_nan) <= 1e-12:
            out[pref] = {"uninformative": True}                              # no information: no pick
            continue
        sc = np.where(np.isnan(sc_nan), np.inf, sc_nan)
        tied = np.flatnonzero(sc <= sc.min() + 1e-12)
        pick_local = int(tied[0])
        pick = int(ok[pick_local])
        if len(tied) == 1:
            picks.append(pick)
        t_obj = true_obj[tied]                                               # ties: expected value over the tie
        rec = {"pick": d.ids[pick], "n_tied": int(len(tied)), "regret": float(t_obj.mean() - true_obj.min()),
               "gain": float(true_obj.mean() - t_obj.mean()),
               "percentile": float(np.mean([(true_obj < v).mean() for v in t_obj]))}
        med = np.nanmedian(Y[ok], 0)
        for h, j in ((h, targets.index(h)) for h in ("wl", "wns", "tns", "ovf") if h in targets):
            if np.isnan(Y[pick, j]) or np.isnan(med[j]):
                continue
            # + = better: WL reduction in %, WNS / TNS slack gained (ns), overflow (log1p) reduction
            rec[f"d_{h}"] = float((1 - Y[pick, j] / med[j]) * 100) if h == "wl" else float(med[j] - Y[pick, j])
        out[pref] = rec
    O, names = obj_matrix(Ztrue, targets, mask)
    setm = {"dims": names, "n_picks": len(set(picks))}
    pool_ok = pool[~np.isnan(O[pool]).any(1)] if O.shape[1] else pool[:0]
    if O.shape[1] and len(pool_ok) >= 4 and picks:
        ref = O[pool_ok].max(0) + 0.1
        hv_pool = hypervolume(O[pool_ok], ref)
        uniq = sorted(set(p for p in picks if p in set(pool_ok.tolist())))
        hv_pick = hypervolume(O[uniq], ref) if uniq else 0.0
        k = max(len(uniq), 1)
        rnd = [hypervolume(O[rng.choice(pool_ok, size=k, replace=False)], ref) for _ in range(n_random)]
        setm.update({"hv_ratio": hv_pick / hv_pool if hv_pool > 0 else float("nan"),
                     "hv_ratio_random": float(np.mean(rnd)) / hv_pool if hv_pool > 0 else float("nan")})
    return out, setm


def local_response(pred, d, targets, min_level=2.0, noise=None):
    """Legal-space perturbation families (child moved 1-3 macros by >= min_level um, then LP): does the predicted
    change z_hat(child) - z_hat(parent) agree with the true change? Pairs whose true |dz| is below 2x the flow
    noise of a difference (2 * sqrt(2) * sigma) are excluded from the sign accuracy."""
    pairs = [(c, p) for c, p in families(d, min_level=min_level) if d.source[c] == "legal_perturb"]
    if not pairs:
        return None
    c = [p[0] for p in pairs]; p = [p[1] for p in pairs]
    zc = soft_clip(d.z.cpu().numpy(), Z_CLIP_EVAL)
    dz = zc[c] - zc[p]
    dp = (pred[c] - pred[p]).numpy()
    ok = (~torch.isnan(d.y[c]) & ~torch.isnan(d.y[p])).cpu().numpy() & d.mask.cpu().numpy()[None]
    out = {}
    for j, t in enumerate(targets):
        m = ok[:, j]
        if m.sum() < 5:
            continue
        thr = 2 * math.sqrt(2) * (float(noise[j]) if noise is not None and np.isfinite(noise[j]) else 0.05)
        big = m & (np.abs(dz[:, j]) > thr)
        out[t] = {"n": int(m.sum()), "n_above_noise": int(big.sum()),
                  "sign_acc": float((np.sign(dp[big, j]) == np.sign(dz[big, j])).mean()) if big.any() else None,
                  "pearson": float(np.corrcoef(dp[m, j], dz[m, j])[0, 1]) if np.std(dp[m, j]) > 0 and np.std(dz[m, j]) > 0 else None,
                  "pred_abs_mean": float(np.abs(dp[m, j]).mean()), "true_abs_mean": float(np.abs(dz[m, j]).mean())}
    return out


@torch.no_grad()
def stage_a_eval(model, d, sel, chunk=32):
    """Refine / macro models: Kendall of the predicted placement-stage scalars (stage A) vs OpenROAD's, per feature."""
    if d.place_ok is None or not bool(d.place_ok.cpu().numpy()[sel].all()):
        return None
    outs = [model(d.final_xy[i:i + chunk], d)["place"].cpu() for i in range(0, d.n_samples, chunk)]
    P = torch.cat(outs).numpy()
    Z = np.where(d.place_valid.cpu().numpy(), d.place_z.cpu().numpy(), np.nan)   # unobserved entries -> NaN
    mask = d.place_mask.cpu().numpy()
    res = {}
    for j, name in enumerate(PLACE_NAMES):
        if mask[j] and name in ("dp_hpwl", "gp_final_hpwl", "pl_wns", "pl_tns", "pl_power", "pl_buffers"):
            r = ranking_metrics(P[sel, j], Z[sel, j])
            res[name] = None if r is None else round(r["kendall"], 3)
    return res


def grad_check(model, d, targets, n=5, eps=1e-3, seed=0):
    """Directional derivative of w.z_hat (balanced w) at a few legal layouts: autograd vs central differences."""
    g = torch.Generator().manual_seed(seed)
    w = torch.tensor(pref_weights("balanced", targets, d.mask.cpu().numpy()) if
                     pref_weights("balanced", targets, d.mask.cpu().numpy()) is not None else np.ones(len(targets)),
                     dtype=torch.float32, device=d.final_xy.device)
    errs = []
    for i in range(min(n, d.n_samples)):
        Y = d.final_xy[i:i + 1].clone().requires_grad_(True)
        f = (model(Y, d)["out"] @ w).sum()
        (gr,) = torch.autograd.grad(f, Y)
        v = torch.randn(Y.shape, generator=g).to(Y.device)
        v = v / v.norm()
        with torch.no_grad():
            fp = (model(Y + eps * v, d)["out"] @ w).sum()
            fm = (model(Y - eps * v, d)["out"] @ w).sum()
        fd = float((fp - fm) / (2 * eps))
        ad = float((gr * v).sum())
        errs.append({"autograd": ad, "finite_diff": fd, "rel_err": abs(ad - fd) / max(abs(fd), 1e-6),
                     "grad_norm": float(gr.norm())})
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--static", required=True)
    ap.add_argument("--stdcell", default="")
    ap.add_argument("--designs", default=",".join(TEST))
    ap.add_argument("--pool", choices=("imf", "all", "with_refine"), default="imf",
                    help="candidate pool: iMF seeds | all sources except jitter probes and refine outputs | "
                         "all including refine outputs (generate -> refine -> select pipeline)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grad-check", action="store_true")
    ap.add_argument("--real", default="", help="student models: dir with <design>.real.pt (position diagnostics)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, cfgs = load_models(a.ckpts, dev)
    cfg = cfgs[0]
    for k in ("use_proxy", "proxy_cfg", "input"):
        vals = {json.dumps(c.get(k), sort_keys=True) for c in cfgs}
        assert len(vals) == 1, f"ensemble members disagree on {k}: {vals}"
    targets = tuple(cfg["targets"])
    use_proxy = bool(cfg.get("use_proxy"))
    designs = load_all(a.data, a.static, a.designs.split(","), targets,
                       stdcell_dir=(a.stdcell or cfg.get("stdcell")) if use_proxy else None,
                       proxy_cfg=cfg.get("proxy_cfg"))
    warnings = []
    for name, d in designs.items():
        trained = (cfg.get("proxy_files") or {}).get(name)
        if trained and d.proxy is not None and trained.get("bytes") != d.proxy.source.get("bytes"):
            warnings.append(f"{name}: {name}.stdcell.pt differs from the file used in training ({trained})")
        if use_proxy and d.proxy is None:
            warnings.append(f"{name}: model uses the std-cell proxy but {name}.stdcell.pt is missing (zero inputs)")
        if cfg["kind"] == "selection" and (d.extra.get("obs") is None or not bool(d.extra["obs_ok"].all())):
            n = 0 if d.extra.get("obs") is None else int((~d.extra["obs_ok"]).sum())
            warnings.append(f"{name}: placement-stage maps missing for {n or 'all'} samples")
    for w_ in warnings:
        print("WARNING", w_)
    rng = np.random.default_rng(a.seed)
    noise = np.array([v if v is not None else np.nan for v in cfg.get("noise_sigma_z", [np.nan] * len(targets))], float)
    per, sel_rows, set_rows, grads, local, stage_a, student = {}, {}, {}, {}, {}, {}, {}
    for name, d in designs.items():
        d.to(dev)
        if a.real and cfg["kind"] == "student":
            attach_real(d, a.real)
        P = torch.stack([predict(m, d, cfg["input"]) for m in models])            # (E,S,T)
        mean = P.mean(0).numpy()
        std = P.std(0).numpy() if len(models) > 1 else np.zeros_like(mean)
        src = np.array(d.source)
        if a.pool == "imf":
            sel = src == "imf"
        else:
            sel = ~d.noise_only.cpu().numpy() & (src != "jitter")
            if a.pool == "all":
                sel &= src != "refine"
        Y = d.y.cpu().numpy()
        # single-signal scorers (z within the pool, lower = better)
        scorers = {"predictor": mean,
                   "macro_hpwl": np.repeat(zscore_within(d.extra["final_macro_hpwl"].numpy(), sel)[:, None], len(targets), 1)}
        if cfg["kind"] == "student":
            hh = torch.stack([predict_hpwl_hat(m, d) for m in models]).mean(0).numpy()
            scorers["student_hpwl"] = np.repeat(zscore_within(hh, sel)[:, None], len(targets), 1)
            student[name] = {k: v for k, v in student_diag(models[0], d, sel).items()}
        if d.proxy is not None:
            with torch.no_grad():
                hp = torch.cat([d.proxy.raw_features(d.final_xy[i:i + 8])[2][:, 0] for i in range(0, d.n_samples, 8)])
            scorers["proxy_hpwl"] = np.repeat(zscore_within(hp.cpu().numpy(), sel)[:, None], len(targets), 1)
        if d.place_ok is not None and bool(d.place_ok.cpu().numpy()[sel].any()):
            pr = d.place_raw.cpu().numpy()
            cols = [zscore_within(pr[:, PLACE_NAMES.index(PLACE_FOR_HEAD[t])], sel) if t in PLACE_FOR_HEAD
                    else np.zeros(d.n_samples) for t in targets]
            scorers["place"] = np.stack(cols, 1)
        per[name] = {}
        for j, t in enumerate(targets):
            if not bool(d.mask[j]):
                continue
            r = {k: ranking_metrics(v[sel, j], Y[sel, j]) for k, v in scorers.items()}
            r["ensemble_std_mean"] = float(std[sel, j].mean())
            per[name][t] = r
        sel_rows[name], set_rows[name] = {}, {}
        for k, v in scorers.items():
            sel_rows[name][k], set_rows[name][k] = selection_eval(v, d, sel, targets, rng)
        lr_ = local_response(P.mean(0), d, targets, noise=noise)
        if lr_:
            local[name] = lr_
        if cfg["kind"] in ("refine", "macro", "student"):
            stage_a[name] = stage_a_eval(models[0], d, sel)
        if a.grad_check and cfg["kind"] in ("refine", "student"):
            grads[name] = grad_check(models[0], d, targets)
        d.to("cpu")
    # summaries
    summary = {}
    for t in targets:
        summary[t] = {}
        for src in SOURCES:
            rows = [per[n][t][src] for n in per if t in per[n] and src in per[n][t]]
            if not rows:
                continue
            k, nk, pos = mean_over(rows, "kendall")
            summary[t][src] = {"kendall": k, "n": nk, "pos": pos, "top5": mean_over(rows, "top5_recall")[0],
                               "regret_z": mean_over(rows, "regret_z")[0]}
    sel_summary = {}
    for src in SOURCES:
        sel_summary[src] = {}
        for pref in PREFS:
            rows = [sel_rows[n][src][pref] for n in sel_rows if src in sel_rows[n] and pref in sel_rows[n][src]]
            if rows:
                sel_summary[src][pref] = {k: float(np.mean([r[k] for r in rows if k in r]))
                                          for k in ("regret", "gain", "percentile", "d_wl", "d_wns", "d_tns", "d_ovf")
                                          if any(k in r for r in rows)}
                sel_summary[src][pref]["n"] = len(rows)
        hv = [set_rows[n][src].get("hv_ratio") for n in set_rows if src in set_rows[n]]
        hr = [set_rows[n][src].get("hv_ratio_random") for n in set_rows if src in set_rows[n]]
        hv = [v for v in hv if v is not None and np.isfinite(v)]
        hr = [v for v in hr if v is not None and np.isfinite(v)]
        sel_summary[src]["_set"] = {"hv_ratio": float(np.mean(hv)) if hv else None,
                                    "hv_ratio_random": float(np.mean(hr)) if hr else None, "n": len(hv)}
    res = {"ckpts": a.ckpts, "kind": cfg["kind"], "designs": list(designs), "pool": a.pool, "ranking_summary": summary,
           "selection_summary": sel_summary, "ranking": per, "selection": sel_rows, "selection_set": set_rows,
           "local_response": local, "stage_a_kendall": stage_a, "grad_check": grads, "student": student,
           "warnings": warnings}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"kind={cfg['kind']} ensemble={len(models)} pool={a.pool} designs={list(designs)}")
    print(f"\n{'head':6s} {'scorer':12s} {'kendall':>8s} {'pos/n':>6s} {'top5':>6s} {'regret':>7s}")
    for t, srcs in summary.items():
        for s, v in srcs.items():
            print(f"{t:6s} {s:12s} {v['kendall']:8.3f} {v['pos']:>3d}/{v['n']:<2d} {v['top5']:6.2f} {v['regret_z']:7.2f}")
    print("\nselection (mean over designs): regret / gain in true w.z (z units); pick vs pool median, + = better:"
          " dWL = WL reduction %, dWNS / dTNS = slack gained (ns), dOVF = log1p overflow reduction")
    print(f"{'scorer':12s} {'pref':12s} {'regret':>7s} {'gain':>6s} {'pctl':>5s} {'dWL%':>6s} {'dWNS':>6s} {'dTNS':>8s} {'dOVF':>6s}")
    for s, prefs in sel_summary.items():
        for p, v in prefs.items():
            if p == "_set":
                continue
            print(f"{s:12s} {p:12s} {v.get('regret', float('nan')):7.2f} {v.get('gain', float('nan')):6.2f} "
                  f"{v.get('percentile', float('nan')):5.2f} {v.get('d_wl', float('nan')):6.2f} "
                  f"{v.get('d_wns', float('nan')):6.3f} {v.get('d_tns', float('nan')):8.2f} {v.get('d_ovf', float('nan')):6.2f}")
        st = prefs.get("_set", {})
        print(f"{s:12s} {'SET':12s} hv_ratio {st.get('hv_ratio')} vs random {st.get('hv_ratio_random')}")
    if local:
        print("\nlocal response on legal-perturbation families (sign accuracy above noise / pearson of dz):")
        for n, v in local.items():
            cells = []
            for t, r in v.items():
                acc = "-" if r["sign_acc"] is None else f"{r['sign_acc']:.2f}"
                pr = "-" if r["pearson"] is None else f"{r['pearson']:.2f}"
                cells.append(f"{t}: acc={acc} r={pr} n={r['n_above_noise']}/{r['n']}")
            print(f"  {n:16s} " + "  ".join(cells))
    if stage_a:
        print("\nstage A (predicted placement-stage scalars) Kendall:", json.dumps(stage_a))
    if student:
        print("\nstudent (C:hpwl = Kendall of its HPWL estimate vs GRT_WL; cluster position RMS error in bins, "
              "proxy vs student):", json.dumps(student))
    if grads:
        print("\ngrad check (rel err autograd vs finite diff):",
              {n: [round(g["rel_err"], 3) for g in v] for n, v in grads.items()})


if __name__ == "__main__":
    main()
