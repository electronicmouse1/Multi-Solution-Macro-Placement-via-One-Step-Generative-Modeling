"""Dataset for the cross-stage PPA predictor.

One DesignData per circuit: static tensors (macro sizes, macro-net incidence with pin offsets,
IO pins, design context) + labeled samples (raw / final macro placements and post-GRT costs).

Targets are COSTS (lower is better):
  wl   = GRT_WL                      tns  = -TNS
  wns  = -WNS                        ovf  = log1p(total_overflow)
  pwr  = Power                       gcell = log(1e-4 + congested_gcell_ratio)      area = StdCellArea
  (aux) setup_vio = log1p(setup violation count)
Each target is normalized PER DESIGN with robust statistics of that design's iMF-seed samples:
  z = (y - median) / (IQR / 1.349). A head is masked for a design whose iMF IQR is ~0
  (e.g. overflow is 0 for every sample of VeriGPU / dft68 / or1200).

Optional per-sample extras (used by the Selection / Refine predictors, absent fields stay None):
  placement-stage observations (labels/<id>/place_features.json via build_dataset.py), placement-stage
  std-cell / macro density maps, legal-perturbation flags, and a per-design std-cell proxy (stdcell.py).
"""
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

TARGETS = ("wl", "wns", "tns", "ovf", "pwr", "setup_vio")

# Placement-stage observations (scripts/ppa/extract_place_features.py), all oriented as COSTS
# (higher = worse): known after floorplan + place only -> inputs of the Selection predictor and
# intermediate targets of the Refine predictor. (name, json key, transform)
PLACE_FEATS = (("gp1_init_hpwl", "gp1_init_hpwl_um", "log"), ("gp_init_hpwl", "gp_init_hpwl_um", "log"),
               ("gp_final_hpwl", "gp_final_hpwl_um", "log"), ("gp_iters", "gp_iters", "id"),
               ("io_hpwl", "io_hpwl_um", "log"), ("dp_hpwl", "dp_hpwl_um", "log"),
               ("dp_hpwl_legal", "dp_hpwl_legalized_um", "log"),
               ("dp_disp", "dp_design_instance_displacement_total", "log"),
               ("dp_disp_max", "dp_design_instance_displacement_max", "log"),
               ("pl_wns_pre", "pl_wns_pre", "neg"), ("pl_tns_pre", "pl_tns_pre", "neglog"),
               ("pl_wns", "pl_wns", "neg"), ("pl_tns", "pl_tns", "neglog"),
               ("pl_buffers", "pl_buffers", "log1p"), ("pl_buffered_nets", "pl_buffered_nets", "log1p"),
               ("pl_resized", "pl_resized", "log1p"), ("pl_slew_viol", "pl_slew_viol", "log1p"),
               ("pl_cap_viol", "pl_cap_viol", "log1p"), ("pl_long_wires", "pl_long_wires", "log1p"),
               ("pl_inst_after", "pl_inst_after", "log"), ("pl_power", "pl_power_total", "id"),
               ("pl_power_comb", "pl_power_combinational", "id"),
               ("pl_worst_end_macro", "pl_worst_endpoint", "is_macro"),
               ("pl_worst_start_macro", "pl_worst_startpoint", "is_macro"))
PLACE_NAMES = tuple(f[0] for f in PLACE_FEATS)
PLACE_Z_CLIP = 6.0          # placement-stage z-scores are soft-clipped to +-6 (model inputs only)
GCELL_EPS = 1e-4            # gcell cost = log(GCELL_EPS + congested_gcell_ratio)
# Robust per-design z of the costs is heavy tailed (|z| up to ~1e4 where a head barely varies across seeds,
# e.g. WNS of near-constant designs): losses use soft_clip(z, Z_CLIP_LOSS), evaluation objectives Z_CLIP_EVAL.
Z_CLIP_LOSS = 4.0
Z_CLIP_EVAL = 6.0


def soft_clip(z, c):
    """c * tanh(z / c) for torch tensors or numpy arrays (identity near 0, bounded by +-c)."""
    if isinstance(z, torch.Tensor):
        return c * torch.tanh(z / c)
    return c * np.tanh(np.asarray(z) / c)
# placement-stage observation that directly estimates a target (Selection skip init, single-feature baselines)
PLACE_PROXY_OF = {"wl": "dp_hpwl", "wns": "pl_wns", "tns": "pl_tns", "pwr": "pl_power"}
# physical prior of the Selection predictor: the placement-stage observation that best tracks each head
# (WL-type observations for the congestion heads); also the single-signal 'place' baseline of evaluate2
PLACE_PRIOR = {**PLACE_PROXY_OF, "ovf": "dp_hpwl", "gcell": "dp_hpwl"}


def _cost(sample, name):
    lab, ex = sample["labels"], sample.get("labels_extra") or {}
    if name == "wl":
        return lab.get("GRT_WL")
    if name == "wns":
        return None if lab.get("WNS") is None else -lab["WNS"]
    if name == "tns":
        return None if lab.get("TNS") is None else -lab["TNS"]
    if name == "ovf":
        v = lab.get("total_overflow")
        return None if v is None else math.log1p(v)
    if name == "pwr":
        return lab.get("Power")
    if name == "gcell":
        v = lab.get("congested_gcell_ratio")       # mostly ~0 with rare spikes: log keeps the robust z sane
        return None if v is None else math.log(GCELL_EPS + v)
    if name == "area":
        return lab.get("StdCellArea")
    if name == "setup_vio":
        v = ex.get("grt__timing__drv__setup_violation_count")
        return None if v is None else math.log1p(v)
    raise KeyError(name)


def _clean(name):
    return str(name).replace("\\", "")


def _place_value(pf, key, how, macro_names):
    v = pf.get(key)
    if v is None:
        return None
    if how == "is_macro":
        return float(_clean(v) in macro_names)
    v = float(v)
    if how == "log":
        return math.log(max(v, 1e-9))
    if how == "log1p":
        return math.log1p(max(v, 0.0))
    if how == "neg":
        return -v
    if how == "neglog":
        return math.log1p(max(-v, 0.0))
    return v


def die_to_canvas(m, canvas_um, die_um):
    """Resample maps (S,C,G,G) [y][x] rasterized on the DIE bbox onto the generator canvas frame
    (normalized = 2 * pos_um / canvas_um - 1, positions measured from the die origin). Outside the die -> 0."""
    if die_um is None:
        return m
    S, C, G, _ = m.shape
    c = (torch.arange(G, dtype=torch.float32) + 0.5) * 2.0 / G - 1.0
    gx = ((c + 1) / 2 * canvas_um[0] - die_um[0]) / (die_um[2] - die_um[0]) * 2 - 1
    gy = ((c + 1) / 2 * canvas_um[1] - die_um[1]) / (die_um[3] - die_um[1]) * 2 - 1
    GY, GX = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([GX, GY], -1).unsqueeze(0).expand(S, -1, -1, -1)
    return torch.nn.functional.grid_sample(m.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def robust_z(x, ref_rows, min_rel_scale=1e-6, min_n=8, binary=()):
    """Column-wise (x - median) / (IQR/1.349) using the statistics of ref_rows (nan ignored).
    Columns in `binary` (0/1 flags) are only centered by their reference mean (an IQR would be 0 or 1).
    Returns z (nan -> 0), mask (columns with a usable scale) and [(center, scale) or None]."""
    z = torch.zeros_like(x)
    mask = torch.zeros(x.shape[1], dtype=torch.bool)
    stats = []
    for j in range(x.shape[1]):
        col = ref_rows[:, j]
        col = col[~torch.isnan(col)]
        if len(col) < min_n:
            stats.append(None)
            continue
        if j in binary:
            if col.min() == col.max():
                stats.append(None)
                continue
            z[:, j] = x[:, j] - col.mean()
            mask[j] = True
            stats.append((float(col.mean()), 1.0))
            continue
        med = col.median()
        scale = (torch.quantile(col, 0.75) - torch.quantile(col, 0.25)) / 1.349
        if not torch.isfinite(scale) or scale <= min_rel_scale * max(1.0, abs(float(med))):
            stats.append(None)
            continue
        z[:, j] = (x[:, j] - med) / scale
        mask[j] = True
        stats.append((float(med), float(scale)))
    return torch.nan_to_num(z, nan=0.0), mask, stats


@dataclass
class DesignData:
    name: str
    split: str
    sizes: torch.Tensor            # (N,2)
    macro_static: torch.Tensor     # (N,Fs) static per-macro features
    pin_macro: torch.Tensor        # (P,)
    pin_off: torch.Tensor          # (P,2)
    pin_net: torch.Tensor          # (P,)
    net_degree: torch.Tensor       # (M,)
    io_xy: torch.Tensor            # (Q,2)
    ctx: torch.Tensor              # (C,)
    raw_xy: torch.Tensor           # (S,N,2)
    final_xy: torch.Tensor         # (S,N,2)
    y: torch.Tensor                # (S,T) raw costs (nan if missing)
    z: torch.Tensor                # (S,T) normalized costs
    mask: torch.Tensor             # (T,) bool, head usable for this design
    stats: dict                    # target -> (median, scale)
    source: list                   # (S,)
    ids: list                      # (S,)
    parent: list                   # (S,) parent id for perturbations
    extra: dict = field(default_factory=dict)   # e.g. corrected macro HPWL per sample
    # --- Selection / Refine predictor extras (None when not available) ---
    place_raw: torch.Tensor = None   # (S,F) placement-stage observations, cost oriented + transformed, nan = missing
    place_z: torch.Tensor = None     # (S,F) per-design robust z (nan -> 0)
    place_mask: torch.Tensor = None  # (F,) feature usable for this design
    place_ok: torch.Tensor = None    # (S,) sample has placement-stage observations
    has_raw: torch.Tensor = None     # (S,) raw iMF output exists (False for legal-space perturbations)
    level_um: torch.Tensor = None    # (S,) legal-perturbation move size in um (nan otherwise)
    noise_only: torch.Tensor = None  # (S,) jitter sample of a test design (flow-noise probe only)
    proxy: object = None             # stdcell.StdCellProxy or None
    place_stats: list = None         # per feature (center, scale) or None, from this design's iMF rows
    place_valid: torch.Tensor = None  # (S,F) entry observed (sample has place features and the key is present)

    @property
    def n_samples(self):
        return self.raw_xy.shape[0]

    def to(self, device):
        for k in ("sizes", "macro_static", "pin_macro", "pin_off", "pin_net", "net_degree", "io_xy", "ctx",
                  "raw_xy", "final_xy", "y", "z", "mask", "place_raw", "place_z", "place_mask", "place_ok",
                  "place_valid", "has_raw", "level_um", "noise_only"):
            if getattr(self, k) is not None:
                setattr(self, k, getattr(self, k).to(device))
        if self.proxy is not None:
            self.proxy.to(device)
        return self


def _ctx_vector(st, stdcell_area_um2=None):
    """Design context. The std-cell area comes from the netlist (label free). Legacy design_static files
    (v1) may still fall back to stdcell_area_med, a median of post-GRT labels; static_version >= 2 never does
    (and has no default clock period either): missing label-free values raise instead of being replaced."""
    c = st["ctx"]
    v2 = st.get("static_version", 1) >= 2
    clk = c.get("clock_period_ns")
    sca = stdcell_area_um2 or c.get("stdcell_area_um2")
    if v2 and (not clk or not sca):
        raise ValueError(f"{st.get('design')}: static v2 ctx lacks clock period / label-free std-cell area")
    clk = clk or 2.0
    sca = sca or c.get("stdcell_area_med") or 1e5
    return torch.tensor([math.log(c["n_macros"]), math.log(c["n_nets"] + 1), math.log(c["n_io"] + 1),
                         c["macro_area_frac"], math.log(c["canvas_aspect"]), math.log(c["canvas_um_x"] / 1000.0),
                         math.log(sca / 1e5), math.log(clk)], dtype=torch.float32)


def load_design(name, data_dir, static_dir, targets=TARGETS, min_rel_scale=1e-6, stdcell_dir=None, proxy_cfg=None):
    ds = torch.load(Path(data_dir) / f"{name}.pt", weights_only=False)
    st = torch.load(Path(static_dir) / f"{name}.pt", weights_only=False)
    samples = [s for s in ds["samples"] if s["final_xy"] is not None]
    S = len(samples)
    y = torch.full((S, len(targets)), float("nan"))
    for i, s in enumerate(samples):
        for j, t in enumerate(targets):
            v = _cost(s, t)
            if v is not None:
                y[i, j] = float(v)
    imf = torch.tensor([s["source"] == "imf" for s in samples])
    ref_rows = y[imf] if imf.sum() >= 8 else y
    z, mask, st_list = robust_z(y, ref_rows, min_rel_scale)
    stats = {t: v for t, v in zip(targets, st_list) if v is not None}
    sizes = st["sizes"].float()
    N = sizes.shape[0]
    pin_cnt = st["macro_pin_count"].float().reshape(N)
    net_deg_m = st["macro_net_degree"].float().reshape(N)
    macro_static = torch.stack([sizes[:, 0], sizes[:, 1], sizes[:, 0] * sizes[:, 1] * 10.0,
                                torch.log1p(pin_cnt) / 5.0, torch.log1p(net_deg_m) / 5.0], dim=1)
    raw = torch.stack([s["raw_xy"].reshape(N, 2).float() for s in samples])
    fin = torch.stack([s["final_xy"].reshape(N, 2).float() for s in samples])
    # OpenROAD maps are rasterized on the DIE bbox; resample them onto the generator canvas frame used by
    # every model-side map (die/canvas differ by up to ~4.5%, i.e. ~3 bins at the far edge on ariane).
    canvas_um = [float(v) for v in st["canvas_um"]]
    die_um = st.get("die_um")
    G = None
    for s in samples:
        if "cell_density" in s["maps"]:
            G = s["maps"]["cell_density"].shape[-1]
            break
    maps = obs = None
    maps_ok = torch.tensor([("congestion" in s["maps"] and "cell_density" in s["maps"]) for s in samples])
    obs_ok = torch.tensor([("cell_density" in s["maps"] and "macro_density" in s["maps"]) for s in samples])
    if G is not None and maps_ok.any():
        # (S,3,G,G): log1p overflow H, log1p overflow V, std-cell density ([y][x]); zeros where maps_ok is False
        maps = torch.stack([torch.cat([torch.log1p(s["maps"]["congestion"].float()),
                                       s["maps"]["cell_density"].float().unsqueeze(0)], 0) if ok else torch.zeros(3, G, G)
                            for s, ok in zip(samples, maps_ok.tolist())])
        maps = die_to_canvas(maps, canvas_um, die_um).half()
    if G is not None and obs_ok.any():
        # (S,2,G,G) placement-stage observations: std-cell density after placement, macro density
        obs = torch.stack([torch.stack([s["maps"]["cell_density"].float(), s["maps"]["macro_density"].float()])
                           if ok else torch.zeros(2, G, G) for s, ok in zip(samples, obs_ok.tolist())])
        obs = die_to_canvas(obs, canvas_um, die_um).half()
    macro_names = {_clean(n) for n in ds.get("names", [])}
    praw = torch.full((S, len(PLACE_FEATS)), float("nan"))
    pok = torch.zeros(S, dtype=torch.bool)
    for i, s in enumerate(samples):
        pf = s.get("place_features")
        if not pf:
            continue
        pok[i] = True
        for j, (_, key, how) in enumerate(PLACE_FEATS):
            v = _place_value(pf, key, how, macro_names)
            if v is not None:
                praw[i, j] = v
    pref = praw[imf & pok] if int((imf & pok).sum()) >= 8 else praw[pok]
    binary = tuple(j for j, f in enumerate(PLACE_FEATS) if f[2] == "is_macro")
    pz, pmask, pstats = robust_z(praw, pref, binary=binary)
    pz = PLACE_Z_CLIP * torch.tanh(pz / PLACE_Z_CLIP)        # robust z has heavy tails (|z| up to ~12 seen)
    level = torch.tensor([float(s["level_um"]) if s.get("level_um") is not None else float("nan") for s in samples])
    proxy = None
    sca_um2 = None
    sc_path = Path(stdcell_dir or Path(static_dir).parent / "stdcell") / f"{name}.stdcell.pt"
    if sc_path.is_file():
        raw_sc = torch.load(sc_path, weights_only=False)
        sca_um2 = float(raw_sc["total_area"]) * canvas_um[0] * canvas_um[1] / 4.0      # normalized area -> um^2
        if stdcell_dir:
            from .stdcell import StdCellProxy
            proxy = StdCellProxy(raw_sc, **(proxy_cfg or {})).attach(sizes)
            crit_path = sc_path.parent / f"{name}.crit.pt"                 # optional (crit_proxy.py)
            if crit_path.is_file():
                proxy.attach_crit(torch.load(crit_path, weights_only=False))
            proxy.calibrate(fin[imf] if imf.sum() >= 8 else fin)
            proxy.source = {"path": str(sc_path), "bytes": sc_path.stat().st_size, "K": int(raw_sc["K"])}
    extra = {"maps": maps, "obs": obs, "maps_ok": maps_ok, "obs_ok": obs_ok,
             "final_macro_hpwl": torch.tensor([float((s.get("macro_hpwl") or {}).get("final_macro_hpwl_um") or float("nan"))
                                               for s in samples]),
             "raw_macro_hpwl": torch.tensor([float((s.get("macro_hpwl") or {}).get("raw_macro_hpwl_um") or float("nan"))
                                             for s in samples]),
             "place_dp_hpwl": torch.tensor([float((s.get("labels_extra") or {}).get("place_dp_hpwl_um") or float("nan"))
                                            for s in samples])}
    return DesignData(name=name, split=ds.get("split", st["split"]), sizes=sizes, macro_static=macro_static,
                      pin_macro=st["pin_macro"].long(), pin_off=st["pin_off"].float(), pin_net=st["pin_net"].long(),
                      net_degree=st["net_degree"].float(), io_xy=st["io_xy"].float(), ctx=_ctx_vector(st, sca_um2),
                      raw_xy=raw, final_xy=fin, y=y, z=z, mask=mask, stats=stats,
                      source=[s["source"] for s in samples], ids=[s["id"] for s in samples],
                      parent=[s.get("parent") for s in samples], extra=extra,
                      place_raw=praw, place_z=pz, place_mask=pmask, place_ok=pok, place_stats=pstats,
                      place_valid=pok.unsqueeze(1) & ~torch.isnan(praw),
                      has_raw=torch.tensor([bool(s.get("has_raw", True)) for s in samples]), level_um=level,
                      noise_only=torch.tensor([bool(s.get("noise_only", False)) for s in samples]), proxy=proxy)


REAL_RUDY_SCALE = 0.05      # real cell-level RUDY target = log1p(REAL_RUDY_SCALE * demand), as the proxy's RUDY maps


@torch.no_grad()
def attach_real(d, real_dir, chunk=16):
    """Privileged REAL std-cell targets of the Refine student (ppa_surrogate.stdcell_real; training only).
    Needs d.proxy (on its final device). Adds to d.extra, aligned to d's samples (CPU tensors):
      real_has (S,) bool, real_xc (S,K,2) real cluster centroids, real_valid (S,K) cluster has placed cells,
      real_rudy (S,G,G) = log1p(REAL_RUDY_SCALE * cell-level RUDY), real_hpwl_um (S,),
      real_oscal (S,7) / real_omacro (S,N,3) = proxy scalars / macro features evaluated AT the real centroids
      (scalars normalized like the proxy's own), clusters without placed cells at the proxy's position.
    Returns the number of samples with real data (0 if the design has no file)."""
    path = Path(real_dir) / f"{d.name}.real.pt"
    if d.proxy is None or not path.is_file():
        return 0
    r = torch.load(path, weights_only=False)
    if r.get("stdcell_bytes") is not None and r["stdcell_bytes"] != d.proxy.source.get("bytes"):
        raise ValueError(f"{d.name}: {path.name} was built against a different {d.name}.stdcell.pt")
    pos = {sid: i for i, sid in enumerate(r["ids"])}
    idx = torch.tensor([pos.get(sid, -1) for sid in d.ids])
    has = idx >= 0
    S, K, G = d.n_samples, r["xc"].shape[1], r["grid"]
    assert K == d.proxy.A_M.shape[0], f"{d.name}: real K {K} vs proxy K {d.proxy.A_M.shape[0]}"
    xc = torch.zeros(S, K, 2)
    valid = torch.zeros(S, K, dtype=torch.bool)
    rudy = torch.zeros(S, G, G)
    hp = torch.full((S,), float("nan"), dtype=torch.float64)
    sel = idx[has]
    xc[has], valid[has] = r["xc"][sel].float(), r["valid"][sel]
    rudy[has] = torch.log1p(REAL_RUDY_SCALE * r["rudy"][sel].float())
    hp[has] = r["hpwl_um"][sel].double()
    dev = d.final_xy.device
    osc, omac = [], []
    for i in range(0, S, chunk):
        Y = d.final_xy[i:i + chunk]
        xs = d.proxy.spread(Y)
        X = torch.where(valid[i:i + chunk].to(dev).unsqueeze(-1), xc[i:i + chunk].to(dev), xs)
        _, om, os_ = d.proxy.features_at(Y, X)
        osc.append(os_.float().cpu()); omac.append(om.float().cpu())
    d.extra.update({"real_has": has, "real_xc": xc, "real_valid": valid, "real_rudy": rudy.half(),
                    "real_hpwl_um": hp, "real_oscal": torch.cat(osc), "real_omacro": torch.cat(omac)})
    return int(has.sum())


def subset(d, idx):
    """DesignData restricted to sample indices idx (static tensors shared; normalization stats unchanged)."""
    import copy
    s = copy.copy(d)
    idx = list(idx)
    for k in ("raw_xy", "final_xy", "y", "z", "place_raw", "place_z", "place_ok", "place_valid", "has_raw", "level_um",
              "noise_only"):
        v = getattr(d, k)
        if v is not None:
            setattr(s, k, v[torch.as_tensor(idx, dtype=torch.long, device=v.device)])
    s.source = [d.source[i] for i in idx]
    s.ids = [d.ids[i] for i in idx]
    s.parent = [d.parent[i] for i in idx]
    s.extra = {k: (v[torch.as_tensor(idx, dtype=torch.long, device=v.device)] if v is not None else None)
               for k, v in d.extra.items()}
    return s


def within_split(d, frac=0.25, seed=0):
    """Hold out a fraction of iMF seeds (with their perturbed children) of a design.
    Returns (train_idx, val_idx); val contains only iMF samples so metrics match the LOCO protocol."""
    rng = random.Random(seed)
    imf = [i for i, s in enumerate(d.source) if s == "imf"]
    held = set(rng.sample(imf, max(4, int(frac * len(imf)))))
    held_ids = {d.ids[i] for i in held}
    train = [i for i in range(d.n_samples) if i not in held and d.parent[i] not in held_ids]
    return train, sorted(held)


def load_all(data_dir, static_dir, designs, targets=TARGETS, stdcell_dir=None, proxy_cfg=None):
    return {d: load_design(d, data_dir, static_dir, targets, stdcell_dir=stdcell_dir, proxy_cfg=proxy_cfg)
            for d in designs}


def families(d, min_level=None):
    """[(child_idx, parent_idx)] of perturbation children whose parent is in d.
    min_level: keep only legal-space perturbations with level_um >= min_level."""
    id2i = {sid: i for i, sid in enumerate(d.ids)}
    out = []
    for i, p in enumerate(d.parent):
        if p is None or p not in id2i:
            continue
        if min_level is not None:
            lv = float(d.level_um[i]) if d.level_um is not None else float("nan")
            if not lv >= min_level:
                continue
        out.append((i, id2i[p]))
    return out


def estimate_noise(designs, targets):
    """OpenROAD flow noise per head, in per-design z units, from jitter children (5 nm / one-site moves) vs
    their parent: both labels carry independent noise, so sigma = sqrt(mean(dz^2) / 2).
    Returns (median over designs (T,), {design: (T,)}); nan where no jitter pairs exist."""
    per = {}
    for name, d in designs.items():
        pairs = [(c, p) for c, p in families(d) if d.source[c] == "jitter"]
        if not pairs:
            continue
        c, p = map(list, zip(*pairs))
        dz = (d.z[c] - d.z[p]).float().cpu()
        ok = (d.mask.cpu().unsqueeze(0).expand_as(dz) & ~torch.isnan(d.y[c].cpu()) & ~torch.isnan(d.y[p].cpu()))
        sig = torch.full((len(targets),), float("nan"))
        for j in range(len(targets)):
            if ok[:, j].sum() >= 3:
                sig[j] = torch.sqrt((dz[ok[:, j], j] ** 2).mean() / 2)
        per[name] = sig
    if not per:
        return torch.full((len(targets),), float("nan")), per
    return torch.nanmedian(torch.stack(list(per.values())), 0).values, per


class DesignBatchSampler:
    """Yields (design, sample indices). Batches stay within one design so the graph is shared and the
    ranking loss is within-design. Half of each batch is built from perturbation families
    (a parent iMF sample + its perturbed children) when available, to teach local sensitivity.
    exclude(d, i) -> True drops sample i from training batches (e.g. test-design jitter probes)."""

    def __init__(self, designs, batch_size=16, family_frac=0.5, seed=0, exclude=None, mix_imf=None):
        self.designs = designs
        self.batch_size = batch_size
        self.family_frac = family_frac
        self.rng = random.Random(seed)
        self.names = list(designs)
        self.allowed = {n: [i for i in range(d.n_samples) if not (exclude and exclude(d, i))]
                        for n, d in designs.items()}
        self.mix_imf = mix_imf                      # None: legacy family sampling; F: F of the batch from iMF seeds
        self.weights = ([1.0] * len(self.names) if mix_imf is not None else
                        [math.sqrt(max(len(self.allowed[n]), 1)) for n in self.names])
        self.families, self.fam_cap = {}, {}
        for n, d in designs.items():
            ok = set(self.allowed[n])
            id2i = {sid: i for i, sid in enumerate(d.ids)}
            fam = {}
            for i, p in enumerate(d.parent):
                if p is not None and p in id2i and i in ok and id2i[p] in ok:
                    fam.setdefault(id2i[p], [id2i[p]]).append(i)
            self.families[n] = list(fam.values())
            self.fam_cap[n] = len(set().union(*map(set, self.families[n]))) if self.families[n] else 0

    def sample(self):
        name = self.rng.choices(self.names, weights=self.weights)[0]
        if self.mix_imf is not None:
            return name, self._sample_mixed(name)
        idx = []
        fams = self.families[name]
        budget = min(int(self.batch_size * self.family_frac), self.fam_cap[name])
        tries = 0
        while fams and len(idx) < budget and tries < 64:
            tries += 1
            fam = self.rng.choice(fams)
            room = budget - len(idx)
            if len(fam) > room:                           # keep the parent, subsample its children
                fam = [fam[0]] + self.rng.sample(fam[1:], max(room - 1, 1))
            idx.extend(i for i in fam if i not in idx)
        taken = set(idx)
        pool = [i for i in self.allowed[name] if i not in taken]
        idx.extend(self.rng.sample(pool, max(0, min(len(pool), self.batch_size - len(idx)))))
        return name, sorted(set(idx))

    def _sample_mixed(self, name):
        """Stratified batch: round(batch * (1 - mix_imf)) samples from the non-iMF allowed sources (perturb, dreamplace,
        ref ...), the rest from the iMF seeds; a short side is topped up from the other one."""
        d = self.designs[name]
        imf = [i for i in self.allowed[name] if d.source[i] == "imf"]
        oth = [i for i in self.allowed[name] if d.source[i] != "imf"]
        n_oth = min(len(oth), int(round(self.batch_size * (1.0 - self.mix_imf))))
        n_imf = min(len(imf), self.batch_size - n_oth)
        n_oth = min(len(oth), self.batch_size - n_imf)
        return sorted(self.rng.sample(imf, n_imf) + self.rng.sample(oth, n_oth))
