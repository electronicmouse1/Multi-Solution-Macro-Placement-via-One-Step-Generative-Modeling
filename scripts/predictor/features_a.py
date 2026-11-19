"""v2.5 / 1A-2: feature group A = corrected macro geometry features (label free, no IO input) of every labeled sample.

python features_a.py [--designs a,b] [--static round0/static_v2] [--out round0/features_v25/A]
Per design writes <out>/<design>.pt {design, ids, source, parent, names (F,), X (S,F) float32, legacy_macro_hpwl_um (S,)}.
Columns (A_NAMES), computed from final_xy, the macro sizes and the static_v2 tables only:
  hpwl_mm     log of the macro-macro net HPWL, every macro pin of the nets with >= 2 distinct macros, corrected offsets
              (partial span: only macro pins are known endpoints)
  hpwl_mm_ng  the same without 'global' nets (nets that touch >= 50% of the macros: clock / reset like)
  centre, corner, edge_touch, edge_dist, bbox, spread, centroid_off, free_largest, free_ncomp, free_near
              pure macro geometry, formulas of ppa_surrogate.macro_select.layout_features
  rudy_peak   99th percentile of the macro-net RUDY map (macro-macro nets, corrected pins)
  pin_peak    99th percentile of the macro-pin density map built from ALL macro pins (corrected offsets)
The old MACRO_FEATS io_block (needs IO coordinates) and graph_hpwl (a duplicate of macro_hpwl on the same nets) are gone.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
from v25_common import ALL, R0  # noqa: E402
from ppa_surrogate import features as feat  # noqa: E402
from ppa_surrogate.macro_select import _box_overlap, _free_space  # noqa: E402

A_NAMES = ("hpwl_mm", "hpwl_mm_ng", "centre", "corner", "edge_touch", "edge_dist", "bbox", "spread", "centroid_off",
           "free_largest", "free_ncomp", "free_near", "rudy_peak", "pin_peak")


def net_extent_um(P, seg, n_seg, to_um):
    """(B,P,2) pin positions, (P,) net index -> (B,n_seg) half-perimeter in um (nets without pins give 0)."""
    B = P.shape[0]
    sg = seg.unsqueeze(0).expand(B, -1)
    ext = 0
    for k in range(2):
        mx = torch.full((B, n_seg), -1e9).scatter_reduce(1, sg, P[..., k], reduce="amax", include_self=True)
        mn = torch.full((B, n_seg), 1e9).scatter_reduce(1, sg, P[..., k], reduce="amin", include_self=True)
        ext = ext + (mx - mn).clamp_min(0) * to_um[k]
    return ext


@torch.no_grad()
def features_a(Y, st, chunk=16):
    """Y (S,N,2) normalized centers -> (S, len(A_NAMES)) float32."""
    canvas = st["canvas_um"].double()
    to_um = (canvas / 2).float()
    sizes = st["sizes"].float()
    half = sizes / 2
    area = (sizes[:, 0] * sizes[:, 1]).clamp_min(1e-12)
    A = float(area.sum())
    M = int(st["net_degree"].numel())
    mid, mglob = st["mnet_id"].numpy(), st["mnet_global"].numpy()
    tab_glob = torch.as_tensor(mglob[np.searchsorted(mid, st["mm_net_global_id"].numpy())])
    d_ns = SimpleNamespace(sizes=sizes, pin_macro=st["pin_macro"], pin_off=st["pin_off"].float(), pin_net=st["pin_net"],
                           net_degree=st["net_degree"].float(), io_xy=st["io_xy"].float())
    all_macro, all_off = st["all_pin_macro"], st["all_pin_off"].float()
    G, eps = feat.GRID, 0.02
    rows = []
    for i0 in range(0, Y.shape[0], chunk):
        y = Y[i0:i0 + chunk].float()
        B = y.shape[0]
        lo, hi = y - half, y + half
        f = {}
        ext = net_extent_um(y[:, st["pin_macro"], :] + st["pin_off"].float(), st["pin_net"], M, to_um)
        f["hpwl_mm"] = torch.log(ext.sum(1).clamp_min(1e-9))
        f["hpwl_mm_ng"] = torch.log(ext[:, ~tab_glob].sum(1).clamp_min(1e-9))
        f["centre"] = _box_overlap(lo, hi, (-0.5, -0.5), (0.5, 0.5)).sum(1) / A
        corner = sum(_box_overlap(lo, hi, (sx * 0.5 if sx > 0 else -1.0, sy * 0.5 if sy > 0 else -1.0),
                                  (1.0 if sx > 0 else -0.5, 1.0 if sy > 0 else -0.5)) for sx in (-1, 1) for sy in (-1, 1))
        f["corner"] = corner.sum(1) / A
        gap = torch.stack([lo[..., 0] + 1, 1 - hi[..., 0], lo[..., 1] + 1, 1 - hi[..., 1]], -1).clamp_min(0)
        f["edge_touch"] = ((gap.amin(-1) <= eps).float() * area).sum(1) / A
        gap_um = gap * torch.stack([to_um[0], to_um[0], to_um[1], to_um[1]])
        f["edge_dist"] = (gap_um.amin(-1) * area).sum(1) / A
        f["bbox"] = ((hi[..., 0].amax(1) - lo[..., 0].amin(1)) * (hi[..., 1].amax(1) - lo[..., 1].amin(1))) / 4.0
        yu = y * to_um
        cen = (yu * area.view(1, -1, 1)).sum(1) / A
        f["spread"] = torch.sqrt((((yu - cen.unsqueeze(1)) ** 2).sum(-1) * area).sum(1) / A)
        f["centroid_off"] = cen.norm(dim=-1)
        fs = np.array([_free_space(lo[b].numpy(), hi[b].numpy()) for b in range(B)])
        f["free_largest"], f["free_ncomp"], f["free_near"] = (torch.tensor(fs[:, k]) for k in range(3))
        f["rudy_peak"] = torch.quantile(feat.spatial_maps(y, d_ns)[:, 3].flatten(1), 0.99, dim=1)
        P_all = y[:, all_macro, :] + all_off
        den = feat.raster_points(P_all, torch.ones(P_all.shape[:2]), G) * (G * G / max(P_all.shape[1], 1))
        f["pin_peak"] = torch.quantile(torch.log1p(torch.relu(den) * 0.2).flatten(1), 0.99, dim=1)
        rows.append(torch.stack([f[k].float() for k in A_NAMES], 1))
    return torch.cat(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--designs", default=",".join(ALL))
    ap.add_argument("--static", default=str(R0 / "static_v2"))
    ap.add_argument("--out", default=str(R0 / "features_v25/A"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for d in a.designs.split(","):
        st = torch.load(Path(a.static) / f"{d}.pt", weights_only=False)
        ds = torch.load(R0 / "dataset" / f"{d}.pt", weights_only=False)
        S = [s for s in ds["samples"] if s["final_xy"] is not None]
        Y = torch.stack([s["final_xy"].reshape(-1, 2).float() for s in S])
        X = features_a(Y, st)
        legacy = torch.tensor([float((s.get("macro_hpwl") or {}).get("final_macro_hpwl_um") or float("nan")) for s in S])
        assert torch.isfinite(X).all(), f"{d}: non-finite features {A_NAMES[int((~torch.isfinite(X)).any(0).nonzero()[0])]}"
        torch.save({"design": d, "ids": [s["id"] for s in S], "source": [s["source"] for s in S],
                    "parent": [s.get("parent") for s in S], "names": A_NAMES, "X": X, "legacy_macro_hpwl_um": legacy,
                    "static": str(Path(a.static) / f"{d}.pt")}, out / f"{d}.pt")
        print(d, "samples", len(S), "features", X.shape[1], flush=True)


if __name__ == "__main__":
    main()
