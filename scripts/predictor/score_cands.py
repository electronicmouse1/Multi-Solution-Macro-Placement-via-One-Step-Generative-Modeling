"""Score candidate macro placements with the frozen PPA predictor ensemble (P1ens, six checkpoints in
checkpoints/predictor/) plus the std-cell proxy HPWL and the macro-only HPWL, as used for predictor-based screening.

Input  IN_DIR/<design>.npz: ids (S,) str, seeds (S,), names (N,) macro instance names, ll_um (S,N,2) lower-left corners
       of the legalized macro placement. One pool per design is scored at once (the std-cell proxy is calibrated on the
       pool).
Design data (built by the dataset pipeline, not part of this release):
       --static-dir  <design>.pt  static design graph (canvas_um, sizes, nets, pins)
       --stdcell-dir <design>.stdcell.pt (and optionally <design>.crit.pt)  std-cell cluster proxy
       --names-dir   <design>.pt  any dataset file of the design; only its macro order ("names") is used
Output OUT/<design>_scores.json: ids, seeds, P1ens {wl,wns,tns} (predicted cost z, lower = better), proxy, hpwl_mm.

Usage: python scripts/predictor/score_cands.py IN_DIR OUT --static-dir S --stdcell-dir C --names-dir R design [design ...]
Run from the repository root (ppa_surrogate/ must be importable)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
from features_a import A_NAMES, features_a  # noqa: E402
from ppa_surrogate.data import load_all  # noqa: E402
from ppa_surrogate.evaluate2 import load_models  # noqa: E402
from ppa_surrogate.train2 import predict  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("in_dir", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("designs", nargs="+")
ap.add_argument("--static-dir", type=Path, required=True)
ap.add_argument("--stdcell-dir", type=Path, required=True)
ap.add_argument("--names-dir", type=Path, required=True)
ap.add_argument("--ckpt-dir", type=Path, default=ROOT / "checkpoints" / "predictor")
a = ap.parse_args()
IN, OUT, designs = a.in_dir, a.out, a.designs
STATIC, SC, R0 = a.static_dir, a.stdcell_dir, a.names_dir
OUT.mkdir(parents=True, exist_ok=True)

# frozen ensemble: three members trained on the v1 dataset, three on the v2 dataset (same model and hyper-parameters)
P1 = [a.ckpt_dir / f"P1_v1_s{s}.pt" for s in range(3)] + [a.ckpt_dir / f"P1_v2_s{s}.pt" for s in range(3)]
sums = a.ckpt_dir / "SHA256SUMS"
if sums.exists():
    want = dict(line.split()[::-1] for line in sums.read_text().splitlines() if line.strip())
    for p in P1:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == want[p.name], f"checkpoint changed: {p}"
clean = lambda s: str(s).replace("\\", "")  # noqa: E731

# ---- candidate pools in the dataset .pt schema (no labels) -----------------------------------------------------------
DS = OUT / "dataset"; DS.mkdir(exist_ok=True)
SCL = OUT / "stdcell"; SCL.mkdir(exist_ok=True)
for D in designs:
    for f in (f"{D}.stdcell.pt", f"{D}.crit.pt"):
        if (SC / f).exists() and not (SCL / f).exists():
            (SCL / f).symlink_to((SC / f).resolve())
SC = SCL
for D in designs:
    z = np.load(IN / f"{D}.npz", allow_pickle=False)
    names = list(torch.load(R0 / f"{D}.pt", weights_only=False)["names"])
    st = torch.load(STATIC / f"{D}.pt", weights_only=False)
    canvas = np.asarray([float(v) for v in st["canvas_um"]])
    size_um = st["sizes"].double().numpy() * canvas / 2.0
    pos = {clean(n): i for i, n in enumerate(z["names"])}
    perm = np.array([pos[clean(n)] for n in names])               # KeyError = macro set mismatch
    ll = z["ll_um"][:, perm, :]
    fin = torch.tensor(2.0 * (ll + size_um / 2.0) / canvas - 1.0, dtype=torch.float32)
    N = len(names)
    samples = [dict(id=str(i), source="imf", parent=None, level_um=None, has_raw=False, noise_only=False,
                    raw_xy=torch.zeros(N, 2), final_xy=fin[k], labels={}, labels_extra={}, macro_hpwl=None, maps={},
                    place_features=None) for k, i in enumerate(z["ids"])]
    torch.save(dict(design=D, names=names, split="test", canvas_um=canvas.tolist(), samples=samples), DS / f"{D}.pt")
    print(D, "pool", len(samples), "macros", N, flush=True)

# ---- predictions -----------------------------------------------------------------------------------------------------
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models, cfgs = load_models([str(p) for p in P1], dev)
assert all(c.get("prior_crit") in (None, "") for c in cfgs), "P1 checkpoints use crit weights"
T = list(cfgs[0]["targets"])
DD = load_all(DS, STATIC, designs, tuple(T), stdcell_dir=SC)
for D in designs:
    d = DD[D].to(dev)
    assert d.proxy is not None, f"no std-cell proxy for {D}"
    with torch.no_grad():
        p = torch.stack([predict(m, d) for m in models]).mean(0).numpy()
        hp = torch.cat([d.proxy.raw_features(d.final_xy[i:i + 8])[2][:, 0] for i in range(0, d.n_samples, 8)]).cpu().numpy()
    st = torch.load(STATIC / f"{D}.pt", weights_only=False)
    XA = features_a(d.final_xy.cpu(), st).numpy().astype(float)
    z = np.load(IN / f"{D}.npz", allow_pickle=False)
    res = dict(ids=list(d.ids), seeds=[int(s) for s in z["seeds"]],
               P1ens={h: p[:, T.index(h)].tolist() for h in ("wl", "wns", "tns")}, proxy=hp.tolist(),
               hpwl_mm=XA[:, list(A_NAMES).index("hpwl_mm")].tolist(), device=str(dev))
    assert res["ids"] == [str(i) for i in z["ids"]]
    (OUT / f"{D}_scores.json").write_text(json.dumps(res))
    print(D, "scored", len(res["ids"]), flush=True)
    d.to("cpu")
print("SCORE_DONE")
