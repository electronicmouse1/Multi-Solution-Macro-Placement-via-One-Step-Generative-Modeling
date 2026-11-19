# Multi-Solution Macro Placement via One-Step Generative Modeling

Anonymous code release accompanying the DAC 2027 submission.

**Status: partial release.** This release contains the one-step generator
(an improved MeanFlow, "iMF", variant of a graph flow-matching placer), its training
code, and the trained weights used in the paper, and the PPA predictor used for screening
with its training code and the six trained checkpoints of the frozen ensemble, the coordinates of all
16,384 candidates of the four scale-up designs, and the DEF files of our placements in Table I. Benchmark conversion
utilities, the predictor dataset pipeline (OpenROAD labelling), the multi-stage screening pipeline,
and the end-to-end evaluation scripts will be added in follow-up updates.

## Paper with appendix

[`paper/WIDEPlace_with_appendix.pdf`](paper/WIDEPlace_with_appendix.pdf) is the submitted paper followed by an appendix
with additional results (pages 8 onward); the first seven pages are identical to the submission.

## Contents

```
flow_matching/            model, networks, training/eval loops, legalizers, Hydra configs
  models.py               IMFPlacementModel (one-step interval sampler, u_theta)
  configs/experiment/imf_raw.yaml   minimal iMF configuration
common/                   checkpointing, logging, metric helpers
scripts/phase1/remote_validation.py  config composition used by training
scripts/phase2/imf_continue_gp.py    training entry point used for the released weights
ppa_surrogate/            PPA predictor
  models2.py              predictor model ("macro": macro-pin-net message passing + map CNN + std-cell proxy prior)
  stdcell.py              std-cell cluster proxy (spreading around the legal macros) and its features
  train2.py               training entry point (Huber + within-design ranking loss) and prediction
  data.py, features.py, model.py, losses.py, metrics.py, evaluate2.py   data loading, layers, losses, checkpoint loading
scripts/predictor/score_cands.py     scores a pool of legal placements with the frozen ensemble
checkpoints/
  imf_g4p4_step15000.pt              trained weights (model state_dict only, 6.29M params)
  imf_g4p4_step15000_config.yaml     resolved config of the training run
  predictor/P1_v{1,2}_s{0,1,2}.pt    frozen predictor ensemble (6 x 1.4M params), with *_config.json and SHA256SUMS
environment.yaml          conda environment
```

## Environment

Tested with Python 3.10, PyTorch 2.2.1 (CUDA 11.8) and PyTorch Geometric 2.4.0.

```bash
conda env create -f environment.yaml
```

## Loading the trained model

```python
import sys, torch
from omegaconf import OmegaConf
sys.path[:0] = [".", "flow_matching"]
import models

cfg = OmegaConf.load("checkpoints/imf_g4p4_step15000_config.yaml")
cfg.model.device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.get_model_class(cfg.family)(**cfg.model).to(cfg.model.device)
ckpt = torch.load("checkpoints/imf_g4p4_step15000.pt", map_location=cfg.model.device, weights_only=True)
model.load_state_dict(ckpt["model"], strict=True)   # step 15000
model.eval()
```

Sampling uses one interval step from the uniform prior
(`cfg.model.imf_sampling_steps = 1`, see `flow_matching/configs/experiment/imf_raw.yaml`).

## Training

The released weights were obtained with `scripts/phase2/imf_continue_gp.py`
(graph batch 4, path batch 4, learning rate 3e-4, seed 61), resuming from a
5k-step checkpoint and stopping at step 15000. The full resolved configuration is in
`checkpoints/imf_g4p4_step15000_config.yaml`. Synthetic training data generation
follows the original FlowPlace data generator and is not included in this batch.

## PPA predictor

The screening predictor is a frozen ensemble of six models with the same architecture and hyper-parameters
(`model = macro`, `--prior-fixed`, 6,000 steps, seeds 0-2): `P1_v1_s*` are trained on the first labelled set and
`P1_v2_s*` on the extended set of 6,229 GRT-labelled placements of 14 ChiPBench circuits; none of the six test
designs is used for training. Each model predicts per-design z-scored costs (lower is better) for
`wl, wns, tns, ovf, gcell, pwr` from a legal macro placement; screening averages the six predictions and uses the
WL, WNS and TNS heads. The predictor additionally needs `scipy` (tested with 1.15.3).

Training (one member; the full resolved arguments are in `checkpoints/predictor/*_config.json`):

```bash
python -m ppa_surrogate.train2 --model macro --prior-fixed --data D/dataset --static D/static_v2 \
    --stdcell D/stdcell --out runs/P1_s0 --seed 0
```

Scoring a pool of legal placements (design data in the dataset format, see the docstring):

```bash
python scripts/predictor/score_cands.py IN_DIR OUT --static-dir D/static_v2 --stdcell-dir D/stdcell \
    --names-dir D/dataset ariane133
```

## Data

### Candidate pools (`data/candidates/`)

The 16,384 generated candidates (seeds 0-16383) of the four designs used in the scale-up experiments
(`bp_fe`, `swerv_wrapper`, `ariane133`, `bp`), after legalization, one file per design:

| key | shape | meaning |
|---|---|---|
| `seeds` | (16384,) | sampling seed of each candidate |
| `names` | (N,) | macro instance names |
| `ll_dbu` | (16384, N, 2) int32 | lower-left corner of each macro in DBU (`ll_um = ll_dbu / dbu_per_um`) |
| `dbu_per_um` | () | 2000 |
| `size_um` | (N, 2) | macro width and height |
| `die_um`, `core_um` | (4,) | die and core area (x0, y0, x1, y1) |

```python
import numpy as np
z = np.load("data/candidates/ariane133.npz")
ll_um = z["ll_dbu"] / z["dbu_per_um"]     # (16384, 132, 2)
```

### Placements of Table I (`data/table1_ours_def/`)

DEF files of the placements reported for `Ours` in Table I, one folder per design, compressed with xz
(`xz -dk <file>.xz`); `SHA256SUMS` checks the `.xz` files and `SHA256SUMS_uncompressed` the DEF files.

| design | seed | file |
|---|---|---|
| ariane133 | 13 | `6_final.def` (routed) |
| ariane136 | 2711 | `6_final.def` (routed) |
| bp | 92 | `6_final.def` (routed) |
| bp_be | 108 | `6_final.def` (routed) |
| bp_fe | 109 | `6_final.def` (routed) |
| swerv_wrapper | 1932 | `6_final.def` (routed) |
| mempool_tile | 270 | `3_4_place_resized.def` (after placement; the routed DEF was not kept) |
| nvdla | 2865 | `3_4_place_resized.def` (after placement; the routed DEF was not kept) |

The macro locations in these DEFs are the ones produced by the OpenROAD flow, which snaps macros to its grid (within
0.11 um of the candidate coordinates; for the four designs above, seed `s` is row `s` of `data/candidates/<design>.npz`).

## Acknowledgement

This code builds on the public implementation of FlowPlace (DAC 2026). Files derived
from it keep their original authorship and license terms.
