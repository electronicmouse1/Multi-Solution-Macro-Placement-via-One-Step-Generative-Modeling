# Multi-Solution Macro Placement via One-Step Generative Modeling

Anonymous code release accompanying the DAC 2027 submission.

**Status: partial release.** This first batch contains the one-step generator
(an improved MeanFlow, "iMF", variant of a graph flow-matching placer), its training
code, and the trained weights used in the paper. Benchmark conversion
utilities, the screening/selection pipeline, and the end-to-end evaluation scripts
will be added in follow-up updates.

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
checkpoints/
  imf_g4p4_step15000.pt              trained weights (model state_dict only, 6.29M params)
  imf_g4p4_step15000_config.yaml     resolved config of the training run
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

## Acknowledgement

This code builds on the public implementation of FlowPlace (DAC 2026). Files derived
from it keep their original authorship and license terms.
