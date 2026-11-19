# Baseline legalizers

Unmodified copies of three baseline legalizers: FlowPlace's hard projection (V0) and ChipDiffusion's legalizer, used in
the legalizer ablation, and FlowPlace's soft legalizer. They are included for
reference only and are not adapted to this repository (imports and entry points are those of their original code bases).

| folder | legalizer | settings |
|---|---|---|
| `flowplace_v0/` | FlowPlace hard projection (`hard_constraint_legalizer.py`, class `Legalizer`) | applied once to the raw sample: `grid_res=128`, `scores={"w_reg": 0.1, "w_dist": 1.0}`, `padding=0.001` |
| `flowplace_soft/` | FlowPlace soft (gradient-based) legalizer (`legalization.py`, `legalize`, with `guidance.py`) | FlowPlace default `standard.yaml` (mode `scheduled`, step size 0.2, 5,000 steps, legality weight 1, HPWL weight 0) |
| `chipdiffusion/` | ChipDiffusion legalizer (`legalization.py`, `legalize_opt`, with `guidance.py`) | ChipDiffusion's `configs/legalizer/opt-adam.yaml` with `alpha_lr=8e-3`, `hpwl_weight=12e-5`, `legality_potential_target=0`, `grad_descent_steps=20000`, `macros_only=False`; sampling guidance not applied |

`flowplace_v0/` and `flowplace_soft/` are the same files as `flow_matching/hard_constraint_legalizer.py`,
`flow_matching/legalization.py`, `flow_matching/guidance.py` and `flow_matching/configs/legalizer/standard.yaml`.
ChipDiffusion's `legalization.py` differs from FlowPlace's by one comment line and imports ChipDiffusion's `utils` module
(not included). In the ablation, all legalized outputs go through the same DEF export and spacing repair before routing.
These files keep the authorship and license terms of FlowPlace and ChipDiffusion.
