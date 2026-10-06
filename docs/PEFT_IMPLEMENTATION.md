# PEFT Implementation Notes

## Architecture

Five PEFT methods share one command and one config schema (`PEFT.METHOD`):

| Method | Injection | Config key | Checkpoint |
|--------|-----------|------------|------------|
| LoRA | Post-hoc `LoRALinear` wrappers | `PEFT.METHOD: lora` | Adapter export + prediction state |
| Adapter | Parallel bottleneck wrappers | `PEFT.METHOD: adapter` | Full / resume state dict |
| BitFit | Bias-only `requires_grad` mask | `PEFT.METHOD: bitfit` | Full / resume state dict |
| LN-tuning | LayerNorm affine mask | `PEFT.METHOD: lntune` | Full / resume state dict |
| SSF | `SSF` modules in `Block.forward()` | `PEFT.METHOD: ssf` | Full / resume state dict |

Manuscript rows: `configs/catalogue/market1501/`. Shared recipe: `configs/base/market1501_transreid.yml`.

```bash
python train.py --config_file configs/catalogue/market1501/<row>.yml
```

## Config flow

1. `merge_config_file()` resolves `_BASE_` inheritance in YAML.
2. `normalize_peft_config()` maps legacy `LORA.ENABLED`, treats `PEFT.METHOD` as authoritative (clears stale ENABLED flags from prior merges), applies SSF Case 2 optimizer overrides.
3. `make_model()` dispatches on `PEFT.METHOD`.

## Block coverage

Configured windows `0–11`, `4–11`, `6–11`. With JPM enabled, backbone block 11 and the original final LayerNorm are registered but unused; published counts are registered counts.

## Optimizer rules

- **LoRA / adapter / LN-tuning / full FT**: standard `BASE_LR` and `WEIGHT_DECAY` on trainable params.
- **SSF**: params with `ssf` in name get `LR = 10 × BASE_LR` (or `PEFT.SSF.LR` if > 0) and `weight_decay = 0`.
- **BitFit**: trainable biases get `LR = 10 × BASE_LR` and `weight_decay = 0` (released rule).
- **SSF Case 2** (`OPTIMIZER_CASE: 2`): `BASE_LR=3.5e-4`, `WEIGHT_DECAY=1e-4`, `BIAS_LR_FACTOR=2`.

## Checkpoint formats

- **LoRA adapter export** (`SAVE_ADAPTER_ONLY`):
  `{"format": "peft_adapter_v1", "adapters": ..., "prediction_state": ..., "meta": {rank, alpha, targets, blocks, ...}}`
  Load uses `copy_` into existing parameters; validates shapes.
- **Full resume**: `{"format": "full_resume_v1", "state_dict": ..., "epoch": ...}` (also accepts a bare state dict).

`MERGE_AT_EVAL` uses a reversible merge that never runs while `training=True`. Prefer `export_merged_lora_copy` for a one-way inference export. Published configs keep merging off.

## Companion code

- SSF publication fork: [TameemaRehman/SSF_TransReID](https://github.com/TameemaRehman/SSF_TransReID)
- In-repo SSF is for reviewer convenience; bit-exact match to the companion was not re-verified.
