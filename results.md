# Configuration catalogue and results (Market-1501)

Source of truth: the revised manuscript *Selecting Parameter-Efficient Fine-Tuning Configurations for Transformer-Based Person Re-Identification*.

## Hardware and measurement

- Workstation: NVIDIA **RTX 4000 Ada (20 GB)**, Intel Core i7-14700, 32 GB RAM.
- Seed: `SOLVER.SEED = 1234`.
- Epoch budget: 60; evaluation at epoch 60. Reported PEFT scores are final-epoch results; no intermediate checkpoint was selected using test results.
- **Adapter / BitFit / LN-tuning memory:** maximum sampled total GPU memory, sampled every five seconds, stored as `peak_vram_gib` (`peak MiB / 1024`). Unit: GiB.
- **LoRA / SSF / full-fine-tuning memory:** recorded as displayed by the monitoring tool (unit basis not retained). Tables show values as recorded (**assumption A**). Under **assumption B** (decimal GB), values in GiB are 6.9% smaller (×0.931).

## Shared PEFT recipe

Inherited from [`configs/base/market1501_transreid.yml`](configs/base/market1501_transreid.yml) unless a row overrides it:

| Setting | Value |
|---------|-------|
| Backbone | ViT-Base TransReID, stride `[12,12]`, 256×128 |
| Head | JPM on, SIE camera 3.0, BNNeck (SSF rows: JPM off) |
| Optimizer | AdamW, LR 3e-4, weight decay 0.05, warmup 5 + cosine |
| Batch / instances | 64 images, 4 per identity |
| Augmentation | flip, pad/crop, random erasing; **no colour jitter**; mean/std 0.5 |
| Dataset | Market-1501 |

SSF Case 2 additionally sets LR 3.5e-4, weight decay 1e-4, bias LR factor 2. SSF and BitFit trainable biases / SSF vectors use 10× base LR and weight decay 0 in the released optimizer.

## Parameter accounting

PEFT adaptation parameters (analytic, registered):

| Family | Formula |
|--------|---------|
| LoRA / Adapter | `P = 12288 · n · r` |
| SSF | `P = 6144 · n + 3072` |
| BitFit | `P = 8448 · n + 1536` |
| LN-tuning | `P = 3072 · n + 1536` |

`n` is the number of blocks in the **configured** window. With JPM enabled, block 11 and the original final LayerNorm are registered but unused; counts stay registered. Head parameters are separate: JPM five-classifier head 2,883,840 (+ 7,680 BNNeck affine if counted); BitFit also trains five BNNeck biases; SSF uses a single classifier (+ BNNeck scale).

## Config-to-result mapping

Command for every row:

```bash
python train.py --config_file configs/catalogue/market1501/<file>
```

| # | Family | Blocks / setting | mAP | R-1 | Memory | P (M) | Config file | Effective overrides |
|---|--------|------------------|----:|----:|--------|------:|-------------|---------------------|
| 1 | Full FT reference | 0–11 | 88.0 | 94.4 | 11.5 (A) | all | *(no recovered YAML)* | 60-epoch reference; optimizer settings not retained. Public 120-epoch SGD: `configs/Market/vit_transreid_stride.yml`. PEFT AdamW with `METHOD: none`: `full_ft_public_120epoch_sgd_NOTE.yml` (labelled as not historical). |
| 2 | LoRA | 0–11, r8 α16 | 85.8 | 93.5 | 11.4 (A) | 1.18 | `lora_0_11_r8_a16.yml` | `R=8 ALPHA=16 BLOCKS=[]` |
| 3 | LoRA | 0–11, r16 α16 | 75.5 | 88.5 | 10.6 (A) | 2.36 | `lora_0_11_r16_a16.yml` | `R=16 ALPHA=16` |
| 4 | LoRA | 0–11, r16 α32 | 74.0 | 87.6 | 11.0 (A) | 2.36 | `lora_0_11_r16_a32.yml` | `R=16 ALPHA=32` |
| 5 | LoRA | 4–11, r8 α16 | 80.5 | 91.5 | 8.03 (A) | 0.79 | `lora_4_11_r8_a16.yml` | `BLOCKS=[4..11] R=8 ALPHA=16` |
| 6 | LoRA | 4–11, r16 α32 | 82.1 | 92.4 | 8.34 (A) | 1.57 | `lora_4_11_r16_a32.yml` | `R=16 ALPHA=32` |
| 7 | LoRA | 4–11, r16 α48 | 82.7 | 92.5 | 8.01 (A) | 1.57 | `lora_4_11_r16_a48.yml` | `R=16 ALPHA=48` |
| 8 | LoRA | 4–11, r32 α64 | 83.2 | 92.8 | 7.84 (A) | 3.15 | `lora_4_11_r32_a64.yml` | Same as legacy `configs/Market/vit_transreid_stride_lora_blocks_6_11.yml` (misnamed; blocks **4–11**) |
| 9 | LoRA | 6–11, r8 α16 | 74.8 | 88.2 | 7.60 (A) | 0.59 | `lora_6_11_r8_a16.yml` | `BLOCKS=[6..11]` |
| 10 | LoRA | 6–11, r16 α16 | 75.3 | 89.2 | 6.90 (A) | 1.18 | `lora_6_11_r16_a16.yml` | |
| 11 | LoRA | 6–11, r16 α32 | 77.4 | 90.1 | 7.59 (A) | 1.18 | `lora_6_11_r16_a32.yml` | |
| 12 | LoRA | 6–11, r16 α32 attn-only | 74.6 | 88.5 | 7.06 (A) | 0.44 | `lora_6_11_r16_a32_attn_only.yml` | `TARGETS=[qkv,proj]` |
| 13 | LoRA | 6–11, r16 α64 | 63.4 | 81.9 | 7.00 (A) | 1.18 | `lora_6_11_r16_a64.yml` | α/r=4; single such run |
| 14 | LoRA | 6–11, r32 α64 | 78.9 | 90.6 | 7.06 (A) | 2.36 | `lora_6_11_r32_a64.yml` | |
| 15 | SSF | 0–11 Case 1 | 79.7 | 91.0 | 10.1 (A) | 0.077 | `ssf_0_11_case1.yml` | `JPM=False OPTIMIZER_CASE=1` |
| 16 | SSF | 0–11 Case 2 | 79.9 | 91.1 | 10.0 (A) | 0.077 | `ssf_0_11_case2.yml` | `JPM=False OPTIMIZER_CASE=2` |
| 17 | SSF | 4–11 Case 1 | 74.1 | 88.0 | 9.43 (A) | 0.052 | `ssf_4_11_case1.yml` | |
| 18 | SSF | 4–11 Case 2 | 74.5 | 88.0 | 9.45 (A) | 0.052 | `ssf_4_11_case2.yml` | |
| 19 | SSF | 6–11 Case 1 | 68.5 | 84.3 | 9.18 (A) | 0.040 | `ssf_6_11_case1.yml` | |
| 20 | SSF | 6–11 Case 2 | 68.9 | 84.7 | 9.25 (A) | 0.040 | `ssf_6_11_case2.yml` | |
| 21 | BitFit | 0–11 | 76.6 | 89.8 | 6.51 GiB | 0.103 | `bitfit_0_11.yml` | |
| 22 | BitFit | 4–11 | 69.4 | 85.1 | 6.51 GiB | 0.069 | `bitfit_4_11.yml` | |
| 23 | BitFit | 6–11 | 61.2 | 79.8 | 6.51 GiB | 0.052 | `bitfit_6_11.yml` | |
| 24 | LN-tuning | 0–11 | 65.9 | 83.7 | 6.51 GiB | 0.038 | `lntune_0_11.yml` | |
| 25 | LN-tuning | 4–11 | 57.7 | 78.4 | 4.83 GiB | 0.026 | `lntune_4_11.yml` | |
| 26 | LN-tuning | 6–11 | 48.0 | 71.5 | 3.99 GiB | 0.020 | `lntune_6_11.yml` | |
| 27 | Adapter | 0–11 r16 | 85.9 | 93.7 | 8.03 GiB | 2.36 | `adapter_0_11_r16.yml` | |
| 28 | Adapter | 4–11 r16 | 80.5 | 90.8 | 5.77 GiB | 1.57 | `adapter_4_11_r16.yml` | |
| 29 | Adapter | 6–11 r16 | 74.8 | 87.8 | 4.67 GiB | 1.18 | `adapter_6_11_r16.yml` | |

Wall-clock time was recorded only for adapter / BitFit / LN-tuning (see manuscript Table 4). Rank-10 was not retained for SSF.

## Batch runner (adapter / BitFit / LN-tuning)

```bash
python tools/run_experiment4.py
```

Logs under `logs/experiment4/`. Progress field `peak_vram_gib` is the GiB sample described above.

## Code links

| Component | Location |
|-----------|----------|
| LoRA, adapter, BitFit, LN-tuning | this repository (`model/peft/`) |
| SSF companion (publication) | [TameemaRehman/SSF_TransReID](https://github.com/TameemaRehman/SSF_TransReID) |
| In-repo SSF (reviewer convenience) | `model/peft/ssf.py` + catalogue `ssf_*.yml` |
| Fixed integrated release | tag `peft-catalogue-v1` on this repository (created at integration) |

Exploratory configs under `configs/exploratory/` are excluded from the manuscript catalogue.
