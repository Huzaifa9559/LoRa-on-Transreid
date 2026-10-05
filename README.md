![Python >=3.8](https://img.shields.io/badge/Python->=3.8-yellow.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA%20%7C%20SSF%20%7C%20Adapter-green.svg)
![Task](https://img.shields.io/badge/Task-Person%20Re--ID-orange.svg)

# PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers

This repository is a research fork of the official [**TransReID**](https://github.com/damo-cv/TransReID) codebase, extended with multiple parameter-efficient fine-tuning (PEFT) methods for the ViT-Base backbone.

> **PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers**
> Huzaifa Naseer, Tameema Rehman, Anas Ashfaq, Farrukh Hasan Syed
> *Department of Computer Science, FAST-NUCES, Karachi, Pakistan*

---

## 1. Overview

Vision Transformer (ViT) backbones such as TransReID achieve strong performance for person re-identification (Re-ID), but full fine-tuning is expensive in memory and compute. Parameter-efficient fine-tuning (PEFT) addresses this by freezing the backbone and learning a small set of task-specific parameters. We conduct a systematic comparison of five structurally distinct PEFT methods on TransReID evaluated under the standard Market-1501 protocol.

### Contributions

1. **Systematic PEFT Comparison:** We evaluate weight-matrix updates (LoRA), activation-affine transformations (SSF), parallel bottleneck adapters, and lightweight parameter tuning (BitFit, LN-tuning) on a frozen ViT-Base backbone.
2. **Classification Control (Framing 6):** We evaluate PEFT methods under pure Softmax Cross-Entropy classification, as a control suggesting that the PEFT accuracy gap in Re-ID is related to the metric-learning objective rather than solely to backbone capacity (single seed, Market-1501; the loss is not isolated, so this is a hypothesis).
3. **Automated Reproducibility:** Diagnostic inspectors, unit test suites, and automated benchmark runners enable full replication of all reported metrics.

---

## 2. PEFT Paradigms & Formulations

Within each adapted transformer block:

- **LoRA:** Injects low-rank decomposition matrices $W' = W + \frac{\alpha}{r}BA$ into linear projections `{qkv, proj, fc1, fc2}`.
- **SSF:** Applies per-channel scale and shift $y = \gamma \odot x + \beta$ post-activation after Attention, MLP, LayerNorm1, and LayerNorm2. Supports zero-FLOPs inference reparameterization.
- **Bottleneck Adapters:** Parallel bottleneck blocks $y = x + W_{up} \text{GELU}(W_{down} x)$ wrapping target linear layers ($r=16$, down Kaiming-init, up zero-init).
- **BitFit & LN-Tuning:** Freezes all backbone weights while training bias terms or LayerNorm parameters ($\gamma, \beta$) respectively.

The backbone weights, patch embeddings, Side-Information Embeddings (SIE), and Jigsaw Patch Module (JPM) parameters remain frozen; PEFT modules, LayerNorm parameters, and the Re-ID head remain trainable.

---

## 3. Configuration Space

The PEFT sweep is driven entirely from YAML (`configs/Market/`, `configs/classification/`).

| Axis | Values | Notes |
|------|--------|-------|
| **PEFT Method** | `lora`, `ssf`, `adapter`, `bitfit`, `lntune` | Selected via `PEFT.METHOD` |
| **Depth (blocks)** | `0–11`, `4–11`, `6–11` | Controlled via `BLOCKS` array; empty = all 12 blocks |
| **LoRA Rank $r$** | `8, 16, 32` | `PEFT.LORA.R`; recipe favors $\alpha \approx 2r$ |
| **LoRA Scaling $\alpha$** | `16, 32, 48, 64` | `PEFT.LORA.ALPHA` |
| **Adapter Bottleneck $r$** | `16` | Parallel bottleneck dimension (`PEFT.ADAPTER.R`) |
| **Module targets** | `{qkv, proj, fc1, fc2}` | Attention + MLP default targets |

Reference baseline: **full fine-tuning** (`PEFT.METHOD: 'none'`), all TransReID weights trainable.

---

## 4. Key Results (Market-1501)

### 4.1. Standard Person Re-ID Protocol

All runs: single NVIDIA GPU, 60 epochs, AdamW, cosine decay, batch size 64, a single fixed seed (`1234`, no repeated runs). Evaluated on Market-1501 under single-query protocol. `ΔmAP` denotes absolute gap from Full FT baseline.

| Blocks | Method | Config / Hyperparams | mAP | Rank-1 | Rank-5 | Rank-10 | GPU (GB) | Params (%) | ΔmAP |
|:------:|:------:|:--------------------:|:---:|:------:|:------:|:-------:|:--------:|:----------:|:----:|
| **Baseline (Full FT)** | Full FT | — | **88.0** | **94.4** | 98.2 | 99.0 | 11.5 | 100.0% | — |
| **0–11** | **Bottleneck Adapter** | r=16 | **85.9** | **93.7** | 97.8 | 98.8 | **8.03** | 4.95% | **−2.1** |
| 0–11 | LoRA | r=8, α=16 | 85.8 | 93.5 | 98.0 | 98.9 | 11.4 | 1.99% | −2.2 |
| **4–11** | **LoRA** | **r=32, α=64** | **83.2** | **92.8** | 97.8 | 98.5 | **7.84** | **4.12%** | **−4.8** |
| 4–11 | Bottleneck Adapter | r=16 | 80.5 | 90.8 | 97.2 | 98.3 | 5.77 | 4.24% | −7.5 |
| 0–11 | SSF | Case 2 | 79.9 | 91.1 | 97.1 | 98.1 | 10.0 | 2.83% | −8.1 |
| 0–11 | BitFit | — | 76.6 | 89.8 | 96.6 | 98.0 | 6.51 | 2.89% | −11.4 |
| 0–11 | LN-Tuning | — | 65.9 | 83.7 | 94.8 | 96.8 | 6.51 | 2.82% | −22.1 |

### 4.2. Classification Control Results (Framing 6)

Evaluated under Softmax Cross-Entropy loss directly on global ViT features (`NECK: 'no'`), removing pairwise metric learning losses. Single seed, Market-1501 only. Memory is lower than in Section 4.1 because BNNeck, JPM and SIE are disabled, so GPU figures are not comparable across the two tables.

| Configuration | Method | Block Coverage | Trainable Params (%) | GPU (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) |
|:-------------:|:------:|:--------------:|:--------------------:|:--------:|:-------:|:----------:|:----------:|
| Full FT Baseline | Full FT | 0–11 | 100.00% | 7.91 | 80.5 | 92.1 | 97.5 |
| **LoRA 0–11 (r=8, α=16)** | LoRA | 0–11 | **1.99%** | 8.14 | **82.0** | **92.6** | **98.0** |
| LoRA 4–11 (r=32, α=64) | LoRA | 4–11 | 4.12% | 5.68 | 79.6 | 90.7 | 97.0 |
| LoRA 6–11 (r=16, α=32) | LoRA | 6–11 | 1.99% | 4.39 | 72.8 | 88.0 | 95.9 |
| SSF 0–11 (Case 2) | SSF | 0–11 | 2.83% | 9.38 | 74.0 | 89.4 | 97.0 |

### Takeaways

- **Bottleneck Adapters reach near-baseline accuracy.** `0–11` adapter ($r=16$) achieves **85.9% mAP / 93.7% Rank-1** — within ~2 mAP of full fine-tuning at only 4.95% trainable parameters.
- **Depth placement dominates.** `4–11` LoRA ($r=32, \alpha=64$) is the best accuracy–memory compromise: **mAP 83.2% at 7.84 GB VRAM — a ~30% memory reduction.**
- **Classification control suggests an objective effect.** Under pure classification loss, LoRA 0–11 ($r=8$) scores **+1.5% mAP over Full FT (82.0% vs. 80.5%)**. This is consistent with the PEFT gap in Re-ID depending on the metric-learning objective, but it comes from a single seed and does not isolate the triplet loss, so it is not conclusive.

### Practical Guidelines

| Constraint | Recommended config | Expected outcome |
|------------|--------------------|------------------|
| Maximize accuracy | Full FT or `0–11` Adapter $r=16$ | Full FT: mAP ≈ 88%, 11.5 GB; Adapter: mAP ≈ 85.9%, 8.03 GB |
| ~30% VRAM savings | `4–11` LoRA, $r=32$, $\alpha=64$ | mAP ≈ 83.2%, 7.84 GB VRAM |
| Min. trainable params | `0–11` SSF (Case 2) | mAP ≈ 79.9%, ~2.83% params |
| Tight GPU (~7 GB) | `6–11` LoRA ($r=16–32$, $\alpha \approx 2r$) | mAP ≈ 75–79%, 7–7.6 GB VRAM |
| Tiny parameter budget | `0–11` BitFit | mAP ≈ 76.6%, ~0.12% backbone params |

---

## 5. Setup

### Installation

```bash
pip install -r requirements.txt
# torch / torchvision / timm / yacs / opencv-python
```

See [`CUDA_SETUP_GUIDE.md`](CUDA_SETUP_GUIDE.md) for a step-by-step CUDA/PyTorch setup and `check_cuda.py` to verify the GPU is visible.

### Prepare Market-1501

Download [Market-1501](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view), unzip, and place it under `data/`:

```
data
└── market1501
    ├── bounding_box_train/
    ├── bounding_box_test/
    └── query/
```

Automated dataset helper scripts are provided:

```bash
python datasets/download_market1501.py
```

### Pretrained ViT backbone

Download the ImageNet-pretrained [ViT-Base](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) and point `MODEL.PRETRAIN_PATH` in the config to it.

---

## 6. Running Experiments

Every PEFT run is driven by a single YAML configuration file.

### Single Model Training

```bash
# LoRA — blocks 4–11, r=32, α=64 (best compromise)
python train.py --config_file configs/Market/lora_blocks_4_11_r32.yml MODEL.DEVICE_ID "('0')"

# SSF — full depth (Case 2 official recipe)
python train.py --config_file configs/Market/ssf_0_11_case2.yml MODEL.DEVICE_ID "('0')"

# Bottleneck Adapters — full depth (r=16)
python train.py --config_file configs/Market/adapter_blocks_0_11_r16.yml MODEL.DEVICE_ID "('0')"

# Full fine-tuning baseline
python train.py --config_file configs/Market/baselines/vit_transreid.yml MODEL.DEVICE_ID "('0')"
```

### Automated Suite Runners

```bash
# Dry-run diagnostic check (CPU or GPU)
python tools/check_peft.py --config_file configs/Market/adapter_blocks_0_11_r16.yml --cpu-only

# Run full Lightweight PEFT Baselines suite (BitFit, LN-Tuning, Adapters)
python tools/run_experiment4.py

# Run Classification Control benchmark suite
python tools/run_classification_control.py

```

### Evaluation

Adapter-only checkpoints are saved when `PEFT.SAVE_ADAPTER_ONLY: True`; they are loaded automatically on top of the frozen backbone at test time.

```bash
python test.py --config_file configs/Market/lora_blocks_4_11_r32.yml \
    MODEL.DEVICE_ID "('0')" TEST.WEIGHT path/to/checkpoint.pth
```

### Unit Testing

```bash
python -m pytest tests/
```

---

## 7. Config Reference

```yaml
PEFT:
  METHOD: "lora" # "none" | "lora" | "ssf" | "bitfit" | "lntune" | "adapter"

  LORA:
    R: 32 # rank r
    ALPHA: 64 # scaling α (aim for α ≈ 2r)
    DROPOUT: 0.05
    TARGETS: ["qkv", "proj", "fc1", "fc2"] # ["qkv","proj"] = attention-only ablation
    BLOCKS: [4, 5, 6, 7, 8, 9, 10, 11] # empty/omit = all 12 blocks

  ADAPTER:
    R: 16 # bottleneck rank r
    DROPOUT: 0.05
    TARGETS: ["qkv", "proj", "fc1", "fc2"]
    BLOCKS: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

---

## 8. Citation

If you use this repository or its findings, please cite our study and the original TransReID:

```bibtex
@article{naseer2025peft,
  title   = {PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers},
  author  = {Naseer, Huzaifa and Rehman, Tameema and Ashfaq, Anas and Syed, Farrukh Hasan},
  year    = {2025}
}

@InProceedings{He_2021_ICCV,
  author    = {He, Shuting and Luo, Hao and Wang, Pichao and Wang, Fan and Li, Hao and Jiang, Wei},
  title     = {TransReID: Transformer-Based Object Re-Identification},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2021},
  pages     = {15013-15022}
}
```

---

## 9. Acknowledgement

Built on top of [TransReID](https://github.com/damo-cv/TransReID), which itself derives from [reid-strong-baseline](https://github.com/michuanhaohao/reid-strong-baseline) and [pytorch-image-models](https://github.com/rwightman/pytorch-image-models). The LoRA formulation follows Hu et al. (ICLR 2022), SSF follows Lian et al. (NeurIPS 2022), and BitFit follows Zaken et al. (ACL 2022).
