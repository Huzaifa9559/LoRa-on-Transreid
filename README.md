![Python >=3.8](https://img.shields.io/badge/Python->=3.8-yellow.svg)
![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.6-blue.svg)
![PEFT](https://img.shields.io/badge/PEFT-catalogue-green.svg)
![Task](https://img.shields.io/badge/Task-Person%20Re--ID-orange.svg)

# PEFT on TransReID — Configuration Catalogue for ViT Person Re-ID

Research fork of [**TransReID**](https://github.com/damo-cv/TransReID) with five parameter-efficient fine-tuning (PEFT) families on the ViT-Base backbone:

> **Selecting Parameter-Efficient Fine-Tuning Configurations for Transformer-Based Person Re-Identification**
> Huzaifa Naseer, Tameema Rehman, Anas Ashfaq, Muhammad Taaha, Farrukh Hasan Syed

Every manuscript PEFT row is a single YAML under [`configs/catalogue/market1501/`](configs/catalogue/market1501/). The same command runs any method; only the config file changes.

```bash
python train.py --config_file configs/catalogue/market1501/<row>.yml
python test.py  --config_file configs/catalogue/market1501/<row>.yml TEST.WEIGHT <checkpoint>
```

Companion SSF publication code (separate fork used for the reported SSF numbers): [TameemaRehman/SSF_TransReID](https://github.com/TameemaRehman/SSF_TransReID). This repository also includes an in-repo SSF path so reviewers can launch SSF from the same command; a bit-exact numerical match to the companion fork was not re-verified here.

Full result tables, parameter accounting, and config-to-result mapping: [`results.md`](results.md).

---

## 1. What this repository provides

| Method | Config key | Implementation |
|--------|------------|----------------|
| LoRA | `PEFT.METHOD: lora` | [`model/peft/lora.py`](model/peft/lora.py) |
| Bottleneck adapter | `PEFT.METHOD: adapter` | [`model/peft/lightweight.py`](model/peft/lightweight.py) |
| BitFit | `PEFT.METHOD: bitfit` | [`model/peft/lightweight.py`](model/peft/lightweight.py) |
| LayerNorm tuning | `PEFT.METHOD: lntune` | [`model/peft/lightweight.py`](model/peft/lightweight.py) |
| SSF | `PEFT.METHOD: ssf` | [`model/peft/ssf.py`](model/peft/ssf.py) + ViT hooks |
| Full fine-tuning | `PEFT.METHOD: none` | all weights trainable |

Shared Market-1501 PEFT recipe: [`configs/base/market1501_transreid.yml`](configs/base/market1501_transreid.yml) (ViT-Base, stride `[12,12]`, 256×128, SIE camera 3.0, AdamW 3e-4, weight decay 0.05, 5 warm-up epochs, cosine, batch 64, seed 1234, 60 epochs, evaluation at epoch 60, mean/std 0.5). The dataloader applies resize, flip, pad/crop, normalize, and random erasing; there is no colour jitter.

Legacy LoRA YAMLs that set `LORA.ENABLED: True` still work: `normalize_peft_config` copies them into `PEFT.LORA` and sets `PEFT.METHOD` to `lora`.

---

## 2. Key results (Market-1501)

Hardware for all reported runs: **NVIDIA RTX 4000 Ada (20 GB)**, Intel Core i7-14700, 32 GB RAM. Single seed 1234. PEFT scores are final-epoch results after 60 epochs; no intermediate checkpoint was selected on the test set.

| Family | Setting | mAP | Rank-1 | Memory | PEFT params P (M) |
|--------|---------|----:|-------:|--------|------------------:|
| Full-fine-tuning reference | 60-epoch reference | **88.0** | **94.4** | 11.5 (assumption A) | all |
| Bottleneck adapter | 0–11, r=16 | **85.9** | **93.7** | **8.03 GiB** | 2.36 |
| LoRA | 0–11, r=8, α=16 | **85.8** | **93.5** | 11.4 (assumption A) | 1.18 |
| LoRA | 4–11, r=32, α=64 | 83.2 | 92.8 | 7.84 (assumption A) | 3.15 |
| SSF | 0–11, Case 2, JPM off | 79.9 | 91.1 | 10.0 (assumption A) | 0.077 |
| BitFit | 0–11 | 76.6 | 89.8 | 6.51 GiB | 0.103 |
| LayerNorm tuning | 0–11 | 65.9 | 83.7 | 6.51 GiB | 0.038 |

Adapter 0–11 (85.9) and LoRA 0–11 r8 α16 (85.8) are within the paper’s 1.0 mAP reporting tolerance. Comparisons use **PEFT-only registered counts** (Section 4.7); trainable head parameters are reported separately in [`results.md`](results.md).

**Memory units.** Adapter, BitFit, and LN-tuning values are the maximum sampled total GPU memory, sampled every five seconds, converted MiB→GiB by dividing by 1024. LoRA, SSF, and full-fine-tuning values are shown as recorded (assumption A). Under assumption B (decimal GB display), those three families’ values in GiB are 6.9% smaller.

**Full-fine-tuning reference.** The 88.0 / 94.4 result is a 60-epoch run whose optimizer, learning rate, weight decay, schedule, and batch size were **not retained**. It is not established that it used the PEFT AdamW recipe. [`configs/Market/vit_transreid_stride.yml`](configs/Market/vit_transreid_stride.yml) is the public 120-epoch SGD configuration and is a different recipe. See `configs/catalogue/market1501/full_ft_public_120epoch_sgd_NOTE.yml`.

---

## 3. Trainability (inspected implementations)

Documented from the released code; do not “correct” trainability to match older README text.

| Method | Backbone LayerNorm | BNNeck affine | Notes |
|--------|--------------------|---------------|-------|
| LoRA | frozen | scale and bias frozen | Only LoRA `A`/`B` (+ optional LoRA bias) and classifiers trainable |
| Adapter | frozen | scale and bias frozen | Parallel bottleneck on `qkv,proj,fc1,fc2` |
| LN-tuning | γ/β in window trainable | scale and bias frozen | Final backbone norm flag controlled by config |
| BitFit | β in window trainable | biases trainable; scales frozen | Patch-embedding bias remains trainable in every window |
| SSF | composed with SSF after LN | scales trainable; biases frozen | Patch-embedding SSF trainable in every window; JPM disabled in reported runs |

In all cases BNNeck running statistics still update during training. JPM branches `b1`/`b2` stay frozen. With JPM enabled, backbone block 11 and the original final LayerNorm are **registered but not executed**; published parameter counts are registered counts that include those unused parameters.

---

## 4. Setup

```bash
pip install -r requirements.txt
# torch / torchvision / timm / yacs / opencv-python (unpinned in this repo)
```

Historical PyTorch / CUDA versions used for the manuscript runs were not recovered from environment files in this repository. For local verification of this integration, a CPU venv was exercised with **PyTorch 2.14.1** (macOS arm64). That is the current integration environment, not a claim about the historical training stack on the RTX 4000 Ada workstation.

Prepare Market-1501 under `./data/market1501/` (or let the optional downloader run when the training split is missing). Point `MODEL.PRETRAIN_PATH` at the ImageNet ViT-Base checkpoint [`jx_vit_base_p16_224-80ecf9dd.pth`](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth).

---

## 5. Running experiments

### Catalogue rows (recommended)

```bash
# Best observed full-depth LoRA (85.8 mAP)
python train.py --config_file configs/catalogue/market1501/lora_0_11_r8_a16.yml

# Best observed full-depth adapter (85.9 mAP)
python train.py --config_file configs/catalogue/market1501/adapter_0_11_r16.yml

# LoRA 4–11, r=32, α=64 (83.2 mAP) — same settings as the historically misnamed
# configs/Market/vit_transreid_stride_lora_blocks_6_11.yml
python train.py --config_file configs/catalogue/market1501/lora_4_11_r32_a64.yml

# SSF Case 2, JPM off
python train.py --config_file configs/catalogue/market1501/ssf_0_11_case2.yml

# BitFit / LN-tuning
python train.py --config_file configs/catalogue/market1501/bitfit_0_11.yml
python train.py --config_file configs/catalogue/market1501/lntune_0_11.yml
```

Evaluate:

```bash
python test.py --config_file configs/catalogue/market1501/lora_0_11_r8_a16.yml \
    TEST.WEIGHT ../logs/catalogue_lora_0_11_r8_a16/transformer_60.pth
```

### Legacy self-contained LoRA configs

```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora.yml
# blocks 4–11, r=32, α=64 (filename is historical; see comment inside the file)
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml
```

### Diagnostics

```bash
python tools/check_peft.py --config_file configs/catalogue/market1501/adapter_0_11_r16.yml --cpu-only
pytest tests/
```

Exploratory classification-control and mini-dataset configs live under [`configs/exploratory/`](configs/exploratory/) and are **not** manuscript results.

---

## 6. PEFT config reference

```yaml
PEFT:
  METHOD: 'lora'   # none | lora | adapter | bitfit | lntune | ssf
  LORA:
    R: 8
    ALPHA: 16
    DROPOUT: 0.05
    TARGETS: ["qkv", "proj", "fc1", "fc2"]
    BLOCKS: []                 # empty = configured 0–11
    TRAIN_HEAD: True
    MERGE_AT_EVAL: False       # keep False during training; use export_merged_lora_copy for inference export
    SAVE_ADAPTER_ONLY: True    # adapter export + prediction state; not a full resume checkpoint
```

---

## 7. Citation

```bibtex
@article{naseer2025peft,
  title   = {Selecting Parameter-Efficient Fine-Tuning Configurations
             for Transformer-Based Person Re-Identification},
  author  = {Naseer, Huzaifa and Rehman, Tameema and Ashfaq, Anas
             and Taaha, Muhammad and Syed, Farrukh Hasan},
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

## 8. Acknowledgement

Built on [TransReID](https://github.com/damo-cv/TransReID). LoRA follows Hu et al. (ICLR 2022); SSF follows Lian et al. (NeurIPS 2022); bottleneck adapters follow Houlsby et al. (2019) / He et al. (2022); BitFit follows Ben Zaken et al. (2022); LayerNorm tuning follows Qi et al. (2022).
