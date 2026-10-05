# PEFT-on-ReID: Parameter-Efficient Fine-Tuning for Person Re-Identification on Vision Transformers

**Huzaifa Naseer**ᵃ, **Tameema Rehman**ᵃ, **Anas Ashfaq**ᵃ, **Farrukh Hasan Syed**ᵇ*

ᵃ Department of Computer Science, FAST National University of Computer and Emerging Sciences, National Highway, Karachi, Pakistan

ᵇ Department of Computer Science, School of Mathematics and Computer Science, Institute of Business Administration, Karachi, Pakistan

*Corresponding author: Farrukh Hasan Syed — fhsyed@iba.edu.pk

---

## Abstract

Vision Transformer (ViT) backbones such as TransReID achieve strong performance for person re-identification (Re-ID), but full fine-tuning is expensive in memory and compute. Parameter-efficient fine-tuning (PEFT) addresses this by freezing the backbone and learning a small set of task-specific parameters. We conduct a systematic, controlled comparison of five structurally distinct PEFT methods — Low-Rank Adaptation (LoRA), Scale and Shift Features (SSF), and the lightweight baselines BitFit, LayerNorm (LN) tuning, and bottleneck adapters — on TransReID evaluated under the standard Market-1501 protocol. Keeping the training recipe fixed across all runs, we freeze the ViT backbone and inject LoRA into transformer-block linear layers (qkv, proj, fc1, fc2), SSF scale-and-shift operations after four activation types (Attention, MLP, LayerNorm1, LayerNorm2), and bottleneck adapters at the same four linear targets; BitFit and LN-tuning train only bias terms or LayerNorm parameters respectively. We systematically vary (i) depth placement (blocks 0–11, 4–11, 6–11), (ii) LoRA rank r ∈ {8, 16, 32} and scaling α, and (iii) module targeting and optimizer configuration for SSF.

Results show that depth placement is the dominant factor for all methods. Bottleneck adapters provide the strongest efficiency point: 0–11 adapters (r=16) reach **85.9 mAP / 93.7 Rank-1** — within ~2 mAP of full fine-tuning (88.0) at only 4.95% trainable parameters — outperforming LoRA (best 85.8 mAP at 0–11 r=8) and SSF (79.9 mAP). LoRA at blocks 4–11 provides the best accuracy–memory compromise among weight-space methods (∼25–30% VRAM reduction, mAP within 5 points of the full fine-tuning baseline). SSF achieves the most compact trainable parameter footprint (∼2.87% of backbone parameters) but shows a larger accuracy gap. The lightweight baselines BitFit (76.6 mAP) and LN-tuning (65.9 mAP) fill the very-low-parameter-budget end of the frontier. A comparative analysis reveals that the methods form an accuracy–efficiency spectrum: adapters are preferred when accuracy recovery is paramount under a memory budget, SSF when parameter storage dominates, and BitFit/LN-tuning when only a tiny budget is available. We provide a configuration-effect map and practical design guidelines for selecting PEFT settings under GPU constraints.

**Code and resources:**
- LoRA: https://github.com/Huzaifa9559/LoRa-on-Transreid
- SSF: https://github.com/TameemaRehman/SSF_TransReID

**Keywords:** Person Re-Identification, Parameter-Efficient Fine-Tuning (PEFT), Vision Transformers (ViT), LoRA, SSF, BitFit, LayerNorm Tuning, Bottleneck Adapters

---

## 1. Introduction

Fine-tuning remains the dominant strategy for adapting large transformer backbones to downstream tasks, but updating all parameters is often expensive in memory, compute, and storage, particularly when multiple task-specific variants must be maintained. Parameter-efficient fine-tuning (PEFT) addresses this challenge by freezing most of a pretrained backbone and learning a small set of task-specific parameters [1]. Among PEFT methods, Low-Rank Adaptation (LoRA) introduces lightweight low-rank updates into existing linear layers [2], while Scale and Shift Features (SSF) [3] takes a structurally distinct approach, learning per-channel affine transformations applied element-wise to intermediate activations. Vision Transformers (ViTs) [4] have become a dominant backbone for visual recognition tasks [5]. While PEFT has been extensively studied in language models, its behavior in ViT-based systems remains less systematically explored [1]. In practice, PEFT performance depends on multiple interacting design choices: rank, scaling, layer targets, depth placement, and optimizer configuration [6, 2]. It is rarer still to find studies that systematically compare structurally different PEFT paradigms on the same frozen ViT backbone under matched conditions.

These questions are especially relevant for person re-identification (Re-ID) [7], where models must match identities across non-overlapping cameras despite variations in viewpoint, illumination, and occlusion. Modern Re-ID systems increasingly rely on transformer-based backbones such as TransReID [8], which extends ViT with task-specific components and achieves strong performance on Market-1501 [9]. Despite this progress, most studies assume full fine-tuning, and there is no existing work that applies and systematically compares multiple PEFT methods to a ViT-based Re-ID backbone. We address this gap by conducting a controlled comparison of LoRA and SSF on TransReID. We freeze the ViT backbone, inject PEFT modules into selected transformer layers, and run controlled experiments over depth placement, rank, scaling, module targeting, and optimizer configuration. Each configuration is evaluated on Market-1501 using mAP, CMC Rank-1/5/10, and peak GPU memory.

### Contributions:

1. We present a systematic comparison of five PEFT methods (LoRA, SSF, BitFit, LN-tuning, bottleneck adapters) on a ViT-based Re-ID backbone (TransReID), varying depth placement, rank, scaling, module targeting, and optimizer configuration across methods.
2. We quantify accuracy–memory trade-offs for all methods and construct a configuration-effect map with a non-dominated frontier analysis that includes the lightweight baselines as additional recommender options.
3. We derive practical guidelines for selecting PEFT configurations under GPU constraints, identifying stable and unstable regions in the combined configuration space and covering very small parameter budgets via BitFit/LN-tuning/adapters.

---

## 2. Related Work

### 2.1. Transformer-based Person Re-ID and TransReID

Person re-identification aims to match images of the same person across non-overlapping cameras under changes in viewpoint, illumination, and occlusion. Benchmarks such as Market-1501 [9] provide a standardized train/query/gallery protocol evaluated using mAP and CMC Rank-k accuracy. With the rise of Vision Transformers [4], transformer-based Re-ID has increasingly replaced CNN-centric pipelines [10]. TransReID [8] augments a ViT-Base backbone with Side-Information Embeddings (SIE) and a Jigsaw Patch Module (JPM), achieving strong performance on Market-1501. Surveys of transformer-based Re-ID identify TransReID as a canonical ViT-based architecture and typically assume full fine-tuning of all transformer blocks [10, 11]. Recent extensions include large-scale self-supervised pretraining [12] and cross-attention designs for occluded scenarios [13]. In our work, TransReID serves as a testbed: we retain its backbone and evaluation protocol, focusing on how to fine-tune the transformer efficiently via PEFT.

### 2.2. Parameter-Efficient Fine-Tuning (PEFT) and LoRA

PEFT freezes most of a pretrained backbone and trains a small set of task-specific parameters, reducing trainable parameter count and per-task storage [1]. LoRA [2] is among the most widely used PEFT methods: it keeps a frozen weight matrix W and learns a low-rank update BA added during training:

$$W' = W + \frac{\alpha}{r} BA \tag{1}$$

where r is the rank, α is a scaling factor, A ∈ ℝ^(r×k) and B ∈ ℝ^(d×r) are trainable. On LLMs, small ranks such as r ∈ {4, 8, 16} with appropriate scaling often match full fine-tuning within a few accuracy points while reducing trainable parameters by orders of magnitude [2]. Systematic LoRA studies on vision transformers remain comparatively rare [1].

### 2.3. PEFT for Vision Transformers

For Vision Transformers, PEFT has received focused attention only recently. Xin et al. [1] survey parameter-efficient fine-tuning for pretrained vision models and emphasize that not all ViT layers contribute equally to downstream adaptation, and PEFT modules are typically placed in attention and MLP linear layers. Sensitivity-Aware Visual PEFT (SPT) [14] formalizes this by concentrating capacity on mid-to-late layers, showing better parameter–accuracy trade-offs than uniform adaptation. These works motivate our choice to attach PEFT modules to linear projections {qkv, proj, fc1, fc2} inside ViT blocks.

### 2.4. LoRA Configuration Studies and Memory Footprint

The original LoRA paper notes that rank r and scaling factor α should not be tuned independently: common recipes set α ≈ 2r to keep the update magnitude stable when rank changes [2]. Most configuration studies are LLM-centric, leaving open how these factors jointly affect accuracy and efficiency in ViTs [1]. Our experimental grid, covering r ∈ {8, 16, 32}, α ∈ {16, 32, 48, 64}, and three block ranges, constitutes a configuration study tailored to a ViT-based Re-ID model.

Memory footprint in PEFT is governed primarily by activation storage during backpropagation rather than by the number of trainable parameters. Layers with trainable parameters must retain activations at full precision during the backward pass. This directly motivates our partial-layer LoRA experiments: LoRA on all 12 blocks retains activations throughout the full depth, whereas LoRA restricted to 4–11 or 6–11 allows early-block activations to be freed immediately, yielding measured drops from ≈11.5 GB to ≈7–8 GB VRAM.

### 2.5. Layer Selection: Early vs. Late Blocks

Layer-wise analyses show that early ViT layers learn generic features while later layers encode higher-level, task-specific semantics [1, 8]. SPT [14] shows that concentrating adapters in mid-to-late blocks yields better parameter–accuracy trade-offs than uniform coverage. Our three block ranges are designed around this: 0–11 as a maximal-coverage reference; 6–11 following the "tune last few layers" heuristic; and 4–11 as the deliberate compromise, extending adaptation into mid-depth blocks to recover accuracy while remaining cheaper than 0–11 in peak VRAM.

### 2.6. Scale and Shift Features: Method, Landscape, and Placement

SSF was introduced by Lian et al. [3] at NeurIPS 2022. Rather than adding weight matrices, SSF learns per-channel scale γ ∈ ℝ^d and shift β ∈ ℝ^d vectors applied element-wise to intermediate activations:

$$z' = \gamma \odot z + \beta \tag{2}$$

where z is the output of a frozen layer. For linear-preceded operations, the scale and shift can be absorbed into the frozen weight via reparameterization at inference, adding zero extra FLOPs at deployment [3]. The original SSF paper demonstrates that ∼0.3M tunable parameters outperform full fine-tuning on FGVC and VTAB-1k [3]. Han et al. [15] categorize SSF as a rank-independent, activation-space technique: unlike LoRA, SSF acts post-activation and is agnostic to the internal weight dimensions of the frozen layer it follows. In the ViT-PEFT-Vision benchmark [16], SSF is positioned as a strong parameter-efficient baseline competitive with LoRA on several task categories. Regarding placement, applying SSF after normalization layers can interact with LayerNorm operations already present in transformer blocks, a nuance relevant to our design across four operation types per block [17].

### 2.7. SSF Optimizer Sensitivity and Configuration

The original SSF paper uses specific training settings: AdamW with BaseLR = 3.5 × 10⁻⁴, WeightDecay = 10⁻⁴, and a bias learning rate factor of 2, which differ from standard ViT fine-tuning recipes [3]. We evaluate SSF under two configurations: Case 1 matches the LoRA training recipe for fair cross-method comparison; Case 2 follows the SSF paper's official settings. Our two-case design explicitly surfaces and controls for the optimizer confound, an aspect of experimental rigor often absent from PEFT comparison papers.

### 2.8. Summary and Link to Our Work

The gaps this work addresses are:

(i) TransReID and Market-1501 provide a strong ViT-based Re-ID testbed, but existing work assumes full fine-tuning and does not explore PEFT on this architecture [8, 9, 10];

(ii) systematic LoRA configuration sweeps linking rank, scaling, and block coverage to concrete accuracy–memory outcomes on a ViT-based Re-ID backbone are absent [2, 1];

(iii) SSF has not been applied to or compared against LoRA on a ViT-based Re-ID model [3]; and

(iv) placement sensitivity, memory footprint, and optimizer configuration are first-class design concerns for PEFT on ViTs [1, 14].

Our project addresses all four gaps using TransReID as a controlled testbed for two independent PEFT experimental phases.

---

## 3. Methodology

### 3.1. Base Architecture: TransReID with ViT-Base Backbone

We adopt TransReID as our base person Re-ID model, using a ViT-Base backbone comprising 12 transformer blocks indexed from 0 to 11. Each block consists of: (1) Multi-Head Self-Attention (MHSA) with learnable linear projections, (2) a Feed-Forward Network (FFN/MLP) with two linear layers and a non-linearity, and (3) LayerNorm and residual connections. TransReID augments the ViT backbone with SIE for camera/view cues, JPM for part-level feature learning, and a Re-ID head combining global and part-based features optimized using classification and metric-learning losses. All changes in this study are restricted to inserting PEFT modules into selected linear layers inside the transformer blocks; SIE, JPM, and the Re-ID head architecture remain unchanged. Evaluation follows the standard Market-1501 protocol, reporting mAP and CMC Rank-1/5/10.

Fig. 1 illustrates the architecture pipelines for both experimental phases.

> **Figure 1(a) — Exp. I: LoRA pipeline.** Low-rank adapter matrices (A, B) are injected into {qkv, proj, fc1, fc2} of each adapted block. Blocks 0–3 remain frozen in the 4–11 configuration shown. The Re-ID head (BNNeck, softmax classifiers, triplet loss) is fully trainable. Rank r ∈ {8, 16, 32}, scale α ∈ {16, 32, 48, 64}.
>
> Pipeline: Input Image → Patch Embedding (Linear Projection) → + Positional Embedding → SIE (Camera Embedding) → ViT-Base Encoder (12 Blocks; Blocks 0–3 frozen, Blocks 4–11 with LoRA on qkv, proj, fc1, fc2) → JPM (Jigsaw Patch Module) → Re-ID Head → Retrieval Embedding (final feature ∈ ℝ^d). Inside one transformer block: LayerNorm → MHSA (qkv projection [LoRA] → scaled dot-product attention → proj [LoRA]) → LayerNorm → MLP (fc1 [LoRA] → GELU → fc2 [LoRA]).

> **Figure 1(b) — Exp. II: SSF pipeline.** Scale-and-shift (SSF-ADA) modules y = γ ⊙ x + β are inserted after {LayerNorm1, Attention, LayerNorm2, MLP} in each adapted block. The ViT backbone is fully frozen; only γ, β ∈ ℝ⁷⁶⁸ vectors and the Re-ID head are trainable (≈2.87% of total parameters). SSF modules inside the JPM branches are re-frozen.
>
> Pipeline: Input image (256×128) → Patch embedding (frozen, stride [12,12]) with SSF patch embed (trainable γ, β) and SIE (camera embed) → ViT-B/16 backbone, 12 transformer blocks (all frozen). Representative Block 0: LayerNorm1 (frozen) → SSF-ADA after LN1 (trainable γ, β) → MHSA (frozen) → SSF-ADA after attention (trainable γ, β) → residual add → LayerNorm2 (frozen) → SSF-ADA after LN2 (trainable γ, β) → MLP fc1/fc2 (frozen) → SSF-ADA after MLP (trainable γ, β) → residual add. Blocks 1–11 identical placement (4 SSF modules each). JPM (last block, deep-copied ×2): Branch b1 (global, full token sequence, SSF frozen/re-frozen), Branch b2 (local, shuffled patch groups, SSF frozen/re-frozen). Final LayerNorm (frozen) → SSF final norm (trainable γ, β) → Re-ID head (fully trainable: BNNeck + softmax classifiers + triplet loss) → Identity features (Rank-1/mAP eval) and softmax+triplet loss (gradient to SSF + head only). Total trainable: 0.7501M (0.87% of backbone) — SSF + Re-ID head.

### 3.2. Dataset and Preprocessing

All experiments use Market-1501 [9]: 1,501 identities, 6 cameras, 751/750 train/test ID split, 3,368 queries, and 19,732 gallery images. We report mAP and CMC Rank-1/5/10 under the single-query protocol. The data pipeline is fixed across all runs: images resized to 256 × 128, random horizontal flip, Random Erasing, consistent Color Jitter, and ImageNet mean/std normalization. The same pipeline applies to both LoRA and SSF experiments without modification, ensuring that all performance differences across configurations are attributable solely to PEFT design choices.

### 3.3. Training Protocol and Measurements

All runs use a single NVIDIA RTX 4000 Ada GPU (20 GB VRAM) with an Intel Core i7-14700 CPU and 32 GB RAM. The training recipe is fixed at 60 epochs, AdamW optimizer, warm-up with cosine decay, and a batch size fitting the most memory-intensive configuration within 20 GB VRAM. The 60-epoch budget is consistent with standard practice for fine-tuning pretrained ViT backbones: fewer epochs are required compared to training from scratch, as the backbone already encodes useful generic features, and extending the schedule primarily risks overfitting rather than improving generalization on the training split [8]. For SSF, optimizer hyperparameters are set per Case 1 or Case 2 as described in Section 3.5. For each configuration we record final mAP, CMC Rank-1/5/10, peak GPU memory, approximate wall-clock time, and trainable parameter ratio. Random seeds are kept consistent across all runs to reduce variance.

### 3.4. LoRA-Based Parameter-Efficient Fine-Tuning

#### 3.4.1. Injection Sites and Trainable Parameters

Within each transformer block, LoRA modules are attached to the combined qkv projection, the attention output proj, and the two FFN layers fc1 and fc2 (default). These four linear projections are chosen because they account for the vast majority of parameter-intensive operations within a transformer block: qkv and proj govern all cross-token information routing in the MHSA sub-layer, while fc1 and fc2 implement the per-token non-linear transformation in the FFN sub-layer. Adapting all four therefore gives LoRA influence over both the relational reasoning capacity of the attention pathway and the representational capacity of the MLP pathway, which is critical for identity discrimination in Re-ID [1]. An attention-only ablation excluding fc1/fc2 is evaluated separately for the 6–11 regime to quantify the cost of omitting MLP adaptation. Trainable parameters consist of the LoRA matrices (A, B) for each selected layer, LayerNorm parameters, and the Re-ID heads; all original ViT weights, patch embeddings, and TransReID's SIE and JPM parameters remain frozen.

#### 3.4.2. LoRA Configuration Space

- **Depth (block coverage):** 0–11 (full-depth, all 12 blocks), 4–11 (middle+late blocks), and 6–11 (late-only blocks).
- **Rank:** r ∈ {8, 16, 32}.
- **Scaling:** α ∈ {16, 32, 48, 64}, with emphasis on proportional scaling α ≈ 2r. One aggressive configuration (6–11, r = 16, α = 64) is included to probe stability limits.
- **Module targets:** Attention+MLP (qkv, proj, fc1, fc2) as the default; attention-only (qkv, proj) as an ablation for the 6–11 regime.

The reference baseline is full fine-tuning without LoRA, where all TransReID weights are trainable.

### 3.5. SSF-Based Parameter-Efficient Fine-Tuning

#### 3.5.1. Injection Sites and Trainable Parameters

Within each adapted block, SSF operations are applied after four activation types: MHSA output (Attention), FFN output (MLP), output of LayerNorm1, and output of LayerNorm2. These four positions are chosen to provide complete coverage of all major information pathways within a transformer block. LayerNorm1 and the subsequent MHSA output together control the normalized input and relational output of the self-attention sub-layer; LayerNorm2 and the subsequent MLP output cover the normalized input and non-linear output of the feed-forward sub-layer. By applying SSF after all four, we ensure that affine modulation can recalibrate features at every stage of the block's computation, consistent with the original SSF paper's recommendation to apply broadly across all layer types rather than selectively [3]. Applying SSF after linear-preceded operations (Attention, MLP) additionally benefits from zero-overhead inference via reparameterization, while the LayerNorm-preceded positions still contribute only negligible parameter overhead [3]. Trainable parameters are the SSF scale (γ) and shift (β) vectors per targeted operation per adapted block, plus the Re-ID heads. The total trainable parameter ratio is ≈2.87% of total parameters for full 0–11 block coverage.

#### 3.5.2. SSF Configuration Space

SSF has no rank hyperparameter. Its configuration axes are:

- **Depth:** The same three block coverage regimes as LoRA (0–11, 4–11, 6–11).
- **Optimizer configuration:** Case 1 matches the LoRA training recipe (AdamW, BaseLR = 3.0 × 10⁻⁴, WeightDecay = 0.05) for fair cross-method comparison. Case 2 follows the SSF paper's official settings (AdamW, BaseLR = 3.5 × 10⁻⁴, WeightDecay = 10⁻⁴, BiasLRFactor = 2) [3].

### 3.6. Lightweight PEFT Baselines: BitFit, LN-Tuning, and Bottleneck Adapters

To broaden the comparison beyond weight-matrix (LoRA) and activation-affine (SSF) methods — and to give the recommender menu options for extremely small parameter budgets — we add three lightweight baselines, each applied to the same frozen ViT-Base backbone and the same training recipe.

- **BitFit.** Only the bias terms of the backbone are trained. In our implementation this includes linear-layer biases and LayerNorm shift (β) parameters, matching the canonical BitFit definition [19]. The head remains fully trainable. BitFit biases are optimized with the SSF-consistent learning-rate rule (10× base LR, zero weight decay) so that a bias-only update has a comparable effective step size.
- **LN-tuning.** Only LayerNorm parameters (γ, β) are trained — all `weight` and `bias` of every `nn.LayerNorm` in the adapted blocks, including the final pre-head norm. This isolates the contribution of the per-channel normalization affine transform without touching any weight matrix.
- **Bottleneck adapters.** A parallel bottleneck adapter `y = Wx + scale · U(GELU(Dx))` wraps each target linear (qkv, proj, fc1, fc2), with bottleneck dimension r = 16, down-projection `D` kaiming-initialized and up-projection `U` zero-initialized (identity at init, same trick as LoRA's zero-initialized `B`). The base linear is frozen.

All three methods are evaluated across the same three depth windows as LoRA/SSF (blocks 0–11, 4–11, 6–11), with the JPM branches frozen, mirroring the existing LoRA treatment. This yields nine additional configurations in the configuration–effect map.

### 3.7. Configuration–Effect Mapping

Each configuration (PEFT method, block range, r, α, module targets, optimizer case) is mapped to an outcome vector (mAP, Rank-1/5/10, peak VRAM, training time, trainable parameter ratio), populating multiple points along the accuracy–efficiency frontier for analysis.

---

## 4. Experiments and Results

### 4.1. Baseline: Full Fine-Tuning (Without PEFT)

The baseline TransReID model fully fine-tuned on Market-1501 for 60 epochs achieves **mAP = 88.0%, Rank-1 = 94.4%** with peak VRAM of **≈11.5 GB**. This baseline serves as the accuracy upper anchor and verifies that our training pipeline reproduces strong TransReID performance before introducing any PEFT method.

### 4.2. Effect of Depth Placement (LoRA)

**Late-only (6–11).** All configurations cluster around 7.0–7.6 GB peak VRAM, a roughly one-third reduction relative to the 11.5 GB baseline. Accuracy remains clearly below full fine-tuning: mAP typically in the mid-to-high 70s, Rank-1 in the high-80s to low-90s. Only the top six blocks adapt; blocks 0–5 remain frozen and contribute only generic pretrained features.

**Full-depth (0–11).** A conservative configuration (r=8, α=16) nearly matches baseline accuracy (mAP = 85.8%, Rank-1 = 93.5%) while consuming ≈11.4 GB. When every block contains trainable adapters, backpropagation must retain activations throughout the full depth, so peak memory is dominated by depth-wise gradient participation rather than by the small number of additional LoRA parameters.

**Middle+late (4–11).** Placing LoRA on blocks 4–11 provides the strongest accuracy–memory compromise. The best configuration (r=32, α=64) reaches mAP = 83.2%, Rank-1 = 92.8% at 7.84 GB, a 25–30% VRAM reduction with a substantially smaller accuracy gap than late-only (6–11). The improvement of 4–11 over 6–11 confirms that mid-depth blocks (4–5) contain adaptation-relevant representations that should not remain frozen.

**Depth placement summary:**
- **0–11:** "LoRA-style full adaptation" with minimal memory savings.
- **6–11:** largest memory reduction, largest accuracy loss.
- **4–11:** best compromise, retaining most baseline performance while significantly reducing VRAM.

### 4.3. Rank–Scaling Behavior Within Each Coverage Regime

**Full-depth (0–11).** Higher rank is detrimental. Increasing from r=8 to r=16 consistently reduces mAP and Rank-1, while VRAM changes only marginally. Injecting higher-capacity updates into every block over-perturbs pretrained features on a relatively small dataset.

**Late-only (6–11).** Moderate capacity increases help; aggressive scaling harms. Moving from (r=8, α=16) to (r=16, α=32) improves accuracy with nearly unchanged VRAM. However, pushing α=64 at r=16 causes a sharp accuracy collapse (mAP = 63.4%, Rank-1 = 81.9%), consistent with the LoRA update form (α/r)BA: excessively large α magnifies adapter updates and destabilizes adaptation.

**Middle+late (4–11).** Increasing capacity from (r=8, α=16) to (r=32, α=64) yields smooth improvement in mAP and Rank-1 while VRAM remains around 8 GB.

**Practical safe region:** Prefer partial depth coverage; use moderate α ≈ 2r or slightly above; avoid (r=16, α=64) when coverage is large.

### 4.4. Attention-Only vs. Attention+MLP Ablation

A controlled ablation within the 6–11 regime compares LoRA on {qkv, proj, fc1, fc2} vs. {qkv, proj} only. Both settings exhibit similar peak VRAM (difference ≈0.5 GB), but attention-only adaptation produces a noticeable drop in mAP and Rank-1. MLP layers contribute critical non-linear feature transformation important for identity discrimination. Including fc1/fc2 in LoRA is therefore beneficial when memory allows.

**Table 1: LoRA on Market-1501: Accuracy and GPU Memory vs. Full Fine-Tuning.** ΔmAP denotes the absolute gap from the baseline. Asterisk (*) marks the attention-only ablation (fc1, fc2 excluded).

| Blocks | r | α | mAP | Rank-1 | Rank-5 | Rank-10 | GPU (GB) | ΔmAP |
|---|---|---|---|---|---|---|---|---|
| Baseline (Full FT) | — | — | 88.0 | 94.4 | 98.2 | 99.0 | 11.5 | — |
| 0–11 | 8 | 16 | 85.8 | 93.5 | 98.0 | 98.9 | 11.4 | −2.2 |
| 0–11 | 16 | 16 | 75.5 | 88.5 | 95.8 | 97.4 | 10.6 | −12.5 |
| 0–11 | 16 | 32 | 74.0 | 87.6 | 94.9 | 97.2 | 11.0 | −14.0 |
| 4–11 | 8 | 16 | 80.5 | 91.5 | 96.9 | 98.2 | 8.03 | −7.5 |
| 4–11 | 16 | 32 | 82.1 | 92.4 | 97.2 | 98.5 | 8.34 | −5.9 |
| 4–11 | 16 | 48 | 82.7 | 92.5 | 97.5 | 98.6 | 8.01 | −5.3 |
| 4–11 | 32 | 64 | 83.2 | 92.8 | 97.8 | 98.5 | 7.84 | −4.8 |
| 6–11 | 8 | 16 | 74.8 | 88.2 | 96.0 | 97.7 | 7.60 | −13.2 |
| 6–11 | 16 | 16 | 75.3 | 89.2 | 96.1 | 97.5 | 6.90 | −12.7 |
| 6–11 | 16 | 32 | 77.4 | 90.1 | 96.5 | 97.9 | 7.59 | −10.6 |
| 6–11 | 16* | 32 | 74.6 | 88.5 | 96.3 | 97.7 | 7.06 | −13.4 |
| 6–11 | 16 | 64 | 63.4 | 81.9 | 93.3 | 95.8 | 7.00 | −24.6 |
| 6–11 | 32 | 64 | 78.9 | 90.6 | 96.8 | 98.4 | 7.06 | −9.1 |

### 4.5. SSF Results on Market-1501

**Full-depth (0–11).** Full block coverage achieves the best SSF accuracy. Under Case 1: mAP = 79.7%, Rank-1 = 91.0%. Under Case 2: mAP = 79.9%, Rank-1 = 91.1%. The optimizer difference is marginal (within 0.2% mAP and 0.1% Rank-1), confirming SSF is not highly sensitive to these hyperparameter differences on Market-1501, though Case 2 consistently achieves marginally better results.

**Middle+late (4–11).** Accuracy drops noticeably to mAP ≈ 74.5%, Rank-1 ≈ 88.0%. The drop from 0–11 to 4–11 is larger for SSF than for LoRA, suggesting SSF's feature-space modulation relies more heavily on early-block representations than LoRA's weight-space updates.

**Late-only (6–11).** The weakest configuration: mAP ≈ 68.9%, Rank-1 ≈ 84.7%. Late-only coverage is particularly limiting for SSF.

**Parameter efficiency vs. memory.** Despite a compact trainable parameter ratio (≈2.87% for 0–11 full coverage), peak GPU memory does not drop proportionally. Memory during training is dominated by intermediate activations retained for backpropagation, not by trainable parameter count. This confirms that parameter efficiency and memory efficiency are not interchangeable metrics.

**Table 2: SSF on Market-1501: Accuracy, GPU Memory, and Trainable Parameter Ratio.** Case 1: LoRA-matched optimizer (AdamW, BaseLR = 3.0 × 10⁻⁴, WD = 0.05). Case 2: SSF paper official settings (BaseLR = 3.5 × 10⁻⁴, WD = 10⁻⁴, BiasLRFactor = 2).

| Blocks | Case | mAP | Rank-1 | Rank-5 | GPU (GB) | Params (%) |
|---|---|---|---|---|---|---|
| Baseline (Full FT) | — | 88.0 | 94.4 | 98.2 | 11.5 | 100% |
| 0–11 | 1 | 79.7 | 91.0 | 96.9 | 10.1 | 2.87% |
| 0–11 | 2 | 79.9 | 91.1 | 97.1 | 10.0 | 2.83% |
| 4–11 | 1 | 74.1 | 88.0 | 95.7 | 9.43 | 2.85% |
| 4–11 | 2 | 74.5 | 88.0 | 95.8 | 9.45 | 2.85% |
| 6–11 | 1 | 68.5 | 84.3 | 94.4 | 9.18 | 2.83% |
| 6–11 | 2 | 68.9 | 84.7 | 94.6 | 9.25 | 2.83% |

### 4.6. Objective-Driven Adaptation Gap: Classification Control Experiment

To determine whether the performance gap between PEFT and Full Fine-Tuning is an inherent representational limitation of adapters on ViT backbones versus an objective-driven phenomenon of fine-grained metric learning, we conduct a controlled classification experiment (Framing 6). We replace the Re-ID metric learning head (triplet loss + BNneck + JPM + SIE) with a standard Softmax Cross-Entropy classification head attached directly to the global ViT feature vector. All 5 frontier configurations are evaluated under this identical setup on Market-1501 for 60 epochs.

**Table 4: Classification Control Results on Market-1501.** Re-ID auxiliary modules (triplet loss, BNneck, JPM, SIE) disabled; trained under Softmax Cross-Entropy loss.

Single seed (1234). Peak VRAM is lower than in Table 3 because BNNeck, JPM and SIE are disabled, so memory is not comparable across the two tables.

| Model / Configuration | Method | Block Coverage | Trainable Params (%) | Peak VRAM (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) | Rank-10 (%) |
|---|---|---|---|---|---|---|---|---|
| **Full FT Baseline** | Full FT | 0–11 | 100.00% | 7.91 GB | 80.5% | 92.1% | 97.5% | 98.4% |
| **LoRA 0–11 ($r=8, \alpha=16$)** | LoRA | 0–11 | **1.99%** | 8.14 GB | **82.0%** | **92.6%** | **98.0%** | **98.7%** |
| **LoRA 4–11 ($r=32, \alpha=64$)** | LoRA | 4–11 | **4.12%** | **5.68 GB** | **79.6%** | **90.7%** | **97.0%** | **98.6%** |
| **LoRA 6–11 ($r=16, \alpha=32$)** | LoRA | 6–11 | **1.99%** | **4.39 GB** | **72.8%** | **88.0%** | **95.9%** | **97.4%** |
| **SSF 0–11 (Case 2)** | SSF | 0–11 | **2.83%** | 9.38 GB | **74.0%** | **89.4%** | **97.0%** | **98.1%** |

#### Key Findings from the Classification Control Experiment:
1. **LoRA Outperforms Full Fine-Tuning Under Pure Classification:** LoRA 0–11 ($r=8$) achieves **82.0% mAP** and **92.6% Rank-1**, outperforming Full FT (80.5% mAP, 92.1% Rank-1) by **+1.5% mAP** and **+0.5% Rank-1** while updating only **1.99% of backbone parameters**.
2. **Evidence consistent with Framing 6:** The accuracy gap observed under the standard Re-ID protocol appears to depend on the training objective rather than being purely an architectural limitation of LoRA or the ViT backbone. This is a hypothesis: the experiment changes the whole head (triplet, BNneck, JPM, SIE) at once and does not isolate the triplet loss, and it uses a single seed. One possible explanation is that freezing 98%+ of the backbone acts as a regularizer under classification loss; we did not test this.
3. **Preservation of the Non-Dominated Frontier:** LoRA 4–11 ($r=32$) retains **79.6% mAP / 90.7% Rank-1** (within 0.9% of Full FT) while reducing peak VRAM from 7.91 GB to **5.68 GB** (a **28.2% memory savings**), supporting its status as the primary efficiency-compromise recommendation across objectives.

### 4.8. Lightweight PEFT Baselines on Market-1501

Following the same protocol, we evaluate the three lightweight baselines (BitFit, LN-tuning, bottleneck adapters) across the three depth windows (0–11, 4–11, 6–11), with the JPM branches frozen and the Re-ID head trainable (60 epochs, AdamW, softmax+triplet). Table 5 summarizes the results alongside the frontier points for reference.

**Table 5: Lightweight PEFT Baselines on Market-1501.** Same recipe as Table 3 runs; JPM branches frozen, head trainable. All runs: 60 epochs, AdamW, seed 1234, RTX 3060.

| Config | Method | Blocks | Trainable Params (%) | Peak VRAM (GB) | mAP (%) | Rank-1 (%) | Rank-5 (%) |
|---|---|---|---|---|---|---|---|
| bitfit 0–11 | BitFit | 0–11 | 2.89% (0.12% backbone) | 6.51 | 76.6 | 89.8 | 96.6 |
| bitfit 4–11 | BitFit | 4–11 | 2.85% (0.09% backbone) | 6.51 | 69.4 | 85.1 | 94.7 |
| bitfit 6–11 | BitFit | 6–11 | 2.84% (0.06% backbone) | 6.51 | 61.2 | 79.8 | 92.1 |
| lntune 0–11 | LN-tuning | 0–11 | 2.82% (0.04% backbone) | 6.51 | 65.9 | 83.7 | 94.8 |
| lntune 4–11 | LN-tuning | 4–11 | 2.81% (0.03% backbone) | 4.83 | 57.7 | 78.4 | 90.9 |
| lntune 6–11 | LN-tuning | 6–11 | 2.80% (0.02% backbone) | 3.99 | 48.0 | 71.5 | 87.0 |
| adapter 0–11 (r=16) | Bottleneck adapter | 0–11 | 4.95% | 8.03 | **85.9** | **93.7** | 97.8 |
| adapter 4–11 (r=16) | Bottleneck adapter | 4–11 | 4.24% | 5.77 | 80.5 | 90.8 | 97.2 |
| adapter 6–11 (r=16) | Bottleneck adapter | 6–11 | 3.88% | 4.67 | 74.8 | 87.8 | 95.8 |
| *(reference)* Full FT | — | 0–11 | 100% | 11.5 | 88.0 | 94.4 | 98.2 |
| *(reference)* 4–11 LoRA r=32 | LoRA | 4–11 | ~4.1% | 7.84 | 83.2 | 92.8 | 97.8 |
| *(reference)* 0–11 SSF Case 2 | SSF | 0–11 | 2.83% | 10.0 | 79.9 | 91.1 | 97.1 |

**Bottleneck adapters are the strongest lightweight method:** adapter 0–11 (r=16) reaches
**85.9 mAP / 93.7 Rank-1** — within ~2 mAP of full fine-tuning (88.0) at only 4.95% trainable
parameters — and outperforms both LoRA 0–11 r8 (85.8) and SSF 0–11 (79.9) on the same backbone.
This makes adapters a new point on the non-dominated frontier for accuracy-vs-efficiency.

The lightweight baselines extend the parameter-efficiency frontier to much smaller budgets than SSF: LN-tuning and BitFit train well under 1% of parameters, while bottleneck adapters (r=16) sit between LoRA and SSF in both parameter count and expected accuracy. These become additional options in the recommender menu for extremely constrained deployments (see Table 3).

### 4.9. Comparative Analysis and Design Guidelines

Aggregating all LoRA, SSF, and lightweight-baseline configurations with the baseline enables a structured comparison across the accuracy–efficiency landscape. A configuration cᵢ dominates cⱼ if:

$$\text{mAP}(c_i) \geq \text{mAP}(c_j) \text{ and } \text{VRAM}(c_i) \leq \text{VRAM}(c_j) \tag{3}$$

with at least one strict inequality. The non-dominated frontier contains:

- **Full fine-tuning:** best accuracy, highest VRAM (∼11.5 GB).
- **0–11 adapter (r=16):** the strongest efficiency point — **85.9 mAP at 4.95% trainable params and 8.03 GB**, within ~2 mAP of full FT; the best accuracy-to-parameter ratio of any PEFT method studied.
- **4–11 LoRA, r=32, α=64:** best LoRA trade-off at ≈7.8 GB (≈32% VRAM reduction) with mAP (83.2) and Rank-1 (92.8) close to baseline.
- **0–11 LoRA, r=8, α=16:** near-baseline accuracy at similar memory (85.8 mAP, 11.4 GB).
- **0–11 SSF, Case 2:** most parameter-efficient operating point, mAP = 79.9% at ≈2.83% trainable parameters.
- **0–11 BitFit:** the lightweight-budget option — 76.6 mAP at 2.89% total params (≈0.12% backbone), on 6.51 GB.
- **6–11 LoRA variants:** lowest-memory options (∼7 GB) with predictable but larger accuracy reductions.

The PEFT methods form a clear accuracy–efficiency spectrum. Bottleneck adapters recover most of the full fine-tuning accuracy at roughly one-third the VRAM, making them the default recommendation when accuracy recovery is paramount under a memory budget. LoRA at 4–11 offers the best balance of accuracy and memory among weight-space methods. SSF provides the smallest trainable footprint with zero inference FLOPs overhead via reparameterization, attractive when parameter storage or communication cost dominates (e.g., federated learning). BitFit and LN-tuning fill the very-low-parameter-budget end with modest accuracy. The methods also respond differently to restricted block coverage: LoRA and adapters recover well at 4–11, while SSF and the lightweight baselines show steeper drops under partial coverage, making full coverage more important for them. Table 3 translates these observations into practical configuration guidelines.

**Table 3: Practical Design Guidelines by Deployment Constraint.**

| Constraint | Recommended Config | Expected Outcome |
|---|---|---|
| Maximize accuracy | Full FT or 0–11 adapter r=16 | Full FT: mAP ≈ 88%, 11.5 GB; adapter: mAP ≈ 85.9%, 8.03 GB |
| ∼30% VRAM savings | 4–11 LoRA, r=16–32, α ≈ 2r | mAP ≈ 82–83%, 7.8–8.3 GB |
| Min. trainable params | 0–11 SSF (Case 2) | mAP ≈ 79.9%, ∼2.83% params |
| Tight GPU (∼7 GB) | 6–11 LoRA, r=16–32, α ≈ 2r | mAP ≈ 75–79%, 7–7.6 GB |
| Tiny parameter budget | 0–11 BitFit / LN-tuning | mAP ≈ 76.6% (BitFit) / 65.9% (LN), < 3% total params (see Table 5) |
| Mid efficiency | 4–11 adapter r=16 | mAP ≈ 80.5%, 4.24% params, 5.77 GB |
| Avoid instability | Keep α ≈ 2r; avoid r=16, α=64 | Smooth accuracy vs. rank curve |

---

## 5. Practical Implications and Deployment Relevance

### 5.1. Where This Work Is Useful in Real Systems

A key bottleneck in adapting transformer-based person re-identification models is the hardware cost of fine-tuning. This work is practically relevant in settings where teams need to adapt a strong ViT-based Re-ID backbone to new camera environments under limited resources, consistent with the broader motivation of PEFT for vision models [1, 2]. Typical use cases include:

- **Multi-camera person search / retrieval** in large facilities with non-overlapping cameras and changing viewpoints, aligned with benchmark protocols such as Market-1501 [9].
- **Re-ID inside tracking and camera-handoff pipelines**, where identity consistency must be preserved through occlusion and viewpoint changes, as emphasized in recent transformer Re-ID surveys [10, 11].
- **Multi-site deployments** (e.g., multiple branches or facilities) where each site exhibits domain shift and requires periodic re-tuning; transformer-based Re-ID literature highlights real-world variations including occlusion and clothing changes [10, 13].

### 5.2. What Our Findings Enable in Practice

Our experiments show that PEFT-based adaptation can substantially reduce peak VRAM compared to full fine-tuning while preserving competitive accuracy (see Section IV). In our setup, full fine-tuning achieves the strongest baseline but requires peak VRAM of approximately 11.5 GB, whereas partial LoRA adaptation operates around 7–8 GB VRAM. These findings support both LoRA and SSF as practical mechanisms for adapting large transformer backbones under constrained compute budgets [1, 2]. This has three direct deployment benefits:

- **Feasibility on smaller GPUs:** reduced peak VRAM enables adaptation on mid-range hardware, lowering dependence on high-end GPUs or costly cloud instances (supported by the measured VRAM reductions in Section IV) [1].
- **Maintainable updates:** lightweight adapter weights (LoRA) or merged affine vectors (SSF, via reparameterization) can be versioned and deployed per site or domain without maintaining multiple full backbone copies, consistent with adapter-style PEFT principles [1, 2].
- **Faster iteration cycles:** lower memory pressure reduces out-of-memory failures and increases experimental throughput, enabling more frequent re-tuning and broader ablations under fixed compute budgets [1].

### 5.3. Ethical and Privacy Considerations

Person Re-ID enables useful operational workflows but creates surveillance risk if misused. Deployments should enforce access control and audit logs, data minimization and retention limits, and human oversight for consequential decisions, along with robustness and bias evaluation where feasible [10, 11].

---

## 6. Limitations and Future Work

This study is deliberately scoped to one backbone (TransReID with ViT-Base), one dataset (Market-1501), and focused configuration grids for each PEFT method. This deliberate scope allows a clean, controlled comparison between two structurally distinct PEFT paradigms, but it also bounds the generality of the findings. We identify three concrete limitations and corresponding directions for future work.

**Single dataset evaluation.** All experiments (LoRA, SSF, BitFit, LN-tuning, adapters and the classification control) are conducted on Market-1501 only, with a single seed (1234) and no repeated runs. Cross-dataset generalization of SSF configuration patterns, for instance on larger or more challenging Re-ID benchmarks, remains untested. Future work should replicate the SSF block coverage sweep on additional datasets to determine whether the steep accuracy degradation under partial coverage (4–11, 6–11) is dataset-specific or a structural property of SSF on ViT backbones.

**Configuration grid boundaries.** The LoRA rank grid (r ∈ {8, 16, 32}) and scaling grid (α ∈ {16, 32, 48, 64}) were chosen to balance coverage with computational cost. Very low ranks (r < 8) and very high ranks (r > 32) are not evaluated; it remains open whether the instability observed at high rank in the 0–11 regime persists at r = 64 or higher. Similarly, the SSF configuration space currently has no operation-type ablation: we apply SSF uniformly after all four operation types per block. A future ablation isolating the contribution of LayerNorm-specific SSF modules (which cannot benefit from zero-FLOPs reparameterization) would clarify whether those positions contribute meaningfully or introduce unnecessary interaction with the existing normalization operations in the transformer block.

**Method coverage.** The comparison includes five PEFT methods (LoRA, SSF, BitFit, LN-tuning, bottleneck adapters), but each is studied as an independent method. Hybrid configurations that combine LoRA weight-space updates with SSF activation-space modulation, or that apply different PEFT methods to different block ranges, are not explored. Future work should investigate whether combining approaches yields additive benefits in accuracy or efficiency, and should extend the comparison to additional PEFT methods such as visual prompt tuning (VPT) and residual-style low-rank fine-tuning [18] under the same controlled testbed. The lightweight baselines (BitFit, LN-tuning, adapters) are additionally evaluated at a single configuration (bottleneck r=16, bias LR 10×) per depth window; per-method hyperparameter sweeps are left to future work.

---

## 7. Conclusion

We presented a systematic comparison of five PEFT methods (LoRA, SSF, BitFit, LN-tuning, bottleneck adapters) on the ViT-Base backbone of TransReID, evaluated on Market-1501. Systematically varying block coverage, rank, scaling, module targeting, and optimizer configuration under a unified 60-epoch training recipe, we constructed a configuration-effect map and comparative analysis across all methods.

Depth placement is the dominant factor across methods. For LoRA, the 4–11 mid+late regime provides the best compromise: approximately 25–30% VRAM reduction with mAP within 5–6 points of full fine-tuning. Moderate α ≈ 2r is stable across partial-coverage regimes; aggressive scaling destabilizes training. Adapting both attention and MLP layers consistently outperforms attention-only configurations. For SSF, full block coverage (0–11) is essential; partial coverage causes steeper accuracy drops than for LoRA. SSF achieves a compact trainable parameter footprint (≈2.87%) but a larger accuracy gap than the best LoRA configurations under matched training budgets. The lightweight baselines (BitFit, LN-tuning, bottleneck adapters) fill the very-low-parameter-budget end of the frontier, giving practitioners additional options between full LoRA and ultra-light tuning.

A unified analysis positions 4–11 LoRA (r=32, α=64) as the default recommendation when accuracy recovery and VRAM savings must both be achieved; 0–11 SSF (Case 2) as the ultra-parameter-efficient alternative; and 6–11 LoRA variants for the most resource-constrained deployments.

---

### Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author(s) used Claude, Perplexity and ChatGPT for literature review and paraphrasing of the text in the manuscript. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article.

---

## References

[1] Y. Xin, K. Xu, J. Yang, et al., Parameter-efficient fine-tuning for pre-trained vision models: A survey, arXiv preprint arXiv:2402.02242 (2024).

[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, et al., Lora: Low-rank adaptation of large language models., Iclr 1 (2) (2022) 3.

[3] D. Lian, D. Zhou, J. Feng, X. Wang, Scaling & shifting your features: A new baseline for efficient model tuning, in: Advances in Neural Information Processing Systems (NeurIPS), 2022.

[4] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16×16 words: Transformers for image recognition at scale, in: International Conference on Learning Representations (ICLR), 2021.

[5] O. Elharrouss, Y. Himeur, Y. Mahmood, S. Alrabaee, A. Ouamane, F. Bensaali, Y. Bechqito, A. Chouchane, Vits as backbones: Leveraging vision transformers for feature extraction, Information Fusion 118 (2025) 102951.

[6] N. J. Prottasha, U. R. Chowdhury, S. Mohanto, T. Nuzhat, A. A. Sami, M. S. Ali, M. S. I. Sobuj, H. Raman, M. Kowsher, O. O. Garibay, Peft a2z: parameter-efficient fine-tuning survey for large language and vision models, arXiv preprint arXiv:2504.14117 (2025).

[7] M. Ye, J. Shen, G. Lin, T. Xiang, L. Shao, S. C. Hoi, Deep learning for person re-identification: A survey and outlook, IEEE transactions on pattern analysis and machine intelligence 44 (6) (2021) 2872–2893.

[8] S. He, H. Luo, P. Wang, F. Wang, H. He, W. Li, W. Jiang, Transreid: Transformer-based object re-identification, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp. 15013–15022.

[9] L. Zheng, L. Shen, L. Tian, S. Wang, J. Wang, Q. Tian, Scalable person re-identification: A benchmark, in: Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1116–1124.

[10] M. Ye, J. Shen, X. Zhang, P. C. Yuen, S.-F. Chang, Transformer for object re-identification: A survey, arXiv preprint arXiv:2401.06960 (2024).

[11] N. Perwaiz, M. Shahzad, M. M. Fraz, Ubiquitous vision of transformers for person re-identification, Machine Vision and Applications 34 (27) (2023).

[12] B. Hu, X. Wang, W. Liu, Personvit: Large-scale self-supervised vision transformer for person re-identification, arXiv preprint arXiv:2408.05398 (2024).

[13] V. D. Nguyen, P. Mantini, S. K. Shah, Crossvit-reid: Cross-attention vision transformer for occluded cloth-changing person re-identification, in: Proceedings of the Asian Conference on Computer Vision (ACCV), 2024.

[14] H. He, J. Cai, J. Zhang, D. Tao, B. Zhuang, Sensitivity-aware visual parameter-efficient fine-tuning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.

[15] Z. Han, C. Gao, J. Liu, J. Zhang, S. Q. Zhang, Parameter-efficient fine-tuning for large models: A comprehensive survey, arXiv preprint arXiv:2403.14608 (2024).

[16] Y. Shi, et al., Revisiting visual prompt tuning for vision transformers, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.

[17] Y. Luo, W. Zhu, B. Zhao, Parameter efficient fine tuning for multi-scanner pet to pet reconstruction, in: Proceedings of the International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2024.

[18] W. Dong, et al., Low-rank rescaled vision transformer fine-tuning: A residual design approach, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024, pp. 16101–16110.