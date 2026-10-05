# Experiment 4 — Three Lightweight PEFT Baselines (BitFit, LN-Tuning, Bottleneck Adapter)

> Status: **implemented** (see `docs/IMPLEMENTATION_EXPERIMENT4.md` — config plumbing, model
> module, factory dispatch, optimizer rule, 9 configs, runner, tests). Experiment runs on vast.ai.
>
> **Deployment target: vast.ai (git-clone workflow).** This experiment is executed on a vast.ai
> GPU instance by `git clone`-ing this branch, running the setup once, then executing all runs
> **sequentially** via a single runner script that streams to a log file you can tail for
> constant monitoring. See §4.

## 1. Goal

Answer Reviewer Concern "two-method scope" and strengthen the Framing-C recommender story.
Add three parameter-efficient fine-tuning methods to the **same frozen ViT-Base backbone** with the
**same training recipe**, each run across the **same three depth windows** as LoRA/SSF
(`0–11`, `4–11`, `6–11`):

1. **BitFit** — train only the **bias** terms of the backbone.
2. **LN-tuning** — train only the **LayerNorm** (gamma/beta) parameters.
3. **Bottleneck adapter** — classic `down-proj → non-linearity → up-proj` adapter inserted in each block.

These are not extra appendices — under Framing C they become additional options in the
recommender's menu, so a practitioner with a tiny parameter budget can look up BitFit in the table.

**Cost:** 3 methods × 3 depth windows = **9 runs**, 1 seed. Additional seeds only for configs that
land on the non-dominated frontier (per the review's "multi-seed" ask).

## 2. What Already Exists (reuse, don't rebuild)

- **Depth-window machinery is generic already.** LoRA uses `PEFT.LORA.BLOCKS` with
  `model/peft/lora.py::inject_lora_into_vit` + `make_model` (`include_blocks`). SSF uses
  `PEFT.SSF.BLOCKS` + `ssf_blocks` in the ViT. All three depth windows are already expressed as
  configs (`0-11`, `4-11`, `6-11`).
- **Backbone-freeze + head-trainable flow:** `model/peft/lora.py::mark_trainable_lora_and_head`
  and `model/make_model.py::_freeze_non_ssf` are the two existing "freeze everything, keep
  select params + head" patterns.
- **PEFT-aware optimizer:** `solver/make_optimizer.py` already gives **SSF params `lr*10`** and
  **bias params `lr * BIAS_LR_FACTOR`** (relevant for BitFit — see §6.2). SSF params get
  weight-decay 0.
- **Evaluation / runner pattern:** `tools/run_classification_control.py` loops configs → `make_model` → `make_loss` →
  `make_optimizer` → `do_train`, then report `param_ratio / peak_vram / time_seconds`.
- **Config pattern:** YAML + `_BASE_` inheritance → `config/loader.merge_config_file` →
  `config/peft_config.normalize_peft_config` (mutual-exclusion + SSF Case-2 overrides).

## 3. Exact Changes to Make

### 3.1 Config schema — `config/defaults.py`

Add one new node alongside the existing `PEFT.LORA` / `PEFT.SSF` nodes:

```yaml
# in _C.PEFT
PEFT.LNTUNE = CN()          # LN-tuning
PEFT.LNTUNE.ENABLED = False
PEFT.LNTUNE.BLOCKS = ()     # () = all blocks; [0..11] / [4..11] / [6..11]
PEFT.LNTUNE.TRAIN_FINAL_NORM = True   # include the final LayerNorm before the head

PEFT.BITFIT = CN()          # BitFit
PEFT.BITFIT.ENABLED = False
PEFT.BITFIT.BLOCKS = ()     # () = all blocks; [0..11] / [4..11] / [6..11]

PEFT.ADAPTER = CN()         # Bottleneck adapter
PEFT.ADAPTER.ENABLED = False
PEFT.ADAPTER.R = 16         # bottleneck dimension (down-project size)
PEFT.ADAPTER.DROPOUT = 0.0
PEFT.ADAPTER.BLOCKS = ()    # () = all blocks; [0..11] / [4..11] / [6..11]
PEFT.ADAPTER.SCALE = 1.0    # optional output scale (init), like LoRA alpha/r
```

Also extend `PEFT.METHOD`'s allowed values comment: `'none' | 'lora' | 'ssf' | 'lntune' | 'bitfit' | 'adapter'`.

> **Don't break legacy.** Keep `PEFT.LORA` and `PEFT.SSF` nodes untouched. The legacy `_C.LORA`
> node and `_copy_legacy_lora` migration stay as-is.

### 3.2 Config normalization — `config/peft_config.py`

Extend `normalize_peft_config(cfg)`:

- Resolve `PEFT.METHOD` from the new `ENABLED` flags (same pattern as SSF: if
  `PEFT.LNTUNE.ENABLED and METHOD == 'none'` → `METHOD = 'lntune'`, etc.).
- Keep the **mutual-exclusion rule**: raise `ValueError` if more than one of
  lora / ssf / lntune / bitfit / adapter is enabled.
- `get_peft_method` and `get_active_blocks` (add lntune/bitfit/adapter to the block resolution)
  need updates so `make_model` and `make_optimizer` see the right method.

### 3.3 New PEFT module — `model/peft/lightweight.py` (new file)

**BitFit.** Unfreeze only bias params in the selected blocks (plus classifier + BNNeck — these must
stay trainable, same as LoRA's `TRAIN_HEAD=True`). Two options:
- Block-scoped BitFit: unfreeze biases only in blocks in `PEFT.BITFIT.BLOCKS` (matching the
  depth-window framing).
- Implement as a freeze-sweep over `model.named_parameters()`, keeping every param whose name
  contains `bias` within an active block. Params with `bias` that live in the head are kept trainable.

```python
def mark_trainable_bitfit(model, blocks=None, train_head=True):
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if "bias" in name and _in_blocks(name, blocks) ...:  # keep trainable
    # then unfreeze classifier / bnneck like mark_trainable_lora_and_head
```

**LN-tuning.** Same sweep but keep params in modules whose type is `nn.LayerNorm` (weight & bias =
gamma & beta) within the selected blocks, plus final norm and head.

**Bottleneck adapter.** A new `AdapterLinear` (or `Adapter` module) wrapping each target
`nn.Linear`, mirroring `LoRALinear`'s structure:

```python
class BottleneckAdapter(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, dropout: float = 0.0, scale: float = 1.0):
        self.base = base           # frozen
        self.down = nn.Linear(in_features, r, bias=False)
        self.up   = nn.Linear(r, out_features, bias=False)
        self.act  = nn.GELU()      # or ReLU — pick one, document it
        self.drop = ...            # optional dropout on the residual path
    def forward(self, x):
        return self.base(x) + self.scale * self.up(self.act(self.down(x)))
```

- `inject_adapters_into_vit(model, r, targets, blocks, ...)` mirrors
  `inject_lora_into_vit` (reuse `iter_linear_targets` + `set_module_by_name` from `lora.py`).
- **Adapter checkpointing**: unlike LoRA there's no merge-into-base option; save full state dict
  (or adapter-only state dict with `adapter_state_dict` / `load_adapter_state_dict`, mirroring
  `lora_state_dict`). Since the base is frozen + ImageNet-pretrained, adapter-only saves are safe
  and match the repo's `SAVE_ADAPTER_ONLY` philosophy.

### 3.4 Model factory — `model/make_model.py`

Extend the PEFT dispatch in `make_model` (and the `build_transformer*` constructors):

- `if peft_method == 'lntune'`: call `mark_trainable_lntune(model, blocks=..., train_head=True)`.
- `if peft_method == 'bitfit'`: call `mark_trainable_bitfit(model, blocks=..., train_head=True)`.
- `if peft_method == 'adapter'`: call `inject_adapters_into_vit(model.base, ...)` then
  `mark_trainable_adapters_and_head(...)` (freeze base, keep down/up + head trainable).
- All three must **freeze the backbone** (same as LoRA) and print the same
  `trainable / total params (%)` line so `tools/*` can pick it up.

### 3.5 Optimizer — `solver/make_optimizer.py`

Current code gives **SSF params `lr*10` and bias params `lr * BIAS_LR_FACTOR`**. Decide & document:
- **BitFit**: biases get the standard `BIAS_LR_FACTOR` treatment (already implemented). Decide
  whether BitFit biases should also get `lr*10` like SSF — recommend matching SSF's
  `lr*10, weight_decay=0` so it's consistent with the SSF paper recipe and gives BitFit a fair shot.
- **Adapter**: down/up params use the normal base LR + weight decay (like LoRA A/B).
- **LN-tuning**: gamma/beta are `weight`/`bias` of LayerNorm; make sure they use a sensible LR
  (base LR is fine; consider whether to add to the SSF-style 10x group — document the choice).

### 3.6 Checkpoint save/load — `processor/processor.py`

`_save_checkpoint` currently branches on `get_peft_method(cfg) == 'lora'`. Extend to save
adapter-only for `adapter` method too (or full state dict for bitfit/lntune — full dict is simpler
since everything else is frozen).

### 3.7 Configs — `configs/Market/` (9 new files, matching existing naming)

For each method × each depth window (mirror `configs/Market/lora_blocks_*` style):

```
configs/Market/bitfit_blocks_0_11.yml
configs/Market/bitfit_blocks_4_11.yml
configs/Market/bitfit_blocks_6_11.yml
configs/Market/lntune_blocks_0_11.yml
configs/Market/lntune_blocks_4_11.yml
configs/Market/lntune_blocks_6_11.yml
configs/Market/adapter_blocks_0_11_r16.yml
configs/Market/adapter_blocks_4_11_r16.yml
configs/Market/adapter_blocks_6_11_r16.yml
```

Each inherits `configs/base/market1501_transreid.yml` (60 epochs, AdamW, softmax+triplet,
JPM+SIE on, same data pipeline) and only overrides the PEFT node + `OUTPUT_DIR`
(`../logs/market_vit_transreid_bitfit_0_11`, etc.).

Depth window semantics:
- `0-11` → `BLOCKS: []` or `BLOCKS: [0..11]` (match what the LoRA configs do — LoRA uses explicit
  `[0..11]` in the 0-11 config and `[4..11]`/`[6..11]` for the others).
- `4-11` → `BLOCKS: [4,5,6,7,8,9,10,11]`
- `6-11` → `BLOCKS: [6,7,8,9,10,11]`

**Decisions to lock before writing configs:**
- Adapter bottleneck dim `r` (16 is the common default — matches LoRA's mid-rank).
- Whether to also run the 5 existing classification-control configs for these 3 methods (recommend
  yes later, for the Sec 4.6 objective-vs-architecture argument, but **not required** for the 9
  main runs).
- Output naming: `../logs/market_vit_transreid_{method}_{window}`.

### 3.8 Runner — `tools/run_experiment4.py` (new file, vast.ai orchestrator)

**This is the piece that runs on vast.ai.** It replaces the "loop-in-process" pattern of the
existing `tools/run_classification_control.py` with a **sequential, resumable, log-streaming
orchestrator** designed for long unattended GPU runs:

**Behavior**
- Runs the 9 configs **one at a time, in order** (never parallel — the instance has one GPU).
- Each run invokes `train.py` **as a subprocess** with the config + seed, so:
  - a crash in one run doesn't take down the whole experiment;
  - the runner records `returncode` and moves on (or stops, configurable);
  - logs from `train.py` stream into a per-run file.
- **Per-run log files** in `logs/experiment4/`:
  - `run_01_bitfit_0_11.log`, `run_02_bitfit_4_11.log`, … (`train.py`'s stdout/stderr, live).
  - `progress.json` — one line per finished run: `{config, seed, returncode, trainable%, peak_vram, wall_time, mAP, R1}` (parsed from the train log tail).
- **Resumable:** `progress.json` is read at startup; completed runs are **skipped** unless
  `--force`. If the instance dies mid-run, re-running the script picks up where it left off.
- **Seed-aware:** `--seed` (default `1234`) is appended as a `train.py` opt override
  (`SOLVER.SEED <n>`). Multi-seed re-runs for frontier configs = just `--seed 5678 --only lntune`.
- **Filtering:** `--only METHOD[,WINDOW]` and `--only-run N` for quick single-config debugging
  (`--only-run 1 --epochs 2` sanity checks on the GPU before the long haul).

**Suggested CLI**

```bash
python tools/run_experiment4.py                      # all 9, 1 seed, sequential
python tools/run_experiment4.py --only-run 1 --epochs 2   # smoke-check bitfit 0-11 first
python tools/run_experiment4.py --seed 5678 --only lntune # extra seed for frontier configs
```

**Run order (also the log numbering)**

| # | Config | Method | Window |
|---|--------|--------|--------|
| 1 | `bitfit_blocks_0_11.yml` | BitFit | 0–11 |
| 2 | `bitfit_blocks_4_11.yml` | BitFit | 4–11 |
| 3 | `bitfit_blocks_6_11.yml` | BitFit | 6–11 |
| 4 | `lntune_blocks_0_11.yml` | LN-tuning | 0–11 |
| 5 | `lntune_blocks_4_11.yml` | LN-tuning | 4–11 |
| 6 | `lntune_blocks_6_11.yml` | LN-tuning | 6–11 |
| 7 | `adapter_blocks_0_11_r16.yml` | Adapter | 0–11 |
| 8 | `adapter_blocks_4_11_r16.yml` | Adapter | 4–11 |
| 9 | `adapter_blocks_6_11_r16.yml` | Adapter | 6–11 |

**Metrics collection**
- `train.py`'s final eval (mAP, Rank-1) is printed to the run's log; the runner greps the last
  `mAP:` / `CMC curve, Rank-1` lines into `progress.json` after `returncode == 0`.
- Param % / peak VRAM / wall time are captured around each `train.py` subprocess, matching the
  existing `tools/` runners' report columns.

**What it does NOT do**
- Does not re-implement training (delegates to `train.py`, the same entry point Experiments 1–3 used).
- Does not call `run.py`'s setup — setup is one-time on the instance (see §4.2).

### 3.9 Docs — `research.md` (and possibly `review.md`)

- **Abstract / Contributions:** change "comparative PEFT study (LoRA vs SSF)" framing to include
  BitFit, LN-tuning, adapters as menu options.
- **Methodology §3.x:** add one short subsection per new method (what gets trained, where it's
  injected, init).
- **Experiments §4.x:** new subsection "Lightweight PEFT Baselines (BitFit, LN-tuning,
  Adapters)" with the 9-run table + a note that these are new non-dominated options for the
  recommender.
- **Comparative Analysis / Table 3 (design guidelines):** add the 3 methods to the
  accuracy-vs-parameter-budget table so practitioners can look up BitFit at low budgets.
- **Limitations:** update the "two-method scope" limitation.

## 4. vast.ai Deployment & Monitoring (how the experiment actually runs)

### 4.1 Execution model

The experiment runs on a **vast.ai GPU instance**, not locally. The flow is:

1. **Local:** commit & push this branch (`experiment4-taaha`).
2. **vast.ai:** `git clone -b experiment4-taaha <repo-url>` (or `git pull` if the clone already exists).
3. **vast.ai (once):** run setup — install deps + download ViT weights + ensure Market-1501.
4. **vast.ai:** run the sequential orchestrator → it launches the 9 runs one-by-one, streaming
   each to its own log file for monitoring.
5. **Local (any time):** `tail -f` the run logs over SSH (or via Jupyter) to monitor progress.

### 4.2 First-time setup on the instance (one-time)

Uses the repo's existing `run.py --setup-only` bootstrap, which already:
- Installs `requirements-vast.txt` (timm, yacs, opencv, pyyaml, pytest, gdown — **no torch install**; the vast.ai PyTorch+CUDA template provides it),
- Downloads the ImageNet-pretrained ViT-Base weights into `.cache/torch/checkpoints/`,
- Ensures Market-1501 in `data/market1501/`.

```bash
cd TransReID-PEFT
git pull origin experiment4-taaha        # get latest commit
python run.py --setup-only --skip-smoke  # deps + weights + Market-1501 (one-time)
```

> `--skip-smoke` is optional; it skips `tools/check_peft.py`. The runner does its own sanity
> check per config (§3.8). Note `run.py` does **not** upload/need the `data/` dir or `.cache/` from
> your local machine — those are downloaded fresh on the instance (Market-1501 is ~153 MB).

### 4.3 Kick off the experiment (sequential)

```bash
nohup python tools/run_experiment4.py > logs/experiment4/driver.log 2>&1 &
echo $! > logs/experiment4/driver.pid   # to kill/restart if needed
```

or run it in a `tmux`/`screen` session so it survives SSH disconnect. The runner:

- runs the 9 configs **strictly one at a time** (single GPU — never parallel),
- appends `SOLVER.SEED <seed>` to each `train.py` invocation,
- streams each run's stdout/stderr to `logs/experiment4/run_XX_<method>_<window>.log`,
- writes `progress.json` after each run completes.

**Suggested first check (before the 9 long runs):**

```bash
python tools/run_experiment4.py --only-run 1 --epochs 2   # quick GPU sanity check
```

### 4.4 Monitoring (constant)

**Live tail of the current run (the file the runner is writing right now):**

```bash
tail -f logs/experiment4/run_01_bitfit_0_11.log
```

**Watch progress across all runs + the driver:**

```bash
tail -f logs/experiment4/driver.log        # runner itself: which run, status, errors
cat logs/experiment4/progress.json         # summary of finished runs
watch -n 60 tail -n 5 logs/experiment4/progress.json
```

**Overall view once running:**

```bash
ls -l logs/experiment4/                    # see which run logs exist = which runs started/finished
```

**Checking GPU is actually training:**

```bash
nvidia-smi                                  # GPU util / VRAM
```

### 4.5 Failure handling & resumability

- **A single run crashes** → runner records `returncode != 0` in `progress.json`, writes a
  `ERROR` line to `driver.log`, and **continues to the next run** (default) or stops
  (`--stop-on-error`).
- **Instance dies / SSH drops** → the runner was started with `nohup`/`tmux`, so it keeps running.
  If it does die, re-run the same command: `progress.json` is read at startup and **completed runs
  are skipped** (unless `--force`), so it resumes at the first unfinished config.
- **Logs & checkpoints are on the instance disk.** If the instance is destroyed, they're gone —
  pull `logs/experiment4/` and `logs/market_*/` (checkpoints) down periodically, or copy the
  final `progress.json` + `research.md` tables after all runs.

### 4.6 What to pull back to the repo afterward

- `logs/experiment4/progress.json` → the 9-run results table (param%, VRAM, wall time, mAP, R1).
- Per-run `.log` files → evidence for the appendix / reproducibility.
- Final checkpoints (`logs/market_vit_transreid_{method}_{window}/*.pth`) → optional, for
  `test.py` re-evaluation or if a run needs a seed extension.

### 4.7 External dependencies on the instance

- **Nothing extra to install** beyond `run.py --setup-only` (deps + weights + Market-1501).
- Market-1501 archive is downloaded by `run.py` (Google Drive with HTTP-mirror fallback).

## 5. What Is NOT in the Repo / Needs to Be Obtained Externally

### Datasets
- **Market-1501** — the raw `data/market1501/Market-1501-v15.09.15.zip` already exists in the
  repo, so **no external download needed** if you run on Market-1501. (You should — the 9 runs
  reuse the existing Market-1501 base config.)

### Pretrained weights
- **ImageNet-pretrained ViT-Base** (`jx_vit_base_p16_224-80ecf9dd.pth`). Referenced by
  `configs/base/*.yml` (`MODEL.PRETRAIN_PATH: '.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'`).
  If it's already downloaded from Experiments 1–3, **nothing new is needed**. If a fresh clone,
  `run.py` downloads it (or fetch from timm/release on HF Hub).
- These weights are **the only external dependency** for the main 9 runs.

### Third-party Python packages
- None new. The repo already depends on `torch`, `torchvision`, `timm`, `yacs`, `gdown`
  (`requirements.txt` / `requirements-vast.txt`). Bottleneck adapter + BitFit + LN-tuning use only
  `torch.nn`.

### Code / modules
- **No external PEFT library** needed — the repo has its own LoRA/SSF implementation and the new
  methods are a few lines each on top of that. Don't pull in `peft`/`adapters`; keep the local
  implementations consistent with the existing frozen-backbone pattern.
- The bottleneck-adapter module, BitFit/LN freeze sweeps, config schema, and runner are all to be
  written locally (see §3).

## 6. Gotchas / Decisions to Make Explicit

### 6.1 "Depth window" semantics for BitFit & LN-tuning
LoRA's `BLOCKS` filters which blocks get adapters; blocks outside stay 100% frozen. For BitFit and
LN-tuning, the analog is: **unfreeze biases/LN-params only in the listed blocks**. `0-11` = all
blocks. Blocks outside the window keep everything frozen. This keeps the 3 methods on the same
comparison grid as LoRA/SSF.

### 6.2 Bias LR interaction (BitFit) — **locked: SSF-consistent 10× LR**
`make_optimizer` already treats bias specially. For BitFit, biases are the *entire* trainable set.
**Implemented:** BitFit biases get `lr * 10, weight_decay = 0` (matching SSF), via an explicit
`elif peft_method == 'bitfit' and 'bias' in name` branch in `make_optimizer`. SSF keeps its own
`PEFT.SSF.LR` override path unchanged.

### 6.3 LayerNorm has `weight` + `bias` — both are trainable in LN-tuning
`nn.LayerNorm` params are named `weight` (γ) and `bias` (β). A naive "unfreeze biases only" sweep
(BitFit) would catch LN's `bias` too. Make the two sweeps mutually exclusive:
- BitFit unfreezes *linear* biases (`nn.Linear.bias`) + BNNeck bias (frozen by design) — but NOT
  LayerNorm params.
- LN-tuning unfreezes *only* LayerNorm `weight` + `bias`.
This keeps the two baselines cleanly separable (and matches the published definitions).

### 6.4 Adapter placement — **locked: parallel, both sublayers**
Adapters wrap each target `nn.Linear` **in parallel** (residual `base(x) + scale*up(act(down(x)))`),
matching LoRA's structure and targets (`qkv, proj, fc1, fc2`) for an apples-to-apples depth
comparison. Config: `PEFT.ADAPTER.TARGETS` makes placement switchable. Bottleneck `r=16`, GELU,
`down`=kaiming / `up`=zeros (identity at init).

### 6.5 Adapter initialization
`down` = kaiming (like `lora_A`), `up` = zeros (so the adapter is identity at init — same trick as
LoRA's zero-init `lora_B`). This keeps the first forward pass identical to the frozen backbone.

### 6.6 JPM is on in the base Market config
`configs/base/market1501_transreid.yml` has `JPM: True`, so there are 5 classifiers + 5 bottlenecks
(main + 4 local heads). The freeze sweeps must keep **all** classifier/BNNeck params trainable
(`classifier`, `classifier_1..4`, `bottleneck`, `bottleneck_1..4`), and the `b1`/`b2` JPM branches
are `deepcopy` of the last block + norm — make sure the block-scoping sweep handles the copied
branches (names are `b1.`, `b2.` — decide whether to freeze or train them consistently with the
rest of the window).

### 6.7 eval-period / checkpoint period
Base config sets `CHECKPOINT_PERIOD: 60` and `EVAL_PERIOD: 60` with `MAX_EPOCHS: 60` — i.e. **a
single final eval**. Keep it (matches existing runs) but note that intermediate evals need a
`--epochs`/period override for debugging.

### 6.8 Seeds
Run the 9 configs with **1 seed** (the base config's `SOLVER.SEED: 1234`). Only configs that land
on the non-dominated frontier get additional seeds (per the review ask). The runner should take a
`--seed` override so re-runs are clean.

## 7. Acceptance Criteria

- 9 new configs exist, all inheriting the same base recipe (60 epochs, AdamW, softmax+triplet).
- `make_model` accepts `PEFT.METHOD in ('lntune','bitfit','adapter')`, freezes backbone, keeps
  head + selected params trainable, and prints the trainable/total % line.
- `normalize_peft_config` enforces mutual exclusion across all 5 methods.
- `tools/run_experiment4.py` runs the 9 configs **sequentially as `train.py` subprocesses**, one
  at a time, with:
  - a per-run log file in `logs/experiment4/run_XX_<method>_<window>.log` (streamed live),
  - `logs/experiment4/progress.json` appended after each run (config, seed, returncode, param%,
    peak VRAM, wall time, mAP, R1),
  - resumability: completed runs are skipped on restart unless `--force`,
  - `--seed`, `--only METHOD[,WINDOW]`, `--only-run N`, `--epochs` overrides,
  - a `nohup`/`tmux`-friendly driver log for constant monitoring.
- Trainable-param percentages are sane: LN-tuning ≪ BitFit ≪ Adapter(r=16) ≪ LoRA, and all ≪ Full FT.
- Docs updated: new methods in Abstract/Contributions, Methodology §3, Experiments §4, Table 3,
  Limitations, plus this plan's §4 deployment/monitoring section verified against the instance.
