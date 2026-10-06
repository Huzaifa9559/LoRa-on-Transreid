# model/peft/lora.py
import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _identity_or_dropout(p: float):
    return nn.Identity() if p <= 0 else nn.Dropout(p)


class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear as:
      y = base(x) + scaling * (B(A(dropout(x)))) + optional lora_bias
    - base params are frozen
    - only A, B (and optional lora_bias) are trainable
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        bias_mode: str = "none",  # "none" | "lora" | "all" (keep "none" by default)
    ):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(self.r, 1)
        self.drop = _identity_or_dropout(dropout)
        self.bias_mode = bias_mode
        self._merged = False

        if self.r > 0:
            # A: (r, in), B: (out, r)
            self.lora_A = nn.Parameter(torch.empty(self.r, self.in_features))
            self.lora_B = nn.Parameter(torch.empty(self.out_features, self.r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if self.base.bias is not None and self.bias_mode in ("lora", "all"):
            self.lora_bias = nn.Parameter(torch.zeros_like(self.base.bias))
        else:
            self.register_parameter("lora_bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # When merged into base, adapters must not be applied again.
        if self.r > 0 and not self._merged:
            x_d = self.drop(x)
            update = F.linear(x_d, self.lora_A)          # (B, r)
            update = F.linear(update, self.lora_B)       # (B, out)
            y = y + self.scaling * update
        if self.lora_bias is not None and not self._merged:
            y = y + self.lora_bias
        return y

    @property
    def adapter_state(self):
        s = {}
        if self.lora_A is not None:
            s["lora_A"] = self.lora_A
        if self.lora_B is not None:
            s["lora_B"] = self.lora_B
        if self.lora_bias is not None:
            s["lora_bias"] = self.lora_bias
        return s

    def merge_into_base_(self, destructive: bool = False):
        """
        Bake LoRA weights into base.weight (and bias if present).

        Reversible by default (adapters kept; ``_merged`` skips the LoRA path).
        Pass ``destructive=True`` only for one-way export that zeros adapters.
        """
        if self.r == 0 or self._merged:
            return
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)  # (out, in)
            self.base.weight += delta
            if self.lora_bias is not None and self.base.bias is not None:
                self.base.bias += self.lora_bias
            if destructive:
                nn.init.zeros_(self.lora_B)
                nn.init.zeros_(self.lora_A)
                if self.lora_bias is not None:
                    nn.init.zeros_(self.lora_bias)
            self._merged = True

    def unmerge_from_base_(self):
        """Undo a non-destructive merge so training can resume with adapters."""
        if self.r == 0 or not self._merged:
            return
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            self.base.weight -= delta
            if self.lora_bias is not None and self.base.bias is not None:
                self.base.bias -= self.lora_bias
            self._merged = False


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Replace a nested submodule given its dotted path name."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    parts = name.split(".")
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


def iter_linear_targets(model: nn.Module, targets: List[str]) -> Iterable[Tuple[str, nn.Linear]]:
    """
    Yield (qualified_name, linear_module) for layers whose qualified name contains
    any of the target keys. Typical ViT/timm names include:
      - attention.qkv (Linear)
      - attention.proj (Linear)
      - mlp.fc1, mlp.fc2 (Linear)
    Adjust `targets` if your repo uses different names.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in targets):
                yield name, module


def inject_lora_into_vit(
    model: nn.Module,
    r: int,
    alpha: float,
    dropout: float,
    targets: List[str],
    bias_mode: str = "none",
    include_blocks: Optional[List[int]] = None,
):
    """
    Wrap matching nn.Linear modules with LoRALinear and return list of replaced names.

    Args:
        model: The model to inject LoRA into
        r: LoRA rank
        alpha: LoRA alpha scaling
        dropout: Dropout rate for LoRA
        targets: List of target module names (e.g., ["qkv", "proj", "fc1", "fc2"])
        bias_mode: Bias handling mode
        include_blocks: Optional list of block indices to apply LoRA to (e.g., [6, 7, 8, 9, 10, 11]).
                       If None, applies to all blocks.
    """
    replaced = []
    for name, lin in list(iter_linear_targets(model, targets)):
        # Check if this layer is in a specified block (if include_blocks is provided)
        if include_blocks is not None and len(include_blocks) > 0:
            # Extract block number from the module name (e.g., "blocks.6.attn.qkv" -> 6)
            if "blocks." in name:
                try:
                    block_idx = int(name.split("blocks.")[1].split(".")[0])
                    if block_idx not in include_blocks:
                        # Skip this layer as it's not in the specified blocks
                        continue
                except (IndexError, ValueError):
                    # If we can't parse the block index, apply LoRA anyway
                    pass

        wrapped = LoRALinear(lin, r=r, alpha=alpha, dropout=dropout, bias_mode=bias_mode)
        set_module_by_name(model, name, wrapped)
        replaced.append(name)
    return replaced


def mark_trainable_lora_and_head(model: nn.Module, train_head: bool = True):
    """
    Freeze everything, then unfreeze:
      - LoRA adapter params (A/B/bias),
      - classifier / ID head params if train_head=True.

    Backbone LayerNorm stays frozen. BNNeck modules are named ``bottleneck*``
    and are not matched by the head keywords below, so their affine parameters
    remain frozen (running statistics still update in train mode).
    """
    for p in model.parameters():
        p.requires_grad = False

    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            # Only unfreeze LoRA adapter parameters, not base parameters
            if m.lora_A is not None:
                m.lora_A.requires_grad = True
            if m.lora_B is not None:
                m.lora_B.requires_grad = True
            if m.lora_bias is not None:
                m.lora_bias.requires_grad = True

    if train_head:
        for name, module in model.named_modules():
            lname = name.lower()
            # Match classifiers only; do not unfreeze BNNeck affine params.
            if any(k in lname for k in ["classifier", "id_head"]):
                for p in module.parameters():
                    p.requires_grad = True


def lora_state_dict(model: nn.Module) -> dict:
    """
    Return only LoRA adapter tensors in a flat dict keyed by qualified module name.
    Example keys:
      "...attn.qkv.lora_A", "...attn.qkv.lora_B", "...attn.qkv.lora_bias"
    """
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
                sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
            if module.lora_bias is not None:
                sd[f"{name}.lora_bias"] = module.lora_bias.detach().cpu()
    return sd


def prediction_state_dict(model: nn.Module) -> dict:
    """
    Extra tensors needed to reproduce predictions with a frozen backbone:
    classifiers, BNNeck affine + buffers, and SIE camera/view embeddings.
    """
    sd = {}
    for name, tensor in model.state_dict().items():
        lname = name.lower()
        keep = (
            "classifier" in lname
            or "bottleneck" in lname
            or "sie_embed" in lname
            or "cam_embed" in lname
            or name.endswith("camera_emb")
            or name.endswith("view_emb")
            or "sie" in lname and ("embed" in lname or "emb" in lname)
        )
        if keep:
            sd[name] = tensor.detach().cpu()
    return sd


def build_adapter_checkpoint(model: nn.Module, cfg=None) -> Dict[str, Any]:
    """Adapter export (not a full training-resume checkpoint)."""
    meta = {
        "format": "peft_adapter_v1",
        "method": "lora",
    }
    if cfg is not None and hasattr(cfg, "PEFT"):
        lora_cfg = cfg.PEFT.LORA
        meta.update(
            {
                "rank": int(lora_cfg.R),
                "alpha": float(lora_cfg.ALPHA),
                "targets": list(lora_cfg.TARGETS),
                "blocks": list(lora_cfg.BLOCKS) if lora_cfg.BLOCKS else [],
                "bias": str(lora_cfg.BIAS),
                "dropout": float(lora_cfg.DROPOUT),
                "transformer_type": str(cfg.MODEL.TRANSFORMER_TYPE),
                "pretrain_path": str(cfg.MODEL.PRETRAIN_PATH),
                "jpm": bool(cfg.MODEL.JPM),
            }
        )
    return {
        "adapters": lora_state_dict(model),
        "prediction_state": prediction_state_dict(model),
        "meta": meta,
    }


def load_lora_state_dict(model: nn.Module, adapter_sd: dict, strict: bool = False):
    """
    Load LoRA adapter tensors into matching modules via ``copy_``.
    Validates shapes; does not replace Parameter objects.
    """
    missing = []
    unexpected = []
    model_keys = set()
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if module.lora_A is not None:
                model_keys.add(f"{name}.lora_A")
                model_keys.add(f"{name}.lora_B")
            if module.lora_bias is not None:
                model_keys.add(f"{name}.lora_bias")

    for k in adapter_sd.keys():
        if k not in model_keys:
            unexpected.append(k)

    for k in model_keys:
        if k not in adapter_sd:
            missing.append(k)
            continue
        try:
            mod_name, tensor_name = k.rsplit(".", 1)
            mod = get_module_by_name(model, mod_name)
            src = adapter_sd[k]
            if isinstance(src, nn.Parameter):
                src = src.data
            dst = getattr(mod, tensor_name)
            if tuple(dst.shape) != tuple(src.shape):
                raise RuntimeError(
                    f"Shape mismatch for {k}: model {tuple(dst.shape)} vs checkpoint {tuple(src.shape)}"
                )
            with torch.no_grad():
                dst.copy_(src.to(device=dst.device, dtype=dst.dtype))
        except Exception as exc:
            missing.append(f"{k} ({exc})")

    if unexpected:
        print(f"Unexpected LoRA keys ignored: {unexpected[:5]}"
              + ("..." if len(unexpected) > 5 else ""))
    if strict and missing:
        raise RuntimeError(f"Missing LoRA keys: {missing}")
    return missing, unexpected


def load_prediction_state(model: nn.Module, pred_sd: dict, strict: bool = False):
    """Copy classifiers / BNNeck / SIE tensors into the model."""
    missing = []
    own = model.state_dict()
    for k, v in pred_sd.items():
        if k not in own:
            missing.append(k)
            continue
        if tuple(own[k].shape) != tuple(v.shape):
            raise RuntimeError(
                f"Shape mismatch for prediction state {k}: "
                f"model {tuple(own[k].shape)} vs checkpoint {tuple(v.shape)}"
            )
        with torch.no_grad():
            own[k].copy_(v.to(device=own[k].device, dtype=own[k].dtype))
    if strict and missing:
        raise RuntimeError(f"Missing prediction-state keys: {missing}")
    return missing


def export_merged_lora_copy(model: nn.Module) -> nn.Module:
    """Deep-copy the model and destructively merge LoRA for inference export."""
    cloned = copy.deepcopy(model)
    for _, module in cloned.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_into_base_(destructive=True)
    return cloned


def maybe_merge_lora(model: nn.Module, enabled: bool, merge_at_eval: bool):
    """
    Optionally register a reversible eval-time merge.

    Automatic merging is disabled while ``model.training`` is True so a train
    forward cannot zero adapters. Published configs keep ``MERGE_AT_EVAL: False``.
    Prefer ``export_merged_lora_copy`` for a one-way inference export.
    """
    if not enabled or not merge_at_eval:
        return

    def merge_hook(m, *args, **kwargs):
        if m.training:
            # Ensure any previous eval merge is reversed before training steps.
            for _, module in m.named_modules():
                if isinstance(module, LoRALinear) and module._merged:
                    module.unmerge_from_base_()
            return
        for _, module in m.named_modules():
            if isinstance(module, LoRALinear) and not module._merged:
                module.merge_into_base_(destructive=False)

    model.register_forward_pre_hook(merge_hook, with_kwargs=False)
