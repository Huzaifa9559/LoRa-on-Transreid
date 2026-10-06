"""PEFT config normalization and helpers."""

from typing import List, Optional, Set, Tuple


def _copy_legacy_lora(cfg):
    if not hasattr(cfg, "LORA") or not cfg.LORA.ENABLED:
        return
    for key in (
        "R", "ALPHA", "DROPOUT", "TARGETS", "TRAIN_HEAD",
        "MERGE_AT_EVAL", "BIAS", "SAVE_ADAPTER_ONLY", "BLOCKS",
    ):
        if hasattr(cfg.LORA, key):
            setattr(cfg.PEFT.LORA, key, getattr(cfg.LORA, key))
    if cfg.PEFT.METHOD == "none":
        cfg.PEFT.METHOD = "lora"


# New lightweight methods resolved from their ENABLED flags, like SSF.
_LIGHTWEIGHT_FLAGS = (
    ("lntune", "LNTUNE"),
    ("bitfit", "BITFIT"),
    ("adapter", "ADAPTER"),
)


def _reset_peft_flags(cfg):
    """Clear all PEFT ENABLED flags except the one matching PEFT.METHOD.

    merge_config_file merges YAML dicts onto a shared global config, so flags
    left set by a previous config (e.g. a prior test) would otherwise be read as
    "still enabled". PEFT.METHOD is authoritative — this keeps the flags
    consistent with it and prevents stale flags from triggering false
    mutual-exclusion errors.
    """
    cfg.PEFT.SSF.ENABLED = (cfg.PEFT.METHOD == "ssf")
    for _, node in _LIGHTWEIGHT_FLAGS:
        getattr(cfg.PEFT, node).ENABLED = False


def normalize_peft_config(cfg):
    """Resolve legacy flags, enforce one PEFT method, and make flags consistent.

    PEFT.METHOD is authoritative. When METHOD == 'none' but exactly one method's
    ENABLED flag is set, resolve that flag into METHOD (legacy-compat). Flags for
    non-selected methods are cleared so a single shared global config never
    accumulates stale enabled flags across runs.
    """
    if cfg.is_frozen():
        cfg.defrost()

    _copy_legacy_lora(cfg)

    # Capture the flags this config actually set before clearing stale ones.
    flag_method = None
    if cfg.PEFT.SSF.ENABLED:
        flag_method = "ssf"
    for method, node in _LIGHTWEIGHT_FLAGS:
        if getattr(cfg.PEFT, node).ENABLED:
            if flag_method is not None:
                raise ValueError(
                    f"Cannot enable more than one PEFT method at a time "
                    f"(flags: {flag_method}, {method}). Set PEFT.METHOD to one method only."
                )
            flag_method = method

    if cfg.PEFT.METHOD == "none":
        if flag_method is not None:
            cfg.PEFT.METHOD = flag_method
    # When METHOD is already set, it is authoritative. Stale ENABLED flags left
    # on the shared global CfgNode by a previous merge are cleared below; they
    # must not raise, because yacs merge does not unset omitted keys.

    # Canonicalize: PEFT.METHOD is the single source of truth.
    method = cfg.PEFT.METHOD
    if method not in ("none", "lora", "ssf", "lntune", "bitfit", "adapter"):
        method = "none"
    cfg.PEFT.METHOD = method
    _reset_peft_flags(cfg)

    if method == "ssf" and cfg.PEFT.SSF.OPTIMIZER_CASE == 2:
        cfg.SOLVER.BASE_LR = 3.5e-4
        cfg.SOLVER.WEIGHT_DECAY = 1e-4
        cfg.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
        cfg.SOLVER.BIAS_LR_FACTOR = 2


def get_peft_method(cfg) -> str:
    return str(cfg.PEFT.METHOD).lower()


def get_lora_blocks(cfg) -> Optional[List[int]]:
    blocks = list(cfg.PEFT.LORA.BLOCKS)
    return blocks if blocks else None


def get_ssf_blocks(cfg) -> Tuple:
    blocks = cfg.PEFT.SSF.BLOCKS
    return tuple(blocks) if blocks else ()


def get_active_blocks(cfg) -> Optional[Set[int]]:
    method = get_peft_method(cfg)
    if method == "lora":
        blocks = get_lora_blocks(cfg)
        return set(blocks) if blocks else None
    if method == "ssf":
        blocks = get_ssf_blocks(cfg)
        return set(blocks) if blocks else None
    if method == "lntune":
        blocks = list(cfg.PEFT.LNTUNE.BLOCKS)
        return set(blocks) if blocks else None
    if method == "bitfit":
        blocks = list(cfg.PEFT.BITFIT.BLOCKS)
        return set(blocks) if blocks else None
    if method == "adapter":
        blocks = list(cfg.PEFT.ADAPTER.BLOCKS)
        return set(blocks) if blocks else None
    return None
