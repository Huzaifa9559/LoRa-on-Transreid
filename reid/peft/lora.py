"""Backward-compatible re-export of LoRA helpers.

Implementation lives in ``model.peft.lora``; keep this module so older
imports (``from reid.peft.lora import ...``) continue to work.
"""

from model.peft.lora import (  # noqa: F401
    LoRALinear,
    build_adapter_checkpoint,
    export_merged_lora_copy,
    get_module_by_name,
    inject_lora_into_vit,
    iter_linear_targets,
    load_lora_state_dict,
    load_prediction_state,
    lora_state_dict,
    mark_trainable_lora_and_head,
    maybe_merge_lora,
    prediction_state_dict,
    set_module_by_name,
)

__all__ = [
    "LoRALinear",
    "build_adapter_checkpoint",
    "export_merged_lora_copy",
    "get_module_by_name",
    "inject_lora_into_vit",
    "iter_linear_targets",
    "load_lora_state_dict",
    "load_prediction_state",
    "lora_state_dict",
    "mark_trainable_lora_and_head",
    "maybe_merge_lora",
    "prediction_state_dict",
    "set_module_by_name",
]
