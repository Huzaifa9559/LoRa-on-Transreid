#!/usr/bin/env python3
"""LoRA merge safety and adapter-checkpoint load tests."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from model.peft.lora import (
    LoRALinear,
    build_adapter_checkpoint,
    export_merged_lora_copy,
    inject_lora_into_vit,
    load_lora_state_dict,
    load_prediction_state,
    mark_trainable_lora_and_head,
    maybe_merge_lora,
)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(16, 16)})}),
                nn.ModuleDict({"attn": nn.ModuleDict({"qkv": nn.Linear(16, 16)})}),
            ]
        )
        self.classifier = nn.Linear(16, 5, bias=False)
        self.bottleneck = nn.BatchNorm1d(16)
        self.bottleneck.bias.requires_grad_(False)

    def forward(self, x):
        for blk in self.blocks:
            x = blk["attn"]["qkv"](x)
        return self.classifier(self.bottleneck(x))


def _build():
    model = _Tiny()
    inject_lora_into_vit(
        model, r=4, alpha=8, dropout=0.0, targets=["qkv"], include_blocks=[0, 1]
    )
    mark_trainable_lora_and_head(model, train_head=True)
    return model


def test_train_eval_train_keeps_adapters():
    model = _build()
    maybe_merge_lora(model, enabled=True, merge_at_eval=True)

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.normal_()
                m.lora_B.normal_()

    x = torch.randn(2, 16)
    layer = model.blocks[0]["attn"]["qkv"]
    captured = {}

    def _hook(_m, _inp, out):
        captured["y"] = out.detach().clone()

    handle = layer.register_forward_hook(_hook)

    model.train()
    model(x)
    y_train1 = captured["y"]

    model.eval()
    model(x)
    y_eval = captured["y"]
    # Reversible merge must keep the LoRALinear output identical (BN may differ).
    assert torch.allclose(y_train1, y_eval, atol=1e-5)

    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert torch.any(m.lora_A != 0)
            assert torch.any(m.lora_B != 0)
            assert m._merged is True  # eval path merged without zeroing

    model.train()
    model(x)
    y_train2 = captured["y"]
    handle.remove()
    assert torch.allclose(y_train1, y_train2, atol=1e-5)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            assert m._merged is False  # training forward unmerges


def test_export_merged_copy_is_destructive_on_copy_only():
    model = _build()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.normal_()
                m.lora_B.normal_()

    before = {
        n: (m.lora_A.detach().clone(), m.lora_B.detach().clone())
        for n, m in model.named_modules()
        if isinstance(m, LoRALinear)
    }
    cloned = export_merged_lora_copy(model)
    for n, m in model.named_modules():
        if isinstance(m, LoRALinear):
            assert torch.equal(m.lora_A, before[n][0])
            assert torch.equal(m.lora_B, before[n][1])
            assert m._merged is False
    for m in cloned.modules():
        if isinstance(m, LoRALinear):
            assert torch.all(m.lora_A == 0)
            assert torch.all(m.lora_B == 0)
            assert m._merged is True


def test_adapter_checkpoint_roundtrip_copy_into_params():
    model = _build()
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.normal_()
                m.lora_B.normal_()
        model.classifier.weight.normal_()

    class _Cfg:
        class PEFT:
            class LORA:
                R = 4
                ALPHA = 8.0
                TARGETS = ["qkv"]
                BLOCKS = [0, 1]
                BIAS = "none"
                DROPOUT = 0.0

        class MODEL:
            TRANSFORMER_TYPE = "vit_base_patch16_224_TransReID"
            PRETRAIN_PATH = "dummy.pth"
            JPM = True

    payload = build_adapter_checkpoint(model, _Cfg())
    assert payload["meta"]["rank"] == 4
    assert "adapters" in payload and "prediction_state" in payload

    other = _build()
    ids_before = {
        n: id(m.lora_A)
        for n, m in other.named_modules()
        if isinstance(m, LoRALinear)
    }
    missing, unexpected = load_lora_state_dict(other, payload["adapters"], strict=True)
    assert missing == []
    assert unexpected == []
    load_prediction_state(other, payload["prediction_state"], strict=False)

    for n, m in other.named_modules():
        if isinstance(m, LoRALinear):
            assert id(m.lora_A) == ids_before[n]
            src = payload["adapters"][f"{n}.lora_A"]
            assert torch.allclose(m.lora_A.cpu(), src.cpu())

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "adapter.pth")
        torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert loaded["meta"]["method"] == "lora"


def test_bnneck_affine_frozen_under_lora():
    model = _build()
    for name, p in model.named_parameters():
        if "bottleneck" in name:
            assert p.requires_grad is False, name
        if "classifier" in name:
            assert p.requires_grad is True, name
