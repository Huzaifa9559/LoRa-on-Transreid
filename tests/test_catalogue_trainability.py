#!/usr/bin/env python3
"""Per-method trainability checks against catalogue YAMLs."""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from config import cfg as _GLOBAL_CFG
from config.loader import merge_config_file
from config.peft_config import get_peft_method, normalize_peft_config
from model import make_model
from model.peft.lightweight import BottleneckAdapter
from model.peft.lora import LoRALinear
from model.peft.ssf import SSF


def _cfg_from(path):
    cfg = _GLOBAL_CFG.clone()
    cfg.defrost()
    merge_config_file(cfg, path)
    cfg.MODEL.PRETRAIN_CHOICE = "none"
    cfg.MODEL.PRETRAIN_PATH = ""
    cfg.MODEL.DEVICE = "cpu"
    normalize_peft_config(cfg)
    cfg.freeze()
    return cfg


def _build(path):
    cfg = _cfg_from(path)
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    return cfg, model


def _bnneck_trainable(model):
    scales, biases = [], []
    for name, p in model.named_parameters():
        if "bottleneck" not in name.lower():
            continue
        if name.endswith(".weight"):
            scales.append((name, p.requires_grad))
        if name.endswith(".bias"):
            biases.append((name, p.requires_grad))
    return scales, biases


def test_lora_catalogue_trainability():
    cfg, model = _build("configs/catalogue/market1501/lora_0_11_r8_a16.yml")
    assert get_peft_method(cfg) == "lora"
    assert any(isinstance(m, LoRALinear) for m in model.modules())
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "classifier" in name:
            assert p.requires_grad, name
        parent = name.rsplit(".", 1)[0]
        mod = dict(model.named_modules()).get(parent)
        if isinstance(mod, nn.LayerNorm) and name.startswith("base."):
            assert p.requires_grad is False, name
    scales, biases = _bnneck_trainable(model)
    assert scales and all(not t for _, t in scales)
    assert biases and all(not t for _, t in biases)


def test_adapter_catalogue_trainability():
    cfg, model = _build("configs/catalogue/market1501/adapter_0_11_r16.yml")
    assert get_peft_method(cfg) == "adapter"
    assert any(isinstance(m, BottleneckAdapter) for m in model.modules())
    scales, biases = _bnneck_trainable(model)
    assert scales and all(not t for _, t in scales)
    assert biases and all(not t for _, t in biases)


def test_bitfit_catalogue_bnneck_biases():
    cfg, model = _build("configs/catalogue/market1501/bitfit_0_11.yml")
    assert get_peft_method(cfg) == "bitfit"
    scales, biases = _bnneck_trainable(model)
    assert scales and all(not t for _, t in scales)
    assert biases and all(t for _, t in biases)


def test_lntune_catalogue_bnneck_frozen():
    cfg, model = _build("configs/catalogue/market1501/lntune_0_11.yml")
    assert get_peft_method(cfg) == "lntune"
    scales, biases = _bnneck_trainable(model)
    assert scales and all(not t for _, t in scales)
    assert biases and all(not t for _, t in biases)


def test_ssf_catalogue_jpm_off_and_modules():
    cfg, model = _build("configs/catalogue/market1501/ssf_0_11_case2.yml")
    assert get_peft_method(cfg) == "ssf"
    assert cfg.MODEL.JPM is False
    assert any(isinstance(m, SSF) for m in model.modules())
    scales, biases = _bnneck_trainable(model)
    # Single BNNeck: scale trainable, bias frozen.
    assert any(t for _, t in scales)
    assert biases and all(not t for _, t in biases)
