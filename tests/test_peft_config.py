"""Additional tests for unified PEFT config handling."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import cfg, normalize_peft_config


def test_method_authoritative_clears_stale_flags():
    """METHOD wins over ENABLED flags left by a previous yacs merge."""
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "lora"
    cfg_copy.PEFT.SSF.ENABLED = True
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "lora"
    assert cfg_copy.PEFT.SSF.ENABLED is False


def test_legacy_lora_enabled_maps_to_method():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.LORA.ENABLED = True
    cfg_copy.LORA.R = 16
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "lora"
    assert cfg_copy.PEFT.LORA.R == 16


def test_ssf_case2_optimizer_overrides():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "ssf"
    cfg_copy.PEFT.SSF.OPTIMIZER_CASE = 2
    normalize_peft_config(cfg_copy)
    assert cfg_copy.SOLVER.BASE_LR == pytest.approx(3.5e-4)
    assert cfg_copy.SOLVER.WEIGHT_DECAY == pytest.approx(1e-4)
    assert cfg_copy.SOLVER.BIAS_LR_FACTOR == 2


def test_lightweight_enabled_resolves_method():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.PEFT.BITFIT.ENABLED = True
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "bitfit"

    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.PEFT.LNTUNE.ENABLED = True
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "lntune"

    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.PEFT.ADAPTER.ENABLED = True
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "adapter"


def test_lightweight_mutual_exclusion():
    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "none"
    cfg_copy.PEFT.BITFIT.ENABLED = True
    cfg_copy.PEFT.LNTUNE.ENABLED = True
    with pytest.raises(ValueError, match="more than one PEFT method"):
        normalize_peft_config(cfg_copy)

    cfg_copy = cfg.clone()
    cfg_copy.defrost()
    cfg_copy.PEFT.METHOD = "bitfit"
    cfg_copy.PEFT.SSF.ENABLED = True
    normalize_peft_config(cfg_copy)
    assert cfg_copy.PEFT.METHOD == "bitfit"
    assert cfg_copy.PEFT.SSF.ENABLED is False
