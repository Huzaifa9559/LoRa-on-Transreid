#!/usr/bin/env python3

"""
Test script to verify LoRA is applied only to specified transformer blocks.
This script will help you verify that LoRA is correctly applied to layers 6-11.
"""

import torch
import sys
import os
sys.path.append('.')

from config import cfg
from model import make_model
from reid.peft.lora import LoRALinear

def test_lora_block_specific():
    """Test that LoRA is applied only to the specified blocks"""
    
    # Load a config with LoRA enabled for blocks 6-11
    cfg.merge_from_file('configs/Market/vit_transreid_stride_lora_blocks_6_11.yml')
    # Don't load pretrained weights for testing
    cfg.MODEL.PRETRAIN_CHOICE = 'none'
    cfg.MODEL.PRETRAIN_PATH = ''
    cfg.MODEL.DEVICE = 'cuda'  # Use CUDA
    cfg.LORA.ENABLED = True
    cfg.LORA.BLOCKS = [6, 7, 8, 9, 10, 11]  # Only apply to these blocks
    cfg.LORA.R = 8
    cfg.LORA.ALPHA = 16
    cfg.LORA.TARGETS = ["qkv", "proj", "fc1", "fc2"]
    
    print("Configuration:")
    print(f"  LoRA Enabled: {cfg.LORA.ENABLED}")
    print(f"  LoRA Blocks: {cfg.LORA.BLOCKS}")
    print(f"  LoRA Targets: {cfg.LORA.TARGETS}")
    print(f"  Device: {cfg.MODEL.DEVICE}")
    print()
    
    # Create the model
    model = make_model(cfg, num_class=751, camera_num=6, view_num=1)
    
    # Move model to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Model moved to: {device}")
    print()
    
    # Test with a dummy input to verify GPU usage
    dummy_input = torch.randn(1, 3, 256, 128).to(device)  # Batch size 1, 3 channels, 256x128 image
    print(f"Dummy input shape: {dummy_input.shape}, device: {dummy_input.device}")
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(dummy_input)
            print(f"✅ Forward pass successful!")
            print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'Multiple outputs'}")
            print(f"Output device: {output.device if hasattr(output, 'device') else 'Multiple tensors'}")
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
    print()
    
    print("Checking LoRA application...")
    
    # Check which modules have LoRA applied
    lora_modules = []
    all_target_modules = []
    
    for name, module in model.named_modules():
        # Check for target modules in transformer blocks
        if any(target in name for target in cfg.LORA.TARGETS):
            all_target_modules.append(name)
            
            if isinstance(module, LoRALinear):
                lora_modules.append(name)
                # Extract block number
                if "blocks." in name:
                    try:
                        block_num = int(name.split("blocks.")[1].split(".")[0])
                        print(f"  ✓ LoRA applied to: {name} (Block {block_num})")
                    except:
                        print(f"  ✓ LoRA applied to: {name}")
    
    print(f"\nSummary:")
    print(f"  Total target modules found: {len(all_target_modules)}")
    print(f"  Modules with LoRA applied: {len(lora_modules)}")
    
    # Verify only blocks 6-11 have LoRA
    expected_blocks = set(cfg.LORA.BLOCKS)
    actual_blocks = set()
    
    for name in lora_modules:
        if "blocks." in name:
            try:
                block_num = int(name.split("blocks.")[1].split(".")[0])
                actual_blocks.add(block_num)
            except:
                pass
    
    print(f"  Expected blocks with LoRA: {sorted(expected_blocks)}")
    print(f"  Actual blocks with LoRA: {sorted(actual_blocks)}")
    
    if actual_blocks == expected_blocks:
        print("  ✅ SUCCESS: LoRA correctly applied only to specified blocks!")
    else:
        print("  ❌ ERROR: LoRA not applied to correct blocks!")
        print(f"     Missing: {expected_blocks - actual_blocks}")
        print(f"     Extra: {actual_blocks - expected_blocks}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nParameter counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")

if __name__ == "__main__":
    test_lora_block_specific()
