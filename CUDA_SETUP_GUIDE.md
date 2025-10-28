# 🚀 Complete Guide: Running LoRA Training on CUDA

## ✅ Prerequisites Checklist

- [x] PyTorch with CUDA 11.8 support installed
- [x] NVIDIA RTX 4000 Ada GPU available  
- [x] CUDA Toolkit 12.8 installed
- [x] cuDNN installed
- [x] Virtual environment activated

## 🔧 How to Run on CUDA Only

### **Step 1: Verify CUDA Setup**

Before training, verify CUDA is working:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**Expected Output:**
```
CUDA: True
GPU: NVIDIA RTX 4000 Ada Generation
```

---

### **Step 2: Configuration Settings for CUDA**

Your config file already has the correct settings. Verify these are in your YAML:

```yaml
MODEL:
  DEVICE: 'cuda'          # ✅ Forces CUDA usage
  DEVICE_ID: '0'          # ✅ GPU device ID (0 for your single GPU)
```

---

### **Step 3: Start Training**

**Command to train with LoRA on blocks 6-11:**

```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml
```

---

### **Step 4: Verify GPU Usage During Training**

To monitor GPU usage in another terminal:

```bash
nvidia-smi -l 1
```

You should see memory being used by your GPU. Example output:
```
+-------------------------+------------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
|   0  NVIDIA RTX 4000... | On  | 00000000:01:00.0  On |             Off |
+-------------------------+------------------------+
| 20475MiB / 20475MiB |   X%      Default |
+-------------------------+------------------------+
```

---

## 🔍 Troubleshooting

### **Problem: Model running on CPU instead of GPU**

**Solution 1:** Check your config has DEVICE: 'cuda'
```bash
grep "DEVICE:" configs/Market/vit_transreid_stride_lora_blocks_6_11.yml
```

**Solution 2:** Force CUDA via command line override:
```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml \
  MODEL.DEVICE cuda
```

**Solution 3:** Verify PyTorch CUDA version:
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Should show: `2.7.1+cu118`

---

### **Problem: CUDA Out of Memory (OOM)**

**Solutions (in order of effectiveness):**

1. **Reduce batch size:**
```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml \
  SOLVER.IMS_PER_BATCH 32
```

2. **Reduce model size (use smaller LoRA rank):**
```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml \
  LORA.R 4
```

3. **Enable gradient accumulation:**
```bash
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml \
  SOLVER.ACCUMULATION_STEPS 2
```

---

## 📊 Expected Performance

With RTX 4000 Ada and your config:
- **Memory Used:** ~15-18 GB
- **Training Speed:** ~2-3 samples/sec
- **Trainable Parameters:** 3.47M (3.33%)
- **Total Parameters:** 104.2M

---

## 🎯 Alternative Configurations

### **For faster training (smaller model):**
Create `configs/Market/vit_transreid_stride_lora_small.yml`:
```yaml
MODEL:
  DEVICE: 'cuda'
  DEVICE_ID: '0'
  TRANSFORMER_TYPE: 'vit_small_patch16_224_TransReID'

SOLVER:
  IMS_PER_BATCH: 128  # Larger batch
  BASE_LR: 5.0e-4

LORA:
  ENABLED: True
  BLOCKS: [6, 7, 8, 9, 10, 11]
  R: 4  # Smaller rank
```

---

## 🚀 Quick Start Commands

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Verify CUDA
python -c "import torch; assert torch.cuda.is_available()"

# 3. Start training
python train.py --config_file configs/Market/vit_transreid_stride_lora_blocks_6_11.yml

# 4. Monitor GPU (in another terminal)
nvidia-smi -l 1
```

---

## 📝 Key Files Modified

1. **`config/defaults.py`** - Added LORA.BLOCKS config option
2. **`reid/peft/lora.py`** - Updated inject_lora_into_vit() to support include_blocks
3. **`model/make_model.py`** - Updated LoRA injection to use block-specific targeting
4. **`configs/Market/vit_transreid_stride_lora_blocks_6_11.yml`** - CUDA config with LoRA blocks 6-11

---

## ✅ Verification Checklist Before Training

- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] GPU detected: `nvidia-smi`
- [ ] PyTorch with CUDA: `python -c "import torch; print(torch.version.cuda)"`
- [ ] Config file exists: `ls configs/Market/vit_transreid_stride_lora_blocks_6_11.yml`
- [ ] Data directory exists: `ls data/market1501/`

---

## 💡 Tips for Optimal Performance

1. **Set number of workers to 0 if GPU memory is tight:**
```yaml
DATALOADER:
  NUM_WORKERS: 0  # Default is 8, use 0 to save memory
```

2. **Enable gradient checkpointing (if available) to save memory:**
```yaml
MODEL:
  USE_GRADIENT_CHECKPOINTING: True
```

3. **Use mixed precision training (already enabled in processor):**
- Automatically uses torch.cuda.amp.autocast()
- Reduces memory usage by ~40%

---

## 🎓 Understanding Device Resolution

The code automatically resolves which device to use in this order:

1. **Check `cfg.MODEL.DEVICE`** → If 'cuda' and CUDA available → Use GPU ✅
2. **Fallback to any available GPU** if config says 'cuda' but fails
3. **Fallback to CPU** as last resort

Since your config has `MODEL.DEVICE: 'cuda'` and CUDA is available, it will always use GPU ✅

---

**For questions or issues, refer to the main README or check nvidia-smi output.**
