# What Changed: Baseline vs SOTA (my_train.py)

## All 7 changes that take you from 1.2244 to 1.1748 BPB

---

### 1. FP16 Tied Embedding Export (-0.007 BPB)
**File:** `quantize_state_dict_int8()` function
**What:** The embedding tensor serves dual duty (input lookup + output projection).
Int8 quantization errors compound through BOTH paths. Keeping it in fp16 costs more
bytes but dramatically reduces quantization error.

```python
# ADDED: before the standard small-tensor check
if "tok_emb" in name:
    kept = t.to(dtype=torch.float16).contiguous()
    passthrough[name] = kept
    passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
    stats["int8_payload_bytes"] += tensor_nbytes(kept)
    continue
```

### 2. 10 Transformer Layers (-0.003 BPB)
**What:** Changed default from 9 to 10 layers. The compression savings from Muon WD
make the extra layer fit within 16MB.

```python
# CHANGED: in Hyperparameters class
num_layers = int(os.environ.get("NUM_LAYERS", 10))  # was 9
```

### 3. Overtone Spectral Embedding Init (-0.002 BPB)
**What:** After random init, reshape the embedding's singular value spectrum to follow
a power-law decay (S_k ~ k^{-0.5}), like harmonic overtones. This gives the embedding
a better starting structure.

```python
# ADDED: in GPT._init_weights(), after normal init
with torch.no_grad():
    U, S, V = torch.linalg.svd(self.tok_emb.weight.data, full_matrices=False)
    target_S = S[0] * (1.0 / torch.arange(1, S.shape[0] + 1, dtype=S.dtype)) ** 0.5
    self.tok_emb.weight.data = (U * target_S[None, :]) @ V
```

### 4. Phase-Transition Residual Mixing (-0.002 BPB)
**What:** Initialize resid_mix with a sigmoid schedule: early layers trust the original
embedding (x0) more, late layers trust the accumulated residual more.

```python
# ADDED: in GPT._init_weights()
num_layers = len(self.blocks)
for i, block in enumerate(self.blocks):
    with torch.no_grad():
        phase = torch.sigmoid(torch.tensor(3.0 * (i / max(num_layers - 1, 1) - 0.5)))
        block.resid_mix.data[0] = phase * torch.ones(block.resid_mix.shape[1])
        block.resid_mix.data[1] = (1 - phase) * torch.ones(block.resid_mix.shape[1])
```

### 5. Sliding Window Evaluation (-0.030 BPB) <<<< BIGGEST WIN
**What:** Instead of evaluating each 1024-token chunk independently (where the first
token has 0 context, the 512th has 511, etc.), use overlapping windows with stride=64.
Every token gets scored with ~960 tokens of context.

```python
# ADDED: new eval_val_sliding() function + forward_logits() method on GPT
# ADDED: new hyperparameters eval_seq_len and eval_stride
eval_seq_len = int(os.environ.get("EVAL_SEQ_LEN", 0))   # 0 = same as train
eval_stride = int(os.environ.get("EVAL_STRIDE", 0))      # 0 = no sliding window
```

### 6. Decoupled Weight Decay for Muon (-0.003 BPB)
**What:** Apply L2 weight decay (0.02) to Muon-optimized matrix params AFTER the Muon
step. This regularizes weights and makes them more quantization-friendly.

```python
# ADDED: after optimizer steps in training loop
with torch.no_grad():
    for p in matrix_params:
        p.mul_(1.0 - 0.02 * optimizer_muon.param_groups[0]["lr"])
```

### 7. AdamW for Token/Scalar Params (-0.001 BPB)
**What:** Switched from Adam to AdamW (with weight_decay=0.01) for the embedding and
scalar parameter optimizers. Mild regularization helps generalization.

```python
# CHANGED: Adam -> AdamW with weight_decay=0.01
optimizer_tok = torch.optim.AdamW(..., weight_decay=0.01)
optimizer_scalar = torch.optim.AdamW(..., weight_decay=0.01)
```

### Also: NTK-Aware RoPE Scaling
**What:** When evaluating at sequence lengths longer than training, dynamically adjust
the RoPE base frequency for better extrapolation. Not directly used in the default
config but enables future eval_seq_len > train_seq_len experiments.

### Also: Simplified Forward Pass (speed improvement)
**What:** Removed LoRA TTT support from the forward path. This means torch.compile
can produce a cleaner, faster graph. The SOTA doesn't use LoRA TTT.

---

## Summary of hyperparameter changes

| Parameter | Baseline | SOTA |
|-----------|----------|------|
| NUM_LAYERS | 9 | **10** |
| WARMDOWN_ITERS | 1200 | **2500** |
| TIED_EMBED_LR | 0.05 | **0.10** |
| EVAL_STRIDE | (none) | **64** |
| Optimizer | Adam | **AdamW** (WD=0.01) |
| Muon WD | none | **0.02** |
| Embedding quant | int8 | **fp16** |
| Embed init | random | **Overtone SVD** |
| resid_mix init | [1,0] | **Sigmoid schedule** |
