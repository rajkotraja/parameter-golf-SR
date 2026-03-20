# Combined SOTA: 10L + FP16 Embed + Sliding Window + Muon WD + Overtone Init

## Summary

This submission combines and reproduces the top techniques from the leaderboard into a single training script, achieving **val_bpb = 1.1739** on 8xH100 SXM in under 10 minutes. This matches/slightly beats the current SOTA record of 1.1748 (notapplica) but is submitted as a non-record entry since we have only 1 seed run.

Additionally, we fixed PyTorch 2.4 compatibility issues (`enable_gqa` and `device_id` in `init_process_group`) that prevented the existing SOTA scripts from running on common RunPod images.

## Techniques Used

### From existing SOTA submissions (combined):
1. **FP16 Tied Embedding Export** (-0.007 BPB): Keep the dual-role embedding tensor in fp16 instead of int8 during quantization, reducing compound error through both input lookup and output projection paths.

2. **10 Transformer Layers** (-0.003 BPB): Increased from 9 to 10 layers. The compression savings from Muon weight decay make the extra layer fit within the 16MB budget.

3. **Overtone Spectral Embedding Init** (-0.002 BPB): After random initialization, reshape the embedding's singular value spectrum to follow a power-law decay (S_k ~ k^{-0.5}) via SVD, giving the embedding better starting structure.

4. **Phase-Transition Residual Mixing** (-0.002 BPB): Initialize resid_mix with a sigmoid schedule — early layers trust the original embedding (x0) more, late layers trust the accumulated residual more.

5. **Sliding Window Evaluation** (-0.030 BPB): Evaluate with overlapping windows at stride=64, so each token is scored with ~960 tokens of context instead of an average of ~512.

6. **Decoupled Muon Weight Decay** (-0.003 BPB): Apply L2 weight decay (0.02) to Muon-optimized matrix parameters after each optimizer step, regularizing weights and improving quantization robustness.

7. **AdamW for Embeddings/Scalars** (-0.001 BPB): Switched from Adam to AdamW (weight_decay=0.01) for embedding and scalar parameter optimizers.

### Compatibility fixes (our contribution):
8. **Manual GQA for PyTorch <2.5**: Replaced `enable_gqa=True` in `scaled_dot_product_attention()` with explicit K/V head repetition, making the script compatible with PyTorch 2.4.x which is common on RunPod images.

9. **Removed `device_id` from `init_process_group`**: This parameter was added in PyTorch 2.5 and causes crashes on 2.4.x. NCCL auto-detects the device from `torch.cuda.set_device()`.

## Results

| Seed | val_bpb | train_time | steps | artifact_size |
|------|---------|------------|-------|---------------|
| 1337 | **1.1739** | 600s | 12,244 | 15,337,480 bytes |

- Previous SOTA: 1.1748 (notapplica)
- Improvement: 0.0009 nats
- Note: Single seed only; additional seeds needed for statistical significance

## Artifact Size
- Model (int8+zlib): 15,281,645 bytes
- Code: 55,835 bytes
- **Total: 15,337,480 bytes** (< 16,000,000 limit)

## Training Details
- **Hardware**: 8x NVIDIA H100 80GB HBM3 SXM (NVLink)
- **Step speed**: ~49ms/step
- **Total steps**: 12,244 / 20,000 (wallclock capped at 600s)
- **Peak memory**: 11,896 MiB allocated

## Reproduction

```bash
# Setup
git clone https://github.com/rajkotraja/parameter-golf-SR.git
cd parameter-golf-SR
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

# If PyTorch < 2.5, apply compatibility fix:
sed -i 's/dist.init_process_group(backend="nccl", device_id=device)/dist.init_process_group(backend="nccl")/' my_train.py

# Run
SEED=1337 RUN_ID=final_submission_8gpu \
NUM_LAYERS=10 WARMDOWN_ITERS=2500 TIED_EMBED_LR=0.10 \
EVAL_STRIDE=64 EVAL_SEQ_LEN=1024 MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 my_train.py
```
