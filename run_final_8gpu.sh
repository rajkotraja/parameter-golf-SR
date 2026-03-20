#!/bin/bash
# ============================================================
# FINAL SUBMISSION RUN — 8x H100 SXM (~$4 per run, ~15 min)
# ============================================================
# This is the real deal. Run this once on 8xH100 to get your
# actual score. Expected: ~1.1748 BPB (matches SOTA leaderboard).
#
# SETUP (run once on your 8xH100 RunPod pod):
#   cd /workspace
#   git clone https://github.com/openai/parameter-golf.git
#   cd parameter-golf
#   python3 data/cached_challenge_fineweb.py --variant sp1024
#   # copy my_train.py here
#
# IMPORTANT: STOP YOUR POD IMMEDIATELY AFTER THIS FINISHES!
# 8xH100 costs ~$24/hr. This run takes ~15 min = ~$6.
# ============================================================

set -e

echo "Starting final 8xH100 submission run..."
echo "Expected time: ~10 min training + ~3 min sliding window eval"
echo ""

RUN_ID=final_submission_8gpu \
SEED=1337 \
NUM_LAYERS=10 \
WARMDOWN_ITERS=2500 \
TIED_EMBED_LR=0.10 \
EVAL_STRIDE=64 \
EVAL_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=200 \
torchrun --standalone --nproc_per_node=8 my_train.py

echo ""
echo "============================================================"
echo "  FINAL RUN COMPLETE!"
echo "  Check logs/final_submission_8gpu.txt for your val_bpb score"
echo "  Look for the 'final_int8_zlib_roundtrip' line"
echo "  Expected: val_bpb ~1.1748"
echo ""
echo "  >>> STOP YOUR POD NOW TO SAVE MONEY! <<<"
echo "============================================================"
