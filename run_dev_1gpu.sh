#!/bin/bash
# ============================================================
# DEVELOPMENT RUN — 1x H100 (~$0.60 per run, ~12 min)
# ============================================================
# This runs the SOTA config on a single GPU for testing.
# Score won't match 8xH100 (fewer steps in 10 min) but lets
# you iterate quickly and verify everything works.
#
# SETUP (run once on your RunPod pod):
#   cd /workspace
#   git clone https://github.com/openai/parameter-golf.git
#   cd parameter-golf
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
#
# Then copy my_train.py to the pod and run this script.
# ============================================================

set -e

RUN_ID=dev_sota_1gpu \
SEED=1337 \
NUM_LAYERS=10 \
WARMDOWN_ITERS=2500 \
TIED_EMBED_LR=0.10 \
EVAL_STRIDE=64 \
EVAL_SEQ_LEN=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=500 \
TRAIN_LOG_EVERY=100 \
torchrun --standalone --nproc_per_node=1 my_train.py

echo ""
echo "============================================"
echo "  DEV RUN COMPLETE"
echo "  Check logs/dev_sota_1gpu.txt for results"
echo "============================================"
