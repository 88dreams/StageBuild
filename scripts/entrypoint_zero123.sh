#!/usr/bin/env bash
set -e

# Ensure data mounts
mkdir -p /data/{input,output,checkpoints,.cache/huggingface}
ln -sfn /data/input /workspace/input
ln -sfn /data/output /workspace/output
ln -sfn /data/checkpoints /workspace/checkpoints
export HF_HOME=/data/.cache/huggingface

INPUT_IMG=${1:-/workspace/input/input.jpg}
OUT_DIR=${2:-/workspace/output/splat1/views}
NUM_VIEWS=${3:-12}
ELEV=${4:-15.0}

mkdir -p "$OUT_DIR"

# Ensure checkpoint exists
if [ ! -f "/workspace/checkpoints/zero123-xl.ckpt" ]; then
  echo "[Zero123] Missing checkpoint at /workspace/checkpoints/zero123-xl.ckpt"
  echo "[Zero123] Place it at /data/checkpoints/zero123-xl.ckpt and re-run."
  exit 2
fi

# Run Zero123-XL multiviews
python /workspace/scripts/generate_views_zero123.py \
  --input "$INPUT_IMG" \
  --output "$OUT_DIR" \
  --num_views "$NUM_VIEWS" \
  --elevation "$ELEV"

# Keep container alive if no further command provided
tail -f /dev/null
