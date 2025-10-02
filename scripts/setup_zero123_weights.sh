#!/usr/bin/env bash
# Download Zero123-XL checkpoint and setup for inference
set -e

CKPT_DIR="/workspace/checkpoints"
mkdir -p "$CKPT_DIR"

echo "[Setup] Downloading Zero123-XL checkpoint..."

# Zero123-XL weights are hosted on Hugging Face
# Download the 105000.ckpt checkpoint
if [ ! -f "$CKPT_DIR/zero123-xl.ckpt" ]; then
    # Use HF CLI to download
    pip install -q -U huggingface_hub
    
    # Download from cvlab/zero123-weights or the project's official HF repo
    python - <<'PY'
import os
from huggingface_hub import hf_hub_download

ckpt_dir = "/workspace/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# Try downloading from known locations
try:
    # Method 1: Official weights
    file = hf_hub_download(
        repo_id="cvlab/zero123-weights",
        filename="105000.ckpt",
        local_dir=ckpt_dir,
        local_dir_use_symlinks=False
    )
    print(f"Downloaded to {file}")
except Exception as e:
    print(f"Failed from cvlab/zero123-weights: {e}")
    
    # Method 2: Community mirror
    try:
        file = hf_hub_download(
            repo_id="kxic/zero123-xl",
            filename="105000.ckpt",
            local_dir=ckpt_dir,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded to {file}")
    except Exception as e2:
        print(f"Failed from kxic/zero123-xl: {e2}")
        print("Manual download required. See: https://github.com/cvlab-columbia/zero123")
        exit(1)
PY
    
    # Normalize name
    mv "$CKPT_DIR/105000.ckpt" "$CKPT_DIR/zero123-xl.ckpt" 2>/dev/null || true
    ls -lh "$CKPT_DIR/zero123-xl.ckpt"
else
    echo "[Setup] Checkpoint already exists at $CKPT_DIR/zero123-xl.ckpt"
fi

echo "[Setup] Done. Zero123-XL checkpoint ready."

