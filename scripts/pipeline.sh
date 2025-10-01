#!/usr/bin/env bash
set -e

# Usage: pipeline.sh /workspace/input/facade.jpg /workspace/output/run1
INPUT_IMG="${1:-/workspace/input/input.jpg}"
OUT_DIR="${2:-/workspace/output/run1}"
CKPT_DIR="/workspace/checkpoints"
Z123_CKPT="${Z123_CKPT:-$CKPT_DIR/zero123plus.pth}"

mkdir -p "$OUT_DIR"

echo "[StageBuild] Input: $INPUT_IMG"
echo "[StageBuild] Output dir: $OUT_DIR"
echo "[StageBuild] Zero123++ ckpt: $Z123_CKPT"

# 1) Generate multiviews
VIEW_DIR="$OUT_DIR/views"
mkdir -p "$VIEW_DIR"

if [ -f "$Z123_CKPT" ]; then
  echo "[StageBuild] Using Zero123++ checkpoint at $Z123_CKPT"
  python zero123plus/demo.py \
    --input "$INPUT_IMG" \
    --output "$VIEW_DIR" \
    --checkpoint "$Z123_CKPT"
else
  echo "[StageBuild] No checkpoint found. Falling back to diffusers Zero123Plus (sudo-ai/zero123plus-v1.2)"
  python /workspace/scripts/generate_views_diffusers.py \
    --input "$INPUT_IMG" \
    --output "$VIEW_DIR" \
    --num_views 12
fi

# 2) Reconstruct mesh
MESH_DIR="$OUT_DIR/mesh"
mkdir -p "$MESH_DIR"

# Prefer TripoSR when available (direct single-image->mesh), otherwise fallback to InstantMesh
if [ -d "/workspace/TripoSR" ]; then
  echo "[StageBuild] Using TripoSR for mesh reconstruction"
  mkdir -p "$MESH_DIR/0"
  python TripoSR/run.py "$INPUT_IMG" \
    --output-dir "$MESH_DIR" \
    --device cuda \
    --model-save-format glb \
    --no-remove-bg || {
      echo "[StageBuild] TripoSR failed; falling back to InstantMesh";
      python InstantMesh/run.py InstantMesh/configs/instant-mesh-base.yaml \
        "$VIEW_DIR" \
        --output_path "$MESH_DIR" \
        --view 6 --no_rembg; }
else
  echo "[StageBuild] Using InstantMesh for mesh reconstruction"
  python InstantMesh/run.py InstantMesh/configs/instant-mesh-base.yaml \
    "$VIEW_DIR" \
    --output_path "$MESH_DIR" \
    --view 6 --no_rembg
fi

echo "[StageBuild] Done."
echo "Views: $VIEW_DIR"
echo "Mesh:  $MESH_DIR"