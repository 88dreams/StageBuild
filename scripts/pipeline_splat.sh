#!/usr/bin/env bash
set -e

# Splat workflow: image → multiviews → COLMAP → 3DGS → SuGaR → mesh
# Usage: pipeline_splat.sh /workspace/input/facade.jpg /workspace/output/splat1

INPUT_IMG="${1:-/workspace/input/input.jpg}"
OUT_DIR="${2:-/workspace/output/splat1}"

mkdir -p "$OUT_DIR"

echo "[StageBuild Splat] Input: $INPUT_IMG"
echo "[StageBuild Splat] Output dir: $OUT_DIR"

# 1) Generate multiviews
VIEW_DIR="$OUT_DIR/views"
mkdir -p "$VIEW_DIR"

echo "[StageBuild Splat] Generating multiviews with Stable-Zero123..."
python /workspace/scripts/generate_views_stable.py \
  --input "$INPUT_IMG" \
  --output "$VIEW_DIR" \
  --num_views 12

# 2) COLMAP: recover camera poses from the multiviews
COLMAP_DIR="$OUT_DIR/colmap"
mkdir -p "$COLMAP_DIR/sparse" "$COLMAP_DIR/dense"

echo "[StageBuild Splat] Running COLMAP SfM..."
cd "$OUT_DIR"
touch "$COLMAP_DIR/database.db"

colmap feature_extractor \
  --database_path "$COLMAP_DIR/database.db" \
  --image_path "$VIEW_DIR" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1

colmap exhaustive_matcher \
  --database_path "$COLMAP_DIR/database.db" \
  --SiftMatching.use_gpu 1

colmap mapper \
  --database_path "$COLMAP_DIR/database.db" \
  --image_path "$VIEW_DIR" \
  --output_path "$COLMAP_DIR/sparse"

# 3) Train 3D Gaussian Splatting
GS_OUT="$OUT_DIR/gs_out"
mkdir -p "$GS_OUT"

echo "[StageBuild Splat] Training 3DGS..."
if [ ! -d "/workspace/gaussian-splatting" ]; then
  cd /workspace && git clone https://github.com/graphdeco-inria/gaussian-splatting.git
  pip install -r /workspace/gaussian-splatting/requirements.txt || true
fi

python /workspace/gaussian-splatting/train.py \
  -s "$OUT_DIR" \
  -m "$GS_OUT" \
  --eval

# 4) Extract mesh from splat with SuGaR
MESH_DIR="$OUT_DIR/mesh"
mkdir -p "$MESH_DIR"

echo "[StageBuild Splat] Extracting mesh with SuGaR..."
if [ ! -d "/workspace/SuGaR" ]; then
  cd /workspace && git clone https://github.com/Anttwo/SuGaR.git
  pip install -r /workspace/SuGaR/requirements.txt || true
fi

PLY=$(find "$GS_OUT/point_cloud" -name "*.ply" | tail -n 1)
if [ -z "$PLY" ]; then
  echo "[StageBuild Splat] ERROR: No .ply found in $GS_OUT/point_cloud"
  exit 1
fi

python /workspace/SuGaR/train.py \
  -s "$OUT_DIR" \
  -c "$PLY" \
  -r "densify" \
  -o "$MESH_DIR"

echo "[StageBuild Splat] Done."
echo "Views:  $VIEW_DIR"
echo "Splat:  $PLY"
echo "Mesh:   $MESH_DIR"

