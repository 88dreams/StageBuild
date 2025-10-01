# StageBuild

**2D Image → 3D Model Pipeline** for architectural visualization and Unity integration.

Converts single architectural photos/drawings to high-quality 3D meshes via:
- **Direct mesh**: TripoSR (fast feed-forward)
- **Splat workflow**: Multiview generation → COLMAP → 3D Gaussian Splatting → SuGaR mesh extraction

---

## Features

- **Portable Docker image**: build on Mac/Windows, run on cloud GPUs (RunPod, Lambda, etc.)
- **Multiview generation**: Stable-Zero123 / Zero123-XL with automatic fallbacks
- **3D Gaussian Splatting**: COLMAP SfM → 3DGS training → SuGaR mesh extraction
- **Direct mesh fallback**: TripoSR for quick results when splats aren't needed
- **S3 upload**: `meshpush.py` to publish results directly to AWS S3
- **Unity-ready**: GLB/OBJ outputs with proper scale/pivot

---

## Quick Start

### 1. Build Locally (Mac or Windows)
```bash
cd StageBuild
docker buildx build --platform linux/amd64 --load -t stagebuild:latest .
docker tag stagebuild:latest <your-dockerhub-user>/stagebuild:latest
docker push <your-dockerhub-user>/stagebuild:latest
```

### 2. Run on Cloud GPU
Launch a GPU pod (e.g., RTX 4090) with:
- **Image**: `<your-dockerhub-user>/stagebuild:latest`
- **Volume Mount**: `/data`
- **Start Command**:
```bash
bash -lc 'mkdir -p /data/{input,output,checkpoints,.cache/huggingface} && \
ln -sfn /data/input /workspace/input && \
ln -sfn /data/output /workspace/output && \
ln -sfn /data/checkpoints /workspace/checkpoints && \
export HF_HOME=/data/.cache/huggingface && \
tail -f /dev/null'
```

### 3. Run Pipeline
```bash
# Upload your image
wget -O /workspace/input/facade.jpg "https://your-image-url"

# Splat workflow (recommended for quality)
bash /workspace/scripts/pipeline_splat.sh /workspace/input/facade.jpg /workspace/output/splat1

# OR direct mesh (faster, lower quality)
bash /workspace/scripts/pipeline.sh /workspace/input/facade.jpg /workspace/output/run1
```

### 4. Download Results
```bash
tar -C /workspace/output -czf /workspace/output/result.tgz splat1
# Transfer via S3, transfer.sh, or pod file manager
```

---

## Pipelines

### Splat Workflow (`pipeline_splat.sh`)
1. **Multiview generation**: Zero123-XL creates 12 views around the input
2. **COLMAP**: Structure-from-Motion to recover camera poses
3. **3D Gaussian Splatting**: Trains a splat scene from images + poses
4. **SuGaR**: Extracts a clean mesh from the splat
5. **Output**: GLB/OBJ mesh ready for Unity/Blender

### Direct Mesh Workflow (`pipeline.sh`)
1. **Multiview generation** (optional, for InstantMesh fallback)
2. **TripoSR**: Fast feed-forward image → mesh
3. **Output**: GLB mesh

---

## Scripts

- `pipeline_splat.sh` — Full splat workflow (multiviews → COLMAP → 3DGS → SuGaR)
- `pipeline.sh` — Direct mesh via TripoSR
- `generate_views_stable.py` — Multiview generation (Zero123-XL, fallback to duplicate)
- `meshpush.py` — Upload meshes to S3 with public ACL or bucket policy

---

## Folder Structure

```
StageBuild/
├── Dockerfile              # Main image definition (PyTorch, CUDA 12.1, COLMAP, deps)
├── entrypoint.sh           # Container entrypoint
├── scripts/
│   ├── pipeline.sh         # Direct mesh pipeline (TripoSR)
│   ├── pipeline_splat.sh   # Splat pipeline (multiviews → 3DGS → SuGaR)
│   ├── generate_views_stable.py
│   ├── generate_views_diffusers.py  # Legacy/alternate multiview generators
│   └── meshpush.py         # S3 uploader
└── data/
    ├── input/              # Mount for source images
    ├── output/             # Mount for results
    └── checkpoints/        # Mount for model weights (optional)
```

---

## Requirements

- **Local build**: Docker Desktop with buildx (Mac/Windows/Linux)
- **Cloud run**: NVIDIA GPU (RTX 4090, A100, etc.) with CUDA 12.1+ drivers
- **Storage**: ~50 GB for model caches + outputs

---

## Configuration

### Environment Variables
- `HF_HOME`: Hugging Face cache dir (default: `/workspace/.cache/huggingface`)
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: for S3 uploads via `meshpush.py`

### Model Selection
Edit `generate_views_stable.py`:
```python
p.add_argument("--model", default="ashawkey/zero123-xl-diffusers", help="Model ID")
```

---

## Troubleshooting

### Multiview generation fails
- **Emergency fallback**: The script duplicates your input into 12 views so COLMAP/3DGS can still run (quality limited).
- **Alternative models**: Try `stabilityai/stable-diffusion-2-1` or other diffusers-compatible multiview generators.

### COLMAP fails ("not enough matches")
- Use more views (edit `--num_views` in `pipeline_splat.sh`).
- Ensure input has texture/features (plain surfaces fail SfM).

### 3DGS / SuGaR errors
- Check GPU memory (requires ≥16 GB VRAM for large scenes).
- Reduce resolution or view count.

### Docker build I/O errors (Mac)
- Prune: `docker system prune -a --volumes`
- Increase disk: Docker Desktop → Settings → Resources → Disk image size

---

## Documentation

See the `2D to 3D` folder in your Obsidian vault (if available) for:
- **Splat Instructions.md**: detailed splat workflow guide
- **Docker Windows - Splat Instructions.md**: Windows build guide

---

## License

MIT (or specify your license)

---

## Credits

- **Zero123 / Zero123++**: [cvlab-columbia/zero123](https://github.com/cvlab-columbia/zero123), [SUDO-AI-3D/zero123plus](https://github.com/SUDO-AI-3D/zero123plus)
- **3D Gaussian Splatting**: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- **SuGaR**: [Anttwo/SuGaR](https://github.com/Anttwo/SuGaR)
- **TripoSR**: [VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR)
- **InstantMesh**: [TencentARC/InstantMesh](https://github.com/TencentARC/InstantMesh)
- **COLMAP**: [colmap/colmap](https://github.com/colmap/colmap)

---

## Contributing

PRs welcome. Please test on a GPU pod before submitting.

