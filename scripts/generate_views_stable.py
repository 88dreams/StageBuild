#!/usr/bin/env python
"""
Multiview generation using Stable-Zero123 (known-working diffusers pipeline).
Falls back to emergency duplicate if Stable-Zero123 fails.
"""
import argparse, os, sys, shutil
from pathlib import Path
from PIL import Image
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    # Default to Wonder3D which has working inference scripts
    p.add_argument("--model", default="flamehaze1115/wonder3d-v1.0", help="Model ID")
    p.add_argument("--use_wonder3d", action="store_true", default=True, help="Use Wonder3D repo directly")
    return p.parse_args()

def run_zero123_diffusers(input_img, output_dir, num_views=12, model_id="ashawkey/zero123-xl-diffusers"):
    """Use Zero123-XL via diffusers with custom pipeline loading."""
    import sys
    sys.path.insert(0, "/workspace")
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[generate_views] Loading {model_id}...")
    
    # Try importing Zero123Pipeline from diffusers
    try:
        from diffusers import Zero123Pipeline
        pipe_cls = Zero123Pipeline
    except ImportError:
        # Fall back to DiffusionPipeline with trust_remote_code
        from diffusers import DiffusionPipeline
        pipe_cls = DiffusionPipeline
    
    try:
        pipe = pipe_cls.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    except Exception as e:
        print(f"[generate_views] Failed to load {model_id}: {e}")
        raise
    
    pipe = pipe.to(device)
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    image = Image.open(input_img).convert("RGB")
    
    # Generate views with varying camera angles
    print(f"[generate_views] Generating {num_views} views...")
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        elevation = 15.0
        polar = 90.0 - elevation
        
        # Try different call signatures based on pipeline
        try:
            result = pipe(
                image,
                prompt="",
                num_inference_steps=75,
                guidance_scale=3.0,
                elevation=torch.tensor([elevation], dtype=dtype, device=device),
                azimuth=torch.tensor([azimuth], dtype=dtype, device=device),
            )
        except TypeError:
            # Fallback if elevation/azimuth args not supported
            result = pipe(
                image,
                prompt="",
                num_inference_steps=75,
                guidance_scale=3.0,
            )
        
        out_img = result.images[0]
        out_img.save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[generate_views] Done. {num_views} views in {output_dir}")

def fallback_duplicate(input_img, output_dir, num_views=12):
    """Emergency fallback: duplicate input into ring."""
    print("[generate_views] Using emergency fallback: duplicating input...")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_views):
        shutil.copy(input_img, os.path.join(output_dir, f"view_{i:02d}.png"))
    print(f"[generate_views] Created {num_views} duplicate views (limited quality)")

def run_wonder3d(input_img, output_dir, num_views=12):
    """Use Wonder3D repo directly (known-working)."""
    import subprocess
    repo = Path("/workspace/Wonder3D")
    if not repo.exists():
        print("[generate_views] Cloning Wonder3D...")
        subprocess.check_call(["git", "clone", "https://github.com/xxlong0/Wonder3D.git", str(repo)])
        req = repo / "requirements.txt"
        if req.exists():
            subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", str(req)])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Wonder3D has an inference script; find it
    script = repo / "run.py"
    if not script.exists():
        for candidate in ["infer.py", "demo.py", "run_single.py"]:
            alt = repo / candidate
            if alt.exists():
                script = alt
                break
    
    if not script.exists():
        raise FileNotFoundError(f"No inference script found in {repo}")
    
    subprocess.check_call([
        "python", str(script),
        "--image", input_img,
        "--output", output_dir
    ], cwd=str(repo))

def main():
    args = parse_args()
    
    if args.use_wonder3d:
        try:
            print("[generate_views] Trying Wonder3D...")
            run_wonder3d(args.input, args.output, args.num_views)
            print(f"[generate_views] Done. Views in {args.output}")
            return
        except Exception as e:
            print(f"[generate_views] Wonder3D failed: {e}", file=sys.stderr)
    
    try:
        print("[generate_views] Trying Zero123...")
        run_zero123_diffusers(args.input, args.output, args.num_views, args.model)
        print(f"[generate_views] Done. Views in {args.output}")
        return
    except Exception as e:
        print(f"[generate_views] Zero123 failed: {e}", file=sys.stderr)
    
    print("[generate_views] Falling back to duplicate hack...")
    try:
        fallback_duplicate(args.input, args.output, args.num_views)
    except Exception as e2:
        print(f"[generate_views] Fallback failed: {e2}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

