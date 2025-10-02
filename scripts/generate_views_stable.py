#!/usr/bin/env python
"""
Multiview generation with Wonder3D (primary) and Zero123 (fallback).
"""
import argparse, os, sys, shutil, subprocess
from pathlib import Path
from PIL import Image
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=6, help="Number of views (Wonder3D outputs 6)")
    return p.parse_args()

def run_wonder3d(input_img, output_dir):
    """Use Wonder3D repo directly with correct inference script."""
    repo = Path("/workspace/Wonder3D")
    if not repo.exists():
        print("[generate_views] Cloning Wonder3D...")
        subprocess.check_call(["git", "clone", "https://github.com/xxlong0/Wonder3D.git", str(repo)])
    
    # Install deps (Wonder3D needs specific versions)
    req = repo / "requirements.txt"
    if req.exists():
        print("[generate_views] Installing Wonder3D dependencies...")
        subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", str(req)])
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Wonder3D uses instant-nsr-pl for inference
    # Check for the actual inference script
    gradio_app = repo / "instant-nsr-pl" / "run_gradio.py"
    infer_script = repo / "instant-nsr-pl" / "run.py"
    
    # Wonder3D typically outputs 6 views (front, back, left, right, top, bottom)
    # The run script takes --image_path and --output_path
    if infer_script.exists():
        subprocess.check_call([
            "python", str(infer_script),
            "--image_path", input_img,
            "--output_path", output_dir,
            "--save_rgb"
        ], cwd=str(repo))
    else:
        # Try programmatic API
        sys.path.insert(0, str(repo))
        from run import Wonder3DInference
        
        model = Wonder3DInference()
        model.run_single_image(input_img, output_dir)

def run_mvdream(input_img, output_dir, num_views=12):
    """Use MVDream as alternative (multiview diffusion)."""
    repo = Path("/workspace/MVDream")
    if not repo.exists():
        print("[generate_views] Cloning MVDream...")
        subprocess.check_call(["git", "clone", "https://github.com/bytedance/MVDream.git", str(repo)])
        req = repo / "requirements.txt"
        if req.exists():
            subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", str(req)])
    
    # MVDream has a simpler interface
    sys.path.insert(0, str(repo))
    from mvdream.pipeline_mvdream import MVDreamPipeline
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = MVDreamPipeline.from_pretrained("MVDream/MVDream", torch_dtype=dtype, trust_remote_code=True)
    pipe = pipe.to(device)
    
    image = Image.open(input_img).convert("RGB")
    
    # MVDream generates 4 orthogonal views by default; we can run multiple times
    for i in range(num_views // 4):
        result = pipe(image, num_inference_steps=50, guidance_scale=7.5)
        for j, img in enumerate(result.images):
            img.save(os.path.join(output_dir, f"view_{i*4+j:02d}.png"))

def fallback_duplicate(input_img, output_dir, num_views=12):
    """Emergency fallback: duplicate input into ring."""
    print("[generate_views] Using emergency fallback: duplicating input...")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_views):
        shutil.copy(input_img, os.path.join(output_dir, f"view_{i:02d}.png"))
    print(f"[generate_views] Created {num_views} duplicate views (limited quality)")

def main():
    args = parse_args()
    
    # Try Wonder3D first
    try:
        print("[generate_views] Trying Wonder3D...")
        run_wonder3d(args.input, args.output)
        # Wonder3D outputs 6 views; if user wants 12, duplicate some
        existing = sorted(Path(args.output).glob("*.png"))
        if len(existing) == 6 and args.num_views == 12:
            for i, src in enumerate(existing):
                shutil.copy(src, os.path.join(args.output, f"view_{i+6:02d}.png"))
        print(f"[generate_views] Done. Views in {args.output}")
        return
    except Exception as e:
        print(f"[generate_views] Wonder3D failed: {e}", file=sys.stderr)
    
    # Try MVDream
    try:
        print("[generate_views] Trying MVDream...")
        run_mvdream(args.input, args.output, args.num_views)
        print(f"[generate_views] Done. Views in {args.output}")
        return
    except Exception as e:
        print(f"[generate_views] MVDream failed: {e}", file=sys.stderr)
    
    # Emergency fallback
    print("[generate_views] All generators failed. Using duplicate fallback...")
    fallback_duplicate(args.input, args.output, args.num_views)

if __name__ == "__main__":
    main()
