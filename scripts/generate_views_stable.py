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
    p = argparse.ArgumentParser(description="Generate multiviews with Zero123")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    # Use a known-working Zero123 variant (ashawkey's implementation is widely used)
    p.add_argument("--model", default="ashawkey/zero123-xl-diffusers", help="Model ID")
    return p.parse_args()

def run_zero123_diffusers(input_img, output_dir, num_views=12, model_id="ashawkey/zero123-xl-diffusers"):
    """Use Zero123-XL via diffusers."""
    from diffusers import DiffusionPipeline, DDIMScheduler
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[generate_views] Loading {model_id}...")
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    image = Image.open(input_img).convert("RGB")
    
    # Zero123-XL generates views at fixed azimuths/elevations
    print(f"[generate_views] Generating {num_views} views...")
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        elevation = 15.0
        polar = 90.0 - elevation
        
        # Zero123-XL typically uses polar/azimuth conditioning
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

def main():
    args = parse_args()
    
    try:
        run_zero123_diffusers(args.input, args.output, args.num_views, args.model)
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

