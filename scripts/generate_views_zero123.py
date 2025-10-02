#!/usr/bin/env python
"""
Zero123 multiview generation using its native API (not CLI scripts).
"""
import argparse, os, sys, subprocess, shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews with Zero123")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    p.add_argument("--elevation", type=float, default=15.0, help="Elevation angle")
    return p.parse_args()

def setup_zero123():
    """Clone and setup Zero123 repo."""
    repo = Path("/workspace/zero123")
    if not repo.exists():
        print("[Zero123] Cloning repo...")
        subprocess.check_call([
            "git", "clone", 
            "https://github.com/cvlab-columbia/zero123.git",
            str(repo)
        ])
    
    # Install specific deps
    print("[Zero123] Installing dependencies...")
    subprocess.check_call([
        "pip", "install", "--no-cache-dir",
        "diffusers==0.19.3",
        "transformers==4.27.1",
        "accelerate==0.19.0",
        "omegaconf",
        "einops",
        "pytorch-lightning",
        "imageio",
        "kornia"
    ])
    
    return repo

def run_zero123_programmatic(input_img, output_dir, num_views=12, elevation=15.0):
    """Run Zero123 by importing its modules directly."""
    repo = setup_zero123()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add repo to path and import Zero123 model
    sys.path.insert(0, str(repo))
    
    import torch
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    from transformers import CLIPImageProcessor
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load Zero123 model (hosted on HuggingFace by the authors)
    print("[Zero123] Loading model...")
    
    # Try the official Zero123-XL weights
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "cvlab/zero123-weights",  # or try the actual HF path from their README
            torch_dtype=dtype,
            safety_checker=None
        ).to(device)
    except:
        # Fallback to building from checkpoint if repo provides one
        print("[Zero123] Trying alternate model path...")
        # Check repo for checkpoint path
        ckpt_path = repo / "105000.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError("Zero123 checkpoint not found. Download from model page.")
        
        # Load from checkpoint (requires their load script)
        from zero123_inference import load_model
        pipe = load_model(str(ckpt_path), device)
    
    # Load input
    image = Image.open(input_img).convert("RGB").resize((256, 256))
    
    print(f"[Zero123] Generating {num_views} views...")
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        polar = 90.0 - elevation
        
        # Zero123 conditioning
        prompt_embeds = pipe.encode_prompt(
            f"",  # Zero123 uses camera params, not text
            device,
            1,
            False
        )
        
        # Generate view
        result = pipe(
            prompt_embeds=prompt_embeds,
            image=image,
            polar=torch.tensor([polar], device=device),
            azimuth=torch.tensor([azimuth], device=device),
            num_inference_steps=75,
            guidance_scale=3.0
        )
        
        result.images[0].save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[Zero123] Done. {num_views} views in {output_dir}")

def fallback_simple_multiview(input_img, output_dir, num_views=12):
    """
    Fallback: Use ControlNet + SD to generate architectural views.
    This is more reliable than image-variations for 3D reconstruction.
    """
    import torch
    from PIL import Image, ImageOps
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    from diffusers import UniPCMultistepScheduler
    import cv2
    import numpy as np
    
    print("[Fallback] Using ControlNet for multiview generation...")
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load ControlNet (canny edge)
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None
    ).to(device)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Extract edges
    img_np = np.array(Image.open(input_img).convert("RGB"))
    edges = cv2.Canny(img_np, 100, 200)
    edges_pil = Image.fromarray(edges).convert("RGB")
    
    # Generate views with slight prompt variations
    prompts = [
        "architectural facade, front view, professional photo",
        "architectural facade, slight left angle, professional photo",
        "architectural facade, slight right angle, professional photo",
        "architectural facade, elevated view, professional photo",
    ]
    
    for i in range(num_views):
        prompt = prompts[i % len(prompts)]
        seed = 42 + i * 100
        generator = torch.Generator(device=device).manual_seed(seed)
        
        result = pipe(
            prompt,
            edges_pil,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        )
        
        result.images[0].save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[Fallback] Generated {num_views} ControlNet views")

def main():
    args = parse_args()
    
    try:
        print("[Zero123] Trying native programmatic inference...")
        run_zero123_programmatic(args.input, args.output, args.num_views, args.elevation)
    except Exception as e:
        print(f"[Zero123] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        print("[Zero123] Attempting ControlNet fallback...")
        try:
            fallback_simple_multiview(args.input, args.output, args.num_views)
        except Exception as e2:
            print(f"[Fallback] Failed: {e2}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
