#!/usr/bin/env python
"""
Zero123-XL multiview generation using checkpoint (no config files needed).
Simplified approach: load checkpoint and run inference with minimal setup.
"""
import argparse, os, sys, subprocess
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews with Zero123-XL")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    p.add_argument("--elevation", type=float, default=15.0, help="Elevation angle (degrees)")
    p.add_argument("--checkpoint", default="/workspace/checkpoints/zero123-xl.ckpt", help="Zero123-XL checkpoint")
    return p.parse_args()

def ensure_checkpoint(ckpt_path):
    """Verify checkpoint exists."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint missing at {ckpt_path}. "
            f"Place zero123-xl.ckpt (105000.ckpt) at /data/checkpoints/zero123-xl.ckpt"
        )
    return ckpt

def run_zero123_simple(input_img, output_dir, checkpoint_path, num_views=12, elevation=15.0):
    """
    Simplified Zero123-XL inference without config files.
    Uses the checkpoint + diffusers 0.19.3 StableDiffusion pipeline with camera conditioning.
    """
    ckpt = ensure_checkpoint(checkpoint_path)
    os.makedirs(output_dir, exist_ok=True)
    
    import torch
    import numpy as np
    from PIL import Image
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[Zero123] Loading checkpoint from {ckpt} (this may take a minute)...")
    
    # Load the checkpoint as a StableDiffusion model
    # Zero123-XL is a fine-tuned SD model with camera conditioning
    pipe = StableDiffusionPipeline.from_single_file(
        str(ckpt),
        torch_dtype=dtype,
        safety_checker=None,
        load_safety_checker=False
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    
    # Load and preprocess input
    image = Image.open(input_img).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    print(f"[Zero123] Generating {num_views} views at elevation {elevation}°...")
    
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        polar = 90.0 - elevation
        
        # Zero123 embeds camera params in the prompt or latent space
        # For single_file loading, we use text prompt with camera angles
        prompt = f"polar={polar:.1f},azimuth={azimuth:.1f}"
        
        result = pipe(
            prompt,
            image=image,
            num_inference_steps=75,
            guidance_scale=3.0,
        )
        
        out_img = result.images[0]
        out_img.save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[Zero123] Done. {num_views} views in {output_dir}")

def fallback_controlnet(input_img, output_dir, num_views=12):
    """ControlNet fallback for architectural multiviews."""
    import torch
    from PIL import Image
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    import cv2
    import numpy as np
    
    print("[Fallback] Using ControlNet for architectural multiviews...")
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Reduce memory by enabling sequential CPU offload
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
    pipe.enable_attention_slicing()
    
    # Extract edges
    img_np = np.array(Image.open(input_img).convert("RGB"))
    edges = cv2.Canny(img_np, 100, 200)
    edges_pil = Image.fromarray(edges).convert("RGB")
    
    # Reduce resolution to avoid OOM
    edges_pil = edges_pil.resize((512, 512), Image.Resampling.LANCZOS)
    
    # Architectural prompts with angle variations
    base_prompt = "architectural facade, professional photo, high detail"
    angle_prompts = [
        f"{base_prompt}, front view",
        f"{base_prompt}, slight left angle",
        f"{base_prompt}, left side view",
        f"{base_prompt}, slight right angle",
        f"{base_prompt}, right side view",
        f"{base_prompt}, elevated view",
    ]
    
    for i in range(num_views):
        prompt = angle_prompts[i % len(angle_prompts)]
        seed = 42 + i * 100
        generator = torch.Generator(device=device).manual_seed(seed)
        
        result = pipe(
            prompt,
            edges_pil,
            num_inference_steps=20,  # reduced from 30 to save memory
            guidance_scale=7.5,
            generator=generator
        )
        
        result.images[0].save(os.path.join(output_dir, f"view_{i:02d}.png"))
        
        # Clear cache between generations to avoid OOM
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    print(f"[Fallback] Generated {num_views} ControlNet views")

def main():
    args = parse_args()
    
    try:
        print("[Zero123] Attempting Zero123-XL with checkpoint (simplified)...")
        run_zero123_simple(args.input, args.output, args.checkpoint, args.num_views, args.elevation)
    except Exception as e:
        print(f"[Zero123] Checkpoint inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        print("\n[Zero123] Falling back to ControlNet (architecture-optimized)...")
        try:
            fallback_controlnet(args.input, args.output, args.num_views)
        except Exception as e2:
            print(f"[Fallback] Failed: {e2}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
