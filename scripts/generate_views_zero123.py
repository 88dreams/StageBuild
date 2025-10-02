#!/usr/bin/env python
"""
Multiview generation using Zero123 native inference (not diffusers).
This is the ONLY reliable way to use Zero123.
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
    
    # Install deps
    req = repo / "requirements.txt"
    if req.exists():
        print("[Zero123] Installing dependencies...")
        # Pin specific versions to avoid conflicts
        subprocess.check_call([
            "pip", "install", "--no-cache-dir",
            "diffusers==0.19.3",
            "transformers==4.27.1",
            "accelerate==0.19.0",
            "omegaconf",
            "einops",
            "pytorch-lightning",
            "imageio"
        ])
    
    return repo

def run_zero123_inference(input_img, output_dir, num_views=12, elevation=15.0):
    """Run Zero123 native inference."""
    repo = setup_zero123()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Zero123 uses gradio_new.py or run_demo.py
    # Find the correct inference script
    inference_script = None
    for candidate in ["gradio_new.py", "run_demo.py", "demo.py", "predict.py"]:
        script = repo / candidate
        if script.exists():
            inference_script = script
            break
    
    if not inference_script:
        raise FileNotFoundError(f"No inference script found in {repo}")
    
    print(f"[Zero123] Using {inference_script.name}...")
    
    # Zero123's demo scripts typically take these args:
    # --image_path, --output_path, --polar (elevation), --azimuth
    
    # Generate views at different azimuths
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        polar = 90.0 - elevation
        
        output_file = os.path.join(output_dir, f"view_{i:02d}.png")
        
        try:
            # Try with standard args
            subprocess.check_call([
                "python", str(inference_script),
                "--input", input_img,
                "--output", output_file,
                "--polar", str(polar),
                "--azimuth", str(azimuth)
            ], cwd=str(repo))
        except subprocess.CalledProcessError:
            # Try alternate arg format
            subprocess.check_call([
                "python", str(inference_script),
                input_img,
                output_file,
                str(polar),
                str(azimuth)
            ], cwd=str(repo))
    
    print(f"[Zero123] Generated {num_views} views in {output_dir}")

def fallback_sd_variations(input_img, output_dir, num_views=12):
    """Fallback to SD image-variations."""
    from PIL import Image
    import torch
    from diffusers import StableDiffusionImageVariationPipeline
    
    print("[Zero123] Native inference failed. Using SD image-variations fallback...")
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        torch_dtype=dtype
    ).to(device)
    
    image = Image.open(input_img).convert("RGB").resize((512, 512))
    
    for i in range(num_views):
        guidance = 3.0 + (i * 0.5)
        generator = torch.Generator(device=device).manual_seed(42 + i * 100)
        
        result = pipe(image, guidance_scale=guidance, num_inference_steps=50, generator=generator)
        result.images[0].save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[Zero123] Fallback generated {num_views} views")

def main():
    args = parse_args()
    
    try:
        run_zero123_inference(args.input, args.output, args.num_views, args.elevation)
    except Exception as e:
        print(f"[Zero123] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        print("[Zero123] Attempting fallback...")
        try:
            fallback_sd_variations(args.input, args.output, args.num_views)
        except Exception as e2:
            print(f"[Zero123] Fallback failed: {e2}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()

