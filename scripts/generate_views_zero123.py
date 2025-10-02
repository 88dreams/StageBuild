#!/usr/bin/env python
"""
Zero123-XL multiview generation using checkpoint + native inference.
This is the CORRECT way to use Zero123 for quality multiviews.
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

def setup_zero123_repo():
    """Clone Zero123 repo if needed."""
    repo = Path("/workspace/zero123")
    if not repo.exists():
        print("[Zero123] Cloning repo...")
        subprocess.check_call([
            "git", "clone", 
            "https://github.com/cvlab-columbia/zero123.git",
            str(repo)
        ])
    return repo

def ensure_checkpoint(ckpt_path):
    """Download checkpoint if not present."""
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        print(f"[Zero123] Checkpoint not found at {ckpt_path}")
        print("[Zero123] Running setup script to download...")
        subprocess.check_call(["bash", "/workspace/scripts/setup_zero123_weights.sh"])
        
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Checkpoint still missing after setup. "
                f"Download manually from https://github.com/cvlab-columbia/zero123 "
                f"and place at {ckpt_path}"
            )
    return ckpt

def run_zero123_xl(input_img, output_dir, checkpoint_path, num_views=12, elevation=15.0):
    """
    Run Zero123-XL inference using the checkpoint and repo code.
    """
    repo = setup_zero123_repo()
    ckpt = ensure_checkpoint(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Add repo to path
    sys.path.insert(0, str(repo))
    
    import torch
    from PIL import Image
    from omegaconf import OmegaConf
    import pytorch_lightning as pl
    
    # Import Zero123 modules
    try:
        # Different repos have different module names
        from zero123 import model_from_config
    except:
        # Try alternate import
        from ldm.util import instantiate_from_config
        model_from_config = instantiate_from_config
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Zero123] Loading checkpoint from {ckpt}...")
    
    # Load config (Zero123 uses OmegaConf configs)
    config_path = repo / "configs" / "sd-objaverse-finetune-c_concat-256.yaml"
    if not config_path.exists():
        # Try alternate config paths
        for alt in ["configs/sd-zero123-c_concat-256.yaml", "configs/default.yaml"]:
            alt_path = repo / alt
            if alt_path.exists():
                config_path = alt_path
                break
    
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found in {repo}/configs")
    
    config = OmegaConf.load(config_path)
    
    # Load model from checkpoint
    try:
        model = model_from_config(config.model)
        checkpoint = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        model = model.to(device).eval()
    except Exception as e:
        print(f"[Zero123] Failed to load with config approach: {e}")
        # Try pytorch-lightning approach
        from pytorch_lightning import Trainer
        model = pl.LightningModule.load_from_checkpoint(str(ckpt), map_location=device)
        model = model.to(device).eval()
    
    # Load and preprocess input
    image = Image.open(input_img).convert("RGB")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    
    print(f"[Zero123] Generating {num_views} views at elevation {elevation}°...")
    
    import numpy as np
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    for i in range(num_views):
        azimuth = (360.0 / num_views) * i
        polar = 90.0 - elevation
        
        # Prepare camera conditioning
        polar_rad = np.deg2rad(polar)
        azimuth_rad = np.deg2rad(azimuth)
        
        # Zero123 expects T (camera transform) as conditioning
        # This varies by implementation; try common patterns
        with torch.no_grad():
            try:
                # Pattern 1: pass polar/azimuth directly
                output = model(
                    img_tensor,
                    polar=torch.tensor([[polar_rad]], device=device, dtype=torch.float32),
                    azimuth=torch.tensor([[azimuth_rad]], device=device, dtype=torch.float32)
                )
            except:
                # Pattern 2: encode as embedding
                camera_emb = model.get_camera_cond(polar_rad, azimuth_rad)
                output = model(img_tensor, camera_emb)
        
        # Convert output to image
        out_np = output[0].cpu().permute(1, 2, 0).numpy()
        out_np = ((out_np + 1) * 127.5).clip(0, 255).astype(np.uint8)
        out_img = Image.fromarray(out_np)
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
    
    # Architectural prompts with angle variations
    base_prompt = "architectural facade, professional photo, high detail"
    angle_prompts = [
        f"{base_prompt}, front view",
        f"{base_prompt}, slight left angle",
        f"{base_prompt}, left view",
        f"{base_prompt}, slight right angle",
        f"{base_prompt}, right view",
        f"{base_prompt}, elevated view",
    ]
    
    for i in range(num_views):
        prompt = angle_prompts[i % len(angle_prompts)]
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
        print("[Zero123] Attempting Zero123-XL with checkpoint...")
        run_zero123_xl(args.input, args.output, args.checkpoint, args.num_views, args.elevation)
    except Exception as e:
        print(f"[Zero123] Checkpoint inference failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        print("\n[Zero123] Falling back to ControlNet (still high quality for architecture)...")
        try:
            fallback_controlnet(args.input, args.output, args.num_views)
        except Exception as e2:
            print(f"[Fallback] Failed: {e2}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
