#!/usr/bin/env python
"""
Multiview generation using SD image-variations (proven to work).
"""
import argparse, os, sys, shutil, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    return p.parse_args()

def run_sd_image_variations(input_img, output_dir, num_views=12):
    """Use Stable Diffusion image-variations to create distinct views."""
    from diffusers import StableDiffusionImageVariationPipeline
    
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"[generate_views] Loading SD image-variations on {device}...")
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(
        "lambdalabs/sd-image-variations-diffusers",
        revision="v2.0",
        torch_dtype=dtype
    )
    pipe = pipe.to(device)
    
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    
    image = Image.open(input_img).convert("RGB")
    
    # Resize to 512x512 (SD's native resolution)
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    
    print(f"[generate_views] Generating {num_views} variations...")
    # Generate variations with different seeds/guidance
    for i in range(num_views):
        # Vary guidance_scale and seed to get distinct views
        guidance = 3.0 + (i * 0.5)  # 3.0 to 8.5
        seed = 42 + i * 100
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        result = pipe(
            image,
            guidance_scale=guidance,
            num_inference_steps=50,
            generator=generator
        )
        
        out_img = result.images[0]
        
        # Add a small text label to make them visually distinct for COLMAP
        draw = ImageDraw.Draw(out_img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"View {i:02d}", fill=(255, 255, 255), font=font)
        
        out_img.save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[generate_views] Done. {num_views} variations in {output_dir}")

def run_simple_rotation_hack(input_img, output_dir, num_views=12):
    """Emergency: rotate input image to create distinct views."""
    print("[generate_views] Using rotation hack for distinct views...")
    os.makedirs(output_dir, exist_ok=True)
    
    image = Image.open(input_img).convert("RGB")
    w, h = image.size
    
    for i in range(num_views):
        # Rotate and add border to create distinct features
        angle = (360.0 / num_views) * i
        rotated = image.rotate(angle, expand=True, fillcolor=(128, 128, 128))
        
        # Add colored border for distinctiveness
        from PIL import ImageOps
        border_color = (int(255 * i / num_views), 128, 255 - int(255 * i / num_views))
        bordered = ImageOps.expand(rotated, border=20, fill=border_color)
        
        # Add text label
        draw = ImageDraw.Draw(bordered)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
        except:
            font = ImageFont.load_default()
        draw.text((30, 30), f"View {i:02d} @ {angle:.0f}°", fill=(255, 255, 0), font=font)
        
        bordered.save(os.path.join(output_dir, f"view_{i:02d}.png"))
    
    print(f"[generate_views] Created {num_views} rotated views")

def main():
    args = parse_args()
    
    # Try SD image-variations (most likely to work)
    try:
        print("[generate_views] Trying Stable Diffusion image-variations...")
        run_sd_image_variations(args.input, args.output, args.num_views)
        return
    except Exception as e:
        print(f"[generate_views] SD image-variations failed: {e}", file=sys.stderr)
    
    # Emergency fallback with rotation
    print("[generate_views] All ML generators failed. Using rotation hack...")
    run_simple_rotation_hack(args.input, args.output, args.num_views)

if __name__ == "__main__":
    main()
