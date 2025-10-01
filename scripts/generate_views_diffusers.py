#!/usr/bin/env python
"""
Multiview generation with robust repo detection and fallbacks.
Tries: Zero123++ → SyncDreamer → TripoSR multiview hack.
"""
import argparse, os, sys, subprocess, shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(description="Generate multiviews")
    p.add_argument("--input", required=True, help="Path to input image")
    p.add_argument("--output", required=True, help="Directory to write generated views")
    p.add_argument("--num_views", type=int, default=12, help="Number of views")
    return p.parse_args()

def ensure_zero123plus():
    repo = Path("/workspace/zero123plus")
    if not repo.exists():
        print("[generate_views] Cloning Zero123++...")
        subprocess.check_call(["git", "clone", "https://github.com/SUDO-AI-3D/zero123plus.git", str(repo)])
        req = repo / "requirements.txt"
        if req.exists():
            subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", str(req)])
    return repo

def ensure_syncdreamer():
    repo = Path("/workspace/SyncDreamer")
    if not repo.exists():
        print("[generate_views] Cloning SyncDreamer...")
        subprocess.check_call(["git", "clone", "https://github.com/liuyuan-pal/SyncDreamer.git", str(repo)])
        req = repo / "requirements.txt"
        if req.exists():
            # Pin numpy<2 to avoid conflicts
            subprocess.check_call(["pip", "install", "--no-cache-dir", "numpy<2.0"])
            subprocess.check_call(["pip", "install", "--no-cache-dir", "-r", str(req)])
    return repo

def find_script(repo, candidates):
    for c in candidates:
        s = repo / c
        if s.exists():
            return s
    return None

def run_zero123plus(input_img, output_dir, num_views=12):
    repo = ensure_zero123plus()
    script = find_script(repo, ["gradio_new.py", "run_img.py", "demo.py", "infer.py", "app.py"])
    if not script:
        raise FileNotFoundError(f"No inference script found in {repo}")
    
    os.makedirs(output_dir, exist_ok=True)
    # Try common CLI patterns; adjust based on actual script
    try:
        subprocess.check_call([
            "python", str(script),
            "--input", input_img,
            "--output", output_dir,
            "--num_views", str(num_views)
        ], cwd=str(repo))
    except subprocess.CalledProcessError:
        # Try alternate arg names
        subprocess.check_call([
            "python", str(script),
            input_img, output_dir
        ], cwd=str(repo))

def run_syncdreamer(input_img, output_dir, num_views=12):
    repo = ensure_syncdreamer()
    # SyncDreamer typically uses main.py or scripts/run_inference.py
    script = find_script(repo, ["main.py", "scripts/run_inference.py", "run.py", "demo.py"])
    if not script:
        raise FileNotFoundError(f"No inference script found in {repo}")
    
    os.makedirs(output_dir, exist_ok=True)
    # Common SyncDreamer usage:
    subprocess.check_call([
        "python", str(script),
        "--input", input_img,
        "--output", output_dir
    ], cwd=str(repo))

def fallback_multiview_hack(input_img, output_dir, num_views=12):
    """Duplicate input into a ring as emergency fallback."""
    print("[generate_views] Using emergency fallback: duplicating input image...")
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_views):
        shutil.copy(input_img, os.path.join(output_dir, f"view_{i:02d}.png"))
    print(f"[generate_views] Created {num_views} duplicate views (quality will be limited)")

def main():
    args = parse_args()
    
    # Try Zero123++
    try:
        print("[generate_views] Trying Zero123++...")
        run_zero123plus(args.input, args.output, args.num_views)
        print(f"[generate_views] Done. Views in {args.output}")
        return
    except Exception as e:
        print(f"[generate_views] Zero123++ failed: {e}")
    
    # Try SyncDreamer
    try:
        print("[generate_views] Trying SyncDreamer...")
        run_syncdreamer(args.input, args.output, args.num_views)
        print(f"[generate_views] Done. Views in {args.output}")
        return
    except Exception as e:
        print(f"[generate_views] SyncDreamer failed: {e}")
    
    # Emergency fallback
    print("[generate_views] All generators failed. Using fallback hack.", file=sys.stderr)
    try:
        fallback_multiview_hack(args.input, args.output, args.num_views)
    except Exception as e:
        print(f"[generate_views] Fallback also failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
