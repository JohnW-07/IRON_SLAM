#!/usr/bin/env python3
"""
IronSLAM Pipeline
Usage: python pipeline.py --model model.obj --image site_photo.jpg --outdir results/
"""

import argparse
import os
import sys
import subprocess
from depth_to_pointcloud import generate_pointcloud as gen_pc

def run_step(description, func, *args, **kwargs):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print(f"{'='*50}")
    result = func(*args, **kwargs)
    print(f"  ✓ Done")
    return result

def main():
    parser = argparse.ArgumentParser(description='IronSLAM: As-Built vs As-Designed Comparison')
    parser.add_argument('--model',   type=str, required=True, help='Path to 3D model (.obj or .gltf)')
    parser.add_argument('--image',   type=str, required=True, help='Path to site photo')
    parser.add_argument('--outdir',  type=str, default='results', help='Output directory')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--ceiling-height', type=float, default=10.0, help='Real-world ceiling height in feet')
    parser.add_argument('--downsample', type=int, default=2, help='Point cloud downsample factor')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.image):
        print(f"ERROR: Image file not found: {args.image}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    depth_dir = os.path.join(args.outdir, 'depth')
    os.makedirs(depth_dir, exist_ok=True)

    # ── Step 1: Parse 3D model ──────────────────────────────────────────────
    run_step("Step 1/4: Parsing 3D model → as_designed.json", parse_model,
             args.model, args.outdir)

    # ── Step 2: Generate depth map ─────────────────────────────────────────
    run_step("Step 2/5: Generating depth map via Depth Anything V2", generate_depth,
             args.image, depth_dir, args.encoder)

    # ── Step 2.5: Generate point cloud ─────────────────────────────────────
    basename       = os.path.splitext(os.path.basename(args.image))[0]
    depth_map_path = os.path.join(depth_dir, basename + '.png')
    ply_path       = os.path.join(args.outdir, basename + '_pointcloud.ply')
    run_step("Step 3/5: Generating point cloud → .ply", generate_pointcloud,
             args.image, depth_map_path, ply_path,
             args.ceiling_height, args.downsample)

    # ── Step 3: Query LLM vision for object locations ──────────────────────
    run_step("Step 4/5: Querying vision model for As-Built object locations", query_vision,
             args.image, args.outdir)

    # ── Step 4: Align, compare, and annotate ──────────────────────────────
    run_step("Step 5/5: Comparing As-Built vs As-Designed & annotating image", compare_and_annotate,
             args.image, depth_dir, args.outdir, args.ceiling_height)

    annotated = os.path.join(args.outdir, 'annotated.png')
    print(f"\n{'='*50}")
    print(f"  Pipeline complete!")
    print(f"  Annotated output: {annotated}")
    print(f"{'='*50}\n")


# ── Step implementations ────────────────────────────────────────────────────

def generate_pointcloud(image_path, depth_map_path, ply_path, ceiling_height, downsample):
    gen_pc(
        image_path        = image_path,
        depth_map_path    = depth_map_path,
        output_path       = ply_path,
        ceiling_height_ft = ceiling_height,
        downsample        = downsample,
    )

def parse_model(model_path, outdir):
    from parse_model import parse_and_save
    parse_and_save(model_path, os.path.join(outdir, 'as_designed.json'))

def generate_depth(image_path, depth_dir, encoder):
    # Reuse existing run.py
    result = subprocess.run([
        sys.executable, 'run.py',
        '--encoder', encoder,
        '--img-path', image_path,
        '--outdir', depth_dir,
        '--pred-only'  # depth map only, no side-by-side
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: depth generation stderr:\n{result.stderr}")

def query_vision(image_path, outdir):
    from vision_query import query_and_save
    as_designed_path = os.path.join(outdir, 'as_designed.json')
    query_and_save(image_path, as_designed_path, os.path.join(outdir, 'as_built.json'))

def compare_and_annotate(image_path, depth_dir, outdir, ceiling_height):
    from annotate import compare_and_draw
    import os
    basename = os.path.splitext(os.path.basename(image_path))[0]
    depth_map_path = os.path.join(depth_dir, basename + '.png')
    as_designed_path = os.path.join(outdir, 'as_designed.json')
    as_built_path    = os.path.join(outdir, 'as_built.json')
    out_path         = os.path.join(outdir, 'annotated.png')
    compare_and_draw(image_path, depth_map_path, as_designed_path, as_built_path, out_path, ceiling_height)


if __name__ == '__main__':
    main()