#!/usr/bin/env python3
"""
IronSLAM Pipeline
Usage: python pipeline.py --model model.obj --video site_video.mp4 --outdir results/
       python pipeline.py --model model.obj --video site_video.mp4 --outdir results/ --mode accurate
"""

import argparse
import os
import sys
import subprocess


def run_step(description, func, *args, **kwargs):
    print(f"\n{'='*50}")
    print(f"  {description}")
    print(f"{'='*50}")
    result = func(*args, **kwargs)
    print(f"  ✓ Done")
    return result


def main():
    parser = argparse.ArgumentParser(description='IronSLAM: As-Built vs As-Designed Comparison')
    parser.add_argument('--model',          type=str,   required=True,  help='Path to 3D model (.obj or .gltf)')
    parser.add_argument('--video',          type=str,   required=True,  help='Path to site video (mp4, mov, etc.)')
    parser.add_argument('--outdir',         type=str,   default='results', help='Output directory')
    parser.add_argument('--encoder',        type=str,   default='vits', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--ceiling-height', type=float, default=10.0,  help='Real-world ceiling height in feet')
    parser.add_argument('--downsample',     type=int,   default=3,     help='Point cloud downsample factor per frame')
    parser.add_argument('--mode',           type=str,   default='fast', choices=['fast', 'accurate'],
                        help='Frame extraction: fast=2fps, accurate=every frame')
    parser.add_argument('--keep-frames',    action='store_true',       help='Keep extracted frames and depth maps')
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.video):
        print(f"ERROR: Video file not found: {args.video}")
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    ply_path       = os.path.join(args.outdir, f'{video_basename}_fused.ply')

    # ── Step 1: Parse 3D model ──────────────────────────────────────────────
    run_step("Step 1/4: Parsing 3D model → as_designed.json",
             parse_model, args.model, args.outdir)

    # ── Step 2: Video → depth maps → fused point cloud ─────────────────────
    run_step(f"Step 2/4: Extracting frames ({args.mode} mode) + depth maps + ICP fusion",
             generate_fused_cloud,
             args.video, ply_path, args.mode, args.encoder,
             args.ceiling_height, args.downsample, args.keep_frames)

    # ── Step 3: Extract representative frame for vision query ───────────────
    keyframe_path = os.path.join(args.outdir, 'keyframe.png')
    run_step("Step 3/4: Extracting keyframe for vision analysis",
             extract_keyframe, args.video, keyframe_path)

    # ── Step 4: Vision query + annotate ────────────────────────────────────
    run_step("Step 4/4: Vision analysis + As-Built vs As-Designed annotation",
             vision_and_annotate,
             keyframe_path, ply_path, args.outdir, args.ceiling_height)

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Pipeline complete! Outputs in: {args.outdir}/")
    print(f"")
    print(f"  {video_basename}_fused.ply   ← open in MeshLab / CloudCompare")
    print(f"  annotated.png               ← labeled comparison image")
    print(f"  annotated_report.json       ← structured discrepancy data")
    print(f"  as_designed.json            ← parsed model objects")
    print(f"  as_built.json               ← detected objects from video")
    print(f"{'='*50}\n")


# ── Step implementations ──────────────────────────────────────────────────────

def parse_model(model_path, outdir):
    from parse_model import parse_and_save
    parse_and_save(model_path, os.path.join(outdir, 'as_designed.json'))


def generate_fused_cloud(video_path, ply_path, mode, encoder,
                          ceiling_height, downsample, keep_frames):
    from video_to_pointcloud import process_video
    process_video(
        video_path        = video_path,
        output_ply        = ply_path,
        mode              = mode,
        encoder           = encoder,
        ceiling_height_ft = ceiling_height,
        downsample        = downsample,
        keep_frames       = keep_frames,
    )


def extract_keyframe(video_path, keyframe_path):
    """Extract the middle frame of the video as the representative keyframe."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(keyframe_path, frame)
        print(f"  Keyframe: frame {total//2}/{total}")
    else:
        raise RuntimeError("Could not extract keyframe from video")


def vision_and_annotate(keyframe_path, ply_path, outdir, ceiling_height):
    """Run vision query on keyframe then generate depth map and annotate."""
    import cv2

    # Generate depth map for the keyframe
    depth_dir = os.path.join(outdir, 'keyframe_depth')
    os.makedirs(depth_dir, exist_ok=True)

    result = subprocess.run([
        sys.executable, 'run.py',
        '--encoder', 'vits',
        '--img-path', keyframe_path,
        '--outdir', depth_dir,
        '--pred-only'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: keyframe depth generation error:\n{result.stderr[-300:]}")

    # Vision query
    from vision_query import query_and_save
    as_designed_path = os.path.join(outdir, 'as_designed.json')
    as_built_path    = os.path.join(outdir, 'as_built.json')
    query_and_save(keyframe_path, as_designed_path, as_built_path)

    # Annotate
    from annotate import compare_and_draw
    depth_map_path = os.path.join(depth_dir, 'keyframe.png')
    compare_and_draw(
        image_path        = keyframe_path,
        depth_map_path    = depth_map_path,
        as_designed_path  = as_designed_path,
        as_built_path     = as_built_path,
        out_path          = os.path.join(outdir, 'annotated.png'),
        ceiling_height_ft = ceiling_height,
    )


if __name__ == '__main__':
    main()