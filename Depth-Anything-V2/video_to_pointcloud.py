"""
video_to_pointcloud.py

Extracts frames from a video, runs Depth Anything V2 on each frame,
then fuses all per-frame point clouds into one using ICP alignment.

Usage:
    python video_to_pointcloud.py --video site.mp4 --out results/fused.ply
    python video_to_pointcloud.py --video site.mp4 --out results/fused.ply --mode fast
    python video_to_pointcloud.py --video site.mp4 --out results/fused.ply --mode accurate

Or import:
    from video_to_pointcloud import process_video
    process_video(video_path, output_ply, mode='fast')
"""

import argparse
import os
import sys
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


# ── ICP implementation (pure numpy, no open3d required) ──────────────────────

def icp_align(source: np.ndarray, target: np.ndarray,
              max_iterations: int = 50,
              tolerance: float = 1e-4,
              max_correspondence_dist: float = 0.5) -> np.ndarray:
    """
    Iterative Closest Point alignment.
    Aligns `source` point cloud onto `target`.
    Returns 4x4 transformation matrix.
    
    Uses random subsampling for speed — works well for dense depth clouds.
    """
    from scipy.spatial import KDTree

    # Subsample both clouds for speed (ICP on full clouds is very slow)
    n_samples = min(5000, len(source), len(target))
    src_idx = np.random.choice(len(source), n_samples, replace=False)
    tgt_idx = np.random.choice(len(target), n_samples, replace=False)
    src = source[src_idx].copy()
    tgt = target[tgt_idx].copy()

    T = np.eye(4)  # accumulated transformation

    for iteration in range(max_iterations):
        # Find nearest neighbors in target for each source point
        tree = KDTree(tgt)
        dists, indices = tree.query(src, workers=-1)

        # Filter by max correspondence distance
        mask = dists < max_correspondence_dist
        if mask.sum() < 10:
            print(f"    ICP: too few correspondences at iteration {iteration}, stopping")
            break

        src_matched = src[mask]
        tgt_matched = tgt[indices[mask]]

        # Compute centroids
        src_centroid = src_matched.mean(axis=0)
        tgt_centroid = tgt_matched.mean(axis=0)

        # Center the clouds
        src_centered = src_matched - src_centroid
        tgt_centered = tgt_matched - tgt_centroid

        # SVD to find optimal rotation
        H = src_centered.T @ tgt_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = tgt_centroid - R @ src_centroid

        # Apply transformation to source
        src = (R @ src.T).T + t

        # Accumulate transformation
        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3]  = t
        T = T_step @ T

        # Check convergence
        mean_dist = dists[mask].mean()
        if mean_dist < tolerance:
            print(f"    ICP: converged at iteration {iteration} (mean dist={mean_dist:.5f})")
            break

    return T


def apply_transform(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply 4x4 homogeneous transform to Nx3 point array."""
    ones = np.ones((len(points), 1))
    pts_h = np.hstack([points, ones])
    return (T @ pts_h.T).T[:, :3]


# ── Frame extraction ──────────────────────────────────────────────────────────

def extract_frames(video_path: str, frames_dir: str, mode: str = 'fast') -> list:
    """
    Extract frames from video.
    
    mode='fast'     → 2 fps (good for slow walking shots)
    mode='accurate' → every frame
    
    Returns list of extracted frame paths.
    """
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps

    if mode == 'accurate':
        sample_fps = video_fps  # every frame
    else:
        # Fast mode: 2fps, but at least 8 frames, at most 60
        sample_fps = 2.0
        min_frames = 8
        max_frames = 60
        natural_count = int(duration_s * sample_fps)
        if natural_count < min_frames:
            sample_fps = min_frames / duration_s
        elif natural_count > max_frames:
            sample_fps = max_frames / duration_s

    frame_interval = max(1, int(round(video_fps / sample_fps)))

    print(f"  Video: {total_frames} frames @ {video_fps:.1f}fps ({duration_s:.1f}s)")
    print(f"  Mode: {mode} → sampling every {frame_interval} frames (~{sample_fps:.1f}fps)")

    frame_paths = []
    frame_idx   = 0
    saved_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            path = os.path.join(frames_dir, f"frame_{saved_idx:04d}.png")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"  Extracted {len(frame_paths)} frames")
    return frame_paths


# ── Depth generation ──────────────────────────────────────────────────────────

def generate_depth_maps(frame_paths: list, depth_dir: str, encoder: str = 'vits') -> list:
    """Run Depth Anything V2 on all frames. Returns list of depth map paths."""
    os.makedirs(depth_dir, exist_ok=True)

    # ── Filter blurry frames before depth generation ──
    sharp_paths = []
    blur_count = 0
    scores = []

    for fp in frame_paths:
        sharp, score = is_sharp(fp, threshold=30.0)
        scores.append(score)
        if sharp:
            sharp_paths.append(fp)
        else:
            blur_count += 1
            os.remove(fp)  # delete from frames dir so run.py ignores it

    if not sharp_paths:
        raise RuntimeError("All frames were filtered as blurry. Lower --blur-threshold.")

    print(f"  Sharpness filter: kept {len(sharp_paths)}/{len(frame_paths)} frames "
          f"(removed {blur_count} blurry, min={min(scores):.0f} max={max(scores):.0f} "
          f"median={float(np.median(scores)):.0f})")

    # Run run.py on the entire frames directory at once (more efficient)
    frames_dir = os.path.dirname(frame_paths[0])

    result = subprocess.run([
        sys.executable, 'run.py',
        '--encoder', encoder,
        '--img-path', frames_dir,
        '--outdir', depth_dir,
        '--pred-only'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: depth generation error:\n{result.stderr[-500:]}")

    # Match depth maps to frames
    depth_paths = []
    for fp in frame_paths:
        basename  = os.path.splitext(os.path.basename(fp))[0]
        depth_path = os.path.join(depth_dir, basename + '.png')
        if os.path.exists(depth_path):
            depth_paths.append(depth_path)
        else:
            print(f"  WARNING: no depth map for {basename}, skipping")

    print(f"  Generated {len(depth_paths)}/{len(frame_paths)} depth maps")
    return depth_paths


# ── Per-frame point cloud ─────────────────────────────────────────────────────

def frame_to_pointcloud(image_path: str, depth_path: str,
                         ceiling_height_ft: float = 10.0,
                         downsample: int = 3) -> tuple:
    """
    Convert one frame + depth map to XYZ + RGB arrays.
    Returns (points Nx3, colors Nx3) or (None, None) on failure.
    """
    image     = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if image is None or depth_map is None:
        return None, None

    h, w = image.shape[:2]
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Estimated camera intrinsics (60° FOV)
    fov_h_rad = np.deg2rad(60.0)
    fx = (w / 2.0) / np.tan(fov_h_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0

    ceiling_height_m = ceiling_height_ft * 0.3048
    depth_norm = depth_map.astype(np.float32) / 255.0
    depth_m    = depth_norm * ceiling_height_m

    # Build sampled pixel grid
    u = np.arange(0, w, downsample)
    v = np.arange(0, h, downsample)
    uu, vv = np.meshgrid(u, v)

    d = depth_m[vv, uu]
    r = image_rgb[vv, uu, 0]
    g = image_rgb[vv, uu, 1]
    b = image_rgb[vv, uu, 2]

    # Filter low-confidence depth
    mask = depth_norm[vv, uu] > 0.05
    uu, vv = uu[mask], vv[mask]
    d, r, g, b = d[mask], r[mask], g[mask], b[mask]

    X = (uu - cx) * d / fx
    Y = -(vv - cy) * d / fy   # flip Y upward
    Z = d

    points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    colors = np.stack([r, g, b], axis=-1).astype(np.uint8)

    return points, colors


# ── ICP fusion ────────────────────────────────────────────────────────────────

def fuse_pointclouds(frame_paths: list, depth_paths: list,
                     ceiling_height_ft: float = 10.0,
                     downsample: int = 3,
                     use_icp: bool = True) -> tuple:
    """
    Build per-frame clouds then fuse with ICP alignment.
    Returns (fused_points Nx3, fused_colors Nx3).
    """
    try:
        from scipy.spatial import KDTree
        has_scipy = True
    except ImportError:
        has_scipy = False
        if use_icp:
            print("  WARNING: scipy not installed, falling back to simple merge.")
            print("           Run: pip install scipy")
            use_icp = False

    all_points = []
    all_colors = []
    reference_points = None  # first cloud is the anchor

    for i, (fp, dp) in enumerate(zip(frame_paths, depth_paths)):
        print(f"  Fusing frame {i+1}/{len(frame_paths)}: {os.path.basename(fp)}")
        points, colors = frame_to_pointcloud(fp, dp, ceiling_height_ft, downsample)

        if points is None or len(points) == 0:
            print(f"    Skipping — empty cloud")
            continue

        if use_icp and reference_points is not None:
            # Align this frame's cloud onto the running reference
            try:
                T = icp_align(points, reference_points)
                points = apply_transform(points, T)
                print(f"    ICP aligned ({len(points):,} pts)")
            except Exception as e:
                print(f"    ICP failed ({e}), using raw frame")
        else:
            if i == 0:
                print(f"    Anchor frame ({len(points):,} pts)")

        all_points.append(points)
        all_colors.append(colors)

        # Update reference: use merged cloud so far (subsampled for speed)
        if use_icp:
            merged_so_far = np.vstack(all_points)
            # Keep reference manageable: random subsample to 50k pts
            if len(merged_so_far) > 50000:
                idx = np.random.choice(len(merged_so_far), 50000, replace=False)
                reference_points = merged_so_far[idx]
            else:
                reference_points = merged_so_far

    if not all_points:
        raise RuntimeError("No valid point clouds were generated from any frame.")

    fused_points = np.vstack(all_points)
    fused_colors = np.vstack(all_colors)

    print(f"  Fused cloud: {len(fused_points):,} points total")

    # Optional: voxel downsample to remove duplicate/overlapping points
    fused_points, fused_colors = voxel_downsample(fused_points, fused_colors, voxel_size=0.02)
    print(f"  After deduplication: {len(fused_points):,} points")

    return fused_points, fused_colors


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float = 0.02):
    """Simple voxel grid downsampling to remove duplicate points after fusion."""
    if len(points) == 0:
        return points, colors

    # Quantize points into voxel grid
    voxel_indices = np.floor(points / voxel_size).astype(np.int32)

    # Use a dict to keep one point per voxel (first seen wins)
    seen = {}
    keep = []
    for i, vi in enumerate(map(tuple, voxel_indices)):
        if vi not in seen:
            seen[vi] = i
            keep.append(i)

    keep = np.array(keep)
    return points[keep], colors[keep]


# ── PLY writer ────────────────────────────────────────────────────────────────

def write_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    xyz = points.astype(np.float32)
    rgb = colors.astype(np.uint8)
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        for i in range(n):
            f.write(xyz[i].tobytes())
            f.write(rgb[i].tobytes())

def is_sharp(image_path: str, threshold: float = 30.0) -> tuple:
    """
    Detect motion blur using Laplacian variance.
    Higher variance = sharper image.
    Returns (is_sharp, score).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False, 0.0
    score = cv2.Laplacian(img, cv2.CV_64F).var()
    return score >= threshold, round(score, 2)
# ── Main entry ────────────────────────────────────────────────────────────────

def process_video(video_path: str,
                  output_ply: str,
                  mode: str = 'fast',
                  encoder: str = 'vits',
                  ceiling_height_ft: float = 10.0,
                  downsample: int = 3,
                  keep_frames: bool = False,
                  blur_threshold: float = 30.0) -> str:
    """
    Full pipeline: video → frames → depth maps → fused point cloud.

    Args:
        video_path:        Input video file
        output_ply:        Output .ply path
        mode:              'fast' (2fps) or 'accurate' (every frame)
        encoder:           Depth Anything encoder size
        ceiling_height_ft: Scene height for depth calibration
        downsample:        Pixel skip factor per frame (3 = ~1/9 density, good balance)
        keep_frames:       Keep extracted frames + depth maps after processing

    Returns:
        Path to output .ply file
    """
    os.makedirs(os.path.dirname(output_ply) if os.path.dirname(output_ply) else '.', exist_ok=True)

    # Working directories
    work_dir   = os.path.join(os.path.dirname(output_ply) or '.', '_video_work')
    frames_dir = os.path.join(work_dir, 'frames')
    depth_dir  = os.path.join(work_dir, 'depth')

    try:
        # 1. Extract frames
        print("\n[1/3] Extracting frames...")
        frame_paths = extract_frames(video_path, frames_dir, mode=mode)

        # 2. Generate depth maps
        print("\n[2/3] Generating depth maps...")
        depth_paths = generate_depth_maps(frame_paths, depth_dir, encoder=encoder)

        if not depth_paths:
            raise RuntimeError("No depth maps generated — check that run.py works on a single frame first.")

        # 3. Fuse point clouds with ICP
        print("\n[3/3] Fusing point clouds with ICP alignment...")
        points, colors = fuse_pointclouds(
            frame_paths, depth_paths,
            ceiling_height_ft = ceiling_height_ft,
            downsample        = downsample,
            use_icp           = True
        )

        # 4. Write output
        write_ply(output_ply, points, colors)
        size_mb = os.path.getsize(output_ply) / 1e6
        print(f"\n  ✓ Saved: {output_ply} ({size_mb:.1f} MB, {len(points):,} points)")

    finally:
        if not keep_frames and os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            print(f"  Cleaned up working directory")

    return output_ply


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video → fused point cloud via Depth Anything V2 + ICP')
    parser.add_argument('--video',          type=str,   required=True, help='Input video file')
    parser.add_argument('--out',            type=str,   required=True, help='Output .ply path')
    parser.add_argument('--mode',           type=str,   default='accurate', choices=['fast', 'accurate'],
                        help='fast=2fps, accurate=every frame')
    parser.add_argument('--encoder',        type=str,   default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--ceiling-height', type=float, default=10.0,  help='Scene height in feet')
    parser.add_argument('--downsample',     type=int,   default=2,     help='Pixel skip per frame (2=~1/4 density)')
    parser.add_argument('--keep-frames',    action='store_true',       help='Keep extracted frames and depth maps')
    parser.add_argument('--blur-threshold', type=float, default=10.0,
                    help='Laplacian variance threshold — higher=stricter (50=lenient, 100=default, 200=strict)')
    args = parser.parse_args()

    process_video(
        video_path        = args.video,
        output_ply        = args.out,
        mode              = args.mode,
        encoder           = args.encoder,
        ceiling_height_ft = args.ceiling_height,
        downsample        = args.downsample,
        keep_frames       = args.keep_frames,
    )