"""
depth_to_pointcloud.py

Converts a Depth Anything V2 depth map + original image into a colored .ply point cloud.
Estimated camera intrinsics from image size (no calibration needed).

Usage:
    python depth_to_pointcloud.py --image test.png --depth depth_output/test.png --out results/cloud.ply

Or import and call directly:
    from depth_to_pointcloud import generate_pointcloud
    generate_pointcloud(image_path, depth_map_path, output_ply_path)
"""

import argparse
import os
import cv2
import numpy as np


def generate_pointcloud(
    image_path: str,
    depth_map_path: str,
    output_path: str,
    ceiling_height_ft: float = 10.0,
    downsample: int = 2,
    min_confidence: float = 0.05,
):
    """
    Generate a colored PLY point cloud from an image + depth map.

    Args:
        image_path:        Path to the original RGB image
        depth_map_path:    Path to the depth map (grayscale PNG from run.py)
        output_path:       Output .ply file path
        ceiling_height_ft: Real-world scene height for depth scaling
        downsample:        Skip every N pixels (2 = quarter points, good for demos)
        min_confidence:    Ignore depth pixels below this normalized value (removes sky/noise)
    """
    image     = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if depth_map is None:
        raise FileNotFoundError(f"Cannot read depth map: {depth_map_path}")

    h, w = image.shape[:2]

    # Resize depth to match image if needed
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert BGR → RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ── Estimate camera intrinsics from image size ────────────────────────────
    # Assume ~60° horizontal FOV (reasonable for phones/construction cameras)
    fov_h_deg  = 60.0
    fov_h_rad  = np.deg2rad(fov_h_deg)
    fx         = (w / 2.0) / np.tan(fov_h_rad / 2.0)
    fy         = fx  # assume square pixels
    cx         = w / 2.0
    cy         = h / 2.0

    # ── Scale depth to real-world meters ─────────────────────────────────────
    ceiling_height_m = ceiling_height_ft * 0.3048
    depth_norm       = depth_map.astype(np.float32) / 255.0  # 0..1
    depth_m          = depth_norm * ceiling_height_m          # 0..ceiling in meters

    # ── Build pixel grid ──────────────────────────────────────────────────────
    u = np.arange(0, w, downsample)
    v = np.arange(0, h, downsample)
    uu, vv = np.meshgrid(u, v)

    d = depth_m[vv, uu]
    r = image_rgb[vv, uu, 0]
    g = image_rgb[vv, uu, 1]
    b = image_rgb[vv, uu, 2]

    # ── Filter out low-confidence (very dark = far/uncertain) depth pixels ────
    confidence_mask = depth_norm[vv, uu] > min_confidence
    uu = uu[confidence_mask]
    vv = vv[confidence_mask]
    d  = d[confidence_mask]
    r  = r[confidence_mask]
    g  = g[confidence_mask]
    b  = b[confidence_mask]

    # ── Back-project pixels → 3D ──────────────────────────────────────────────
    # Standard pinhole camera model:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    # Z = depth
    X = (uu - cx) * d / fx
    Y = (vv - cy) * d / fy  # Y is downward in image space
    Z = d

    # Flip Y so "up" is positive (standard 3D convention)
    Y = -Y

    points = np.stack([X, Y, Z], axis=-1)
    colors = np.stack([r, g, b], axis=-1)

    # ── Write PLY ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    write_ply(output_path, points, colors)

    print(f"  Point cloud: {len(points):,} points → {output_path}")
    return output_path


def write_ply(filepath: str, points: np.ndarray, colors: np.ndarray):
    """Write a binary PLY file with XYZ + RGB."""
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
        # Interleave xyz + rgb per vertex
        for i in range(n):
            f.write(xyz[i].tobytes())
            f.write(rgb[i].tobytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth map → colored PLY point cloud')
    parser.add_argument('--image',  required=True, help='Original RGB image')
    parser.add_argument('--depth',  required=True, help='Depth map PNG from Depth Anything V2')
    parser.add_argument('--out',    required=True, help='Output .ply file')
    parser.add_argument('--ceiling-height', type=float, default=10.0, help='Scene height in feet')
    parser.add_argument('--downsample',     type=int,   default=2,    help='Sample every N pixels (1=full, 2=quarter, 3=ninth...)')
    args = parser.parse_args()

    generate_pointcloud(
        image_path        = args.image,
        depth_map_path    = args.depth,
        output_path       = args.out,
        ceiling_height_ft = args.ceiling_height,
        downsample        = args.downsample,
    )