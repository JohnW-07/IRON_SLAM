#!/usr/bin/env python3
"""
IronSLAM Pipeline - Rebuilt
Two independent tracks:
  A) Point cloud generation (visual demo)
  B) Deviation detection via Gemini vision (actual intelligence)

Usage:
    python pipeline.py --model site.obj --video site.mp4 --outdir results/
    python pipeline.py --model site.obj --video site.mp4 --outdir results/ --mode accurate
    python pipeline.py --model site.obj --video site.mp4 --outdir results/ --skip-pointcloud
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='IronSLAM: BIM deviation detection from body camera footage')
    p.add_argument('--model',           type=str,   required=True,       help='.obj or .gltf BIM model')
    p.add_argument('--image', type=str, default=None, help='Single image for testing (skips point cloud)')
    p.add_argument('--video',           type=str,   default=None,       help='Body camera footage')
    p.add_argument('--outdir',          type=str,   default='results',   help='Output directory')
    p.add_argument('--encoder',         type=str,   default='vits',      choices=['vits', 'vitb', 'vitl'])
    p.add_argument('--mode',            type=str,   default='fast',      choices=['fast', 'accurate'],
                   help='fast=2fps, accurate=every frame')
    p.add_argument('--blur-threshold',  type=float, default=15.0,
                   help='Laplacian sharpness threshold. Higher=stricter. Try 50-200.')
    p.add_argument('--keyframes',       type=int,   default=8,
                   help='Number of best keyframes to use for deviation detection')
    p.add_argument('--ceiling-height',  type=float, default=10.0,        help='Scene height in feet')
    p.add_argument('--downsample',      type=int,   default=3,           help='Point cloud pixel skip factor')
    p.add_argument('--skip-pointcloud', action='store_true',             help='Skip Track A (point cloud)')
    p.add_argument('--skip-detection',  action='store_true',             help='Skip Track B (deviation detection)')
    p.add_argument('--keep-frames',     action='store_true',             help='Keep extracted frames after run')
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def section(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")


def laplacian_score(image_path):
    """Sharpness score via Laplacian variance. Higher = sharper."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def encode_image_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION & FILTERING
# ═══════════════════════════════════════════════════════════════════════════════

def extract_frames(video_path, frames_dir, mode='fast'):
    """Extract frames from video at target fps based on mode."""
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / video_fps

    if mode == 'accurate':
        sample_fps = video_fps
    else:
        sample_fps = max(2.0, min(8.0, 60.0 / duration_s))

    frame_interval = max(1, int(round(video_fps / sample_fps)))

    print(f"  Video: {total_frames} frames @ {video_fps:.1f}fps ({duration_s:.1f}s)")
    print(f"  Mode '{mode}': sampling every {frame_interval} frames (~{video_fps/frame_interval:.1f}fps)")

    paths = []
    idx = saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            path = os.path.join(frames_dir, f"frame_{saved:04d}.png")
            cv2.imwrite(path, frame)
            paths.append(path)
            saved += 1
        idx += 1
    cap.release()

    print(f"  Extracted {len(paths)} frames")
    return paths


def filter_sharp_frames(frame_paths, threshold):
    """Remove blurry frames. Returns (sharp_paths, scores_dict)."""
    scores = {fp: laplacian_score(fp) for fp in frame_paths}
    sharp  = [fp for fp in frame_paths if scores[fp] >= threshold]
    blurry = [fp for fp in frame_paths if scores[fp] < threshold]

    print(f"  Sharpness filter (threshold={threshold}):")
    print(f"    Kept   : {len(sharp)} frames")
    print(f"    Removed: {len(blurry)} blurry frames")
    if scores:
        vals = list(scores.values())
        print(f"    Scores : min={min(vals):.0f}  median={float(np.median(vals)):.0f}  max={max(vals):.0f}")

    if not sharp:
        # Auto-relax threshold to keep at least 10% of frames
        fallback = sorted(scores.values(), reverse=True)[max(1, len(scores)//10) - 1]
        print(f"  WARNING: No sharp frames found. Auto-relaxing threshold to {fallback:.0f}")
        sharp = [fp for fp in frame_paths if scores[fp] >= fallback]

    # Delete blurry frames so run.py ignores them
    sharp_dir = os.path.join(os.path.dirname(frame_paths[0]), 'sharp')
    os.makedirs(sharp_dir, exist_ok=True)
    sharp_in_subdir = []
    for fp in sharp:
        dest = os.path.join(sharp_dir, os.path.basename(fp))
        shutil.copy2(fp, dest)
        sharp_in_subdir.append(dest)

    return sharp_in_subdir, scores


def select_keyframes(sharp_paths, scores, n_keyframes):
    """
    Select N best keyframes spread across the video timeline.
    Divides video into N equal segments, picks sharpest frame from each.
    This ensures coverage of the whole scene, not just the sharpest moment.
    """
    if len(sharp_paths) <= n_keyframes:
        return sharp_paths

    segment_size = len(sharp_paths) / n_keyframes
    keyframes = []
    for i in range(n_keyframes):
        start = int(i * segment_size)
        end   = int((i + 1) * segment_size)
        segment = sharp_paths[start:end]
        if segment:
            best = max(segment, key=lambda p: scores.get(p, 0))
            keyframes.append(best)

    print(f"  Selected {len(keyframes)} keyframes spread across {len(sharp_paths)} sharp frames")
    return keyframes


# ═══════════════════════════════════════════════════════════════════════════════
# DEPTH GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_depth_maps(frame_paths, depth_dir, encoder='vits'):
    """Run Depth Anything V2 on all frames at once."""
    os.makedirs(depth_dir, exist_ok=True)
    frames_dir = os.path.join(os.path.dirname(frame_paths[0]), 'sharp')

    result = subprocess.run([
    sys.executable, 'run.py',
    '--encoder', encoder,
    '--img-path', frames_dir,
    '--outdir', depth_dir,
    '--pred-only'
], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  WARNING: depth stderr: {result.stderr[-400:]}")

    depth_paths = {}
    for fp in frame_paths:
        base  = os.path.splitext(os.path.basename(fp))[0]
        dpath = os.path.join(depth_dir, base + '.png')
        if os.path.exists(dpath):
            depth_paths[fp] = dpath

    print(f"  Generated {len(depth_paths)}/{len(frame_paths)} depth maps")
    return depth_paths  # dict: frame_path → depth_path


# ═══════════════════════════════════════════════════════════════════════════════
# BIM MODEL RENDERING
# ═══════════════════════════════════════════════════════════════════════════════

def render_bim_wireframe(model_path, keyframe_path, depth_path, output_path, ceiling_height_ft=10.0):
    """
    Render the BIM model as a wireframe overlay approximating the keyframe viewpoint.
    Uses depth map to estimate camera distance, optical flow for rough orientation.
    Falls back to a clean front/side/top orthographic render if estimation fails.
    """
    try:
        import trimesh
    except ImportError:
        print("  WARNING: trimesh not installed. Run: pip install trimesh")
        print("  Falling back to BIM summary image")
        return _render_bim_fallback(model_path, output_path)

    try:
        scene = trimesh.load(model_path, force='scene')
        if hasattr(scene, 'dump'):
            meshes = scene.to_geometry()
        else:
            meshes = scene

        # Estimate camera distance from median depth
        depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if depth_map is not None:
            median_depth_norm = float(np.median(depth_map)) / 255.0
            est_distance      = median_depth_norm * (ceiling_height_ft * 0.3048)
        else:
            est_distance = ceiling_height_ft * 0.3048 * 0.5

        # Get scene bounds and center
        bounds = meshes.bounds  # [[min_x,y,z],[max_x,y,z]]
        center = meshes.centroid
        extent = meshes.extents
        scale  = max(extent)

        # Render 3 views: front, side, perspective — stitch them together
        img_h, img_w = cv2.imread(keyframe_path).shape[:2]
        panel_w      = img_w // 3

        renders = []
        for angle_deg in [0, 90, 45]:
            angle_rad = np.deg2rad(angle_deg)
            cam_offset = np.array([
                np.sin(angle_rad) * scale * 1.5,
                -scale * 0.3,
                np.cos(angle_rad) * scale * 1.5
            ])
            render = _render_mesh_view(meshes, center + cam_offset, center,
                                        panel_w, img_h)
            renders.append(render)

        combined = np.hstack(renders)
        cv2.imwrite(output_path, combined)
        return output_path

    except Exception as e:
        print(f"  WARNING: BIM render failed ({e}), using fallback")
        return _render_bim_fallback(model_path, output_path)


def _render_mesh_view(mesh, camera_pos, look_at, width, height):
    """Software rasterizer: project 3D edges onto 2D image plane."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 30  # dark background

    # Build view matrix
    forward = look_at - camera_pos
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    up      = np.array([0, 1, 0])
    right   = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up    = np.array([0, 0, 1])
        right = np.cross(forward, up)
    right   = right / (np.linalg.norm(right) + 1e-8)
    up      = np.cross(right, forward)

    R = np.stack([right, up, forward], axis=0)
    t = -R @ camera_pos

    # Project vertices
    verts = mesh.vertices
    verts_cam = (R @ verts.T).T + t

    # Frustum cull: keep points in front of camera
    in_front = verts_cam[:, 2] > 0.01
    if in_front.sum() < 3:
        return img

    # Perspective projection
    fov    = np.deg2rad(60)
    f      = (width / 2) / np.tan(fov / 2)
    cx, cy = width / 2, height / 2

    u = (verts_cam[:, 0] / (verts_cam[:, 2] + 1e-8)) * f + cx
    v = (verts_cam[:, 1] / (verts_cam[:, 2] + 1e-8)) * f + cy  # no flip needed (already in cam space)
    v = height - v  # flip Y

    # Draw edges (subsample for speed)
    edges = mesh.edges_unique
    step  = max(1, len(edges) // 8000)
    for i, j in edges[::step]:
        if not (in_front[i] and in_front[j]):
            continue
        x1, y1 = int(u[i]), int(v[i])
        x2, y2 = int(u[j]), int(v[j])
        if (0 <= x1 < width and 0 <= y1 < height and
                0 <= x2 < width and 0 <= y2 < height):
            cv2.line(img, (x1, y1), (x2, y2), (0, 200, 100), 1, cv2.LINE_AA)

    # Label
    label = f"BIM view {int(np.rad2deg(np.arctan2(camera_pos[0]-0, camera_pos[2]-0)))%360}°"
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    return img


def _render_bim_fallback(model_path, output_path):
    """If trimesh fails, generate a text card showing BIM model info."""
    img = np.ones((480, 640, 3), dtype=np.uint8) * 25
    cv2.putText(img, "BIM MODEL", (220, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 100), 2)
    cv2.putText(img, os.path.basename(model_path), (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    cv2.putText(img, "Install trimesh for visual render", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    cv2.imwrite(output_path, img)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI DEVIATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

DEVIATION_PROMPT = """You are an expert construction quality inspector AI with deep knowledge of BIM (Building Information Model) standards.

I am giving you TWO images:
1. LEFT or TOP: A wireframe render of the BIM model (the AS-DESIGNED intent) shown from multiple angles
2. RIGHT or BOTTOM: An actual photo from a body camera on a construction site (the AS-BUILT reality)

Your job is to identify construction deviations — things that appear to be built differently from the design.

Analyze carefully for:
- Structural elements in wrong positions (walls, columns, beams)
- Missing elements that should be present per the BIM
- Elements present that shouldn't be (additions not in design)
- Incorrect dimensions or proportions
- Wrong materials or finishes where identifiable
- Misaligned joints, connections, or openings

Be specific about WHERE in the image you see each issue (use quadrant descriptions: top-left, center, bottom-right etc.)
Be honest if the image quality, blur, or angle makes assessment difficult.

Respond ONLY in this exact JSON format:
{
  "overall_confidence": 0.0,
  "image_quality": "good|fair|poor",
  "quality_notes": "brief note on image quality issues",
  "deviations": [
    {
      "element": "element name/type",
      "severity": "critical|warning|info",
      "location_in_image": "description of where",
      "description": "what is wrong",
      "recommendation": "what should be done"
    }
  ],
  "elements_confirmed_ok": ["list of elements that look correctly placed"],
  "frame_summary": "one sentence summary of overall compliance"
}

If image quality is too poor to assess, still return the JSON with empty deviations and explain in quality_notes.
"""

ELEMENT_PROMPT = """You are an expert construction quality inspector AI.

I will show you a series of annotated construction site photos. Each photo has been analyzed for deviations from the BIM design.

Based on ALL the frame analyses below, produce a consolidated per-element report that answers:
- Which specific building elements have confirmed deviations across multiple frames?
- Which elements appear consistently compliant?
- What are the highest priority issues to address?

Frame analyses:
{frame_summaries}

Respond ONLY in this exact JSON format:
{
  "elements": [
    {
      "element": "element name",
      "status": "deviated|compliant|uncertain",
      "severity": "critical|warning|info|ok",
      "seen_in_frames": [0, 2, 5],
      "description": "consolidated description across all frames",
      "recommendation": "action required"
    }
  ],
  "priority_issues": ["top issue 1", "top issue 2", "top issue 3"],
  "overall_compliance_score": 0.0,
  "executive_summary": "2-3 sentence summary for site manager"
}
"""


def call_gemini(prompt, image_paths):
    try:
        from google import genai
        from google.genai import types
        import PIL.Image
    except ImportError:
        raise ImportError("Run: pip install google-genai pillow")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)

    contents = [prompt]
    for ip in image_paths:
        contents.append(PIL.Image.open(ip))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())

def analyze_frame(keyframe_path, bim_render_path, frame_idx):
    """Compare one keyframe against BIM render using Gemini."""
    # Stitch keyframe + BIM render side by side for the LLM
    kf  = cv2.imread(keyframe_path)
    bim = cv2.imread(bim_render_path)

    # Resize BIM render to match keyframe height
    h = kf.shape[0]
    scale = h / bim.shape[0]
    bim   = cv2.resize(bim, (int(bim.shape[1] * scale), h))

    # Add labels
    cv2.putText(kf,  "AS-DESIGNED (BIM)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
    cv2.putText(bim, "AS-BUILT (Site)",   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)

    # Divider
    divider  = np.ones((h, 4, 3), dtype=np.uint8) * 200
    stitched = np.hstack([bim, divider, kf])

    # Save stitched comparison
    tmp_path = keyframe_path.replace('.png', '_comparison.png')
    cv2.imwrite(tmp_path, stitched)

    try:
        result = call_gemini(DEVIATION_PROMPT, [tmp_path])
        result['frame_idx']      = frame_idx
        result['keyframe_path']  = keyframe_path
        result['comparison_path'] = tmp_path
        print(f"    Frame {frame_idx}: {len(result.get('deviations', []))} deviations "
              f"| quality={result.get('image_quality','?')} "
              f"| confidence={result.get('overall_confidence', 0):.0%}")
        return result
    except Exception as e:
        print(f"    Frame {frame_idx}: Gemini call failed — {e}")
        return {
            'frame_idx': frame_idx,
            'keyframe_path': keyframe_path,
            'error': str(e),
            'deviations': [],
            'image_quality': 'unknown'
        }


def consolidate_elements(frame_results):
    """Ask Gemini to consolidate all frame analyses into per-element report."""
    summaries = []
    for r in frame_results:
        if 'error' not in r:
            summaries.append({
                'frame': r['frame_idx'],
                'quality': r.get('image_quality'),
                'deviations': r.get('deviations', []),
                'ok_elements': r.get('elements_confirmed_ok', []),
                'summary': r.get('frame_summary', '')
            })

    if not summaries:
        return {'elements': [], 'priority_issues': [], 'overall_compliance_score': 0.0,
                'executive_summary': 'No frames could be analyzed.'}

    prompt = ELEMENT_PROMPT.format(frame_summaries=json.dumps(summaries, indent=2))

    try:
        return call_gemini(prompt, [])
    except Exception as e:
        print(f"  WARNING: Element consolidation failed — {e}")
        return {'elements': [], 'priority_issues': [], 'error': str(e),
                'executive_summary': 'Consolidation failed — see individual frame reports.'}


# ═══════════════════════════════════════════════════════════════════════════════
# ANNOTATION & REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

SEVERITY_COLOR = {
    'critical': (0, 0, 220),
    'warning':  (0, 165, 255),
    'info':     (0, 255, 255),
    'ok':       (50, 205, 50),
}


def annotate_keyframe(keyframe_path, frame_result, output_path):
    """Draw deviation annotations on a keyframe."""
    img = cv2.imread(keyframe_path)
    if img is None:
        return

    h, w = img.shape[:2]
    overlay = img.copy()

    deviations = frame_result.get('deviations', [])
    quality    = frame_result.get('image_quality', 'unknown')
    confidence = frame_result.get('overall_confidence', 0)

    # Header bar
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    status_color = (50, 205, 50) if not deviations else (0, 0, 220)
    label = f"Frame {frame_result['frame_idx']} | {len(deviations)} deviations | quality={quality} | conf={confidence:.0%}"
    cv2.putText(overlay, label, (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1, cv2.LINE_AA)

    # Deviation list on right side panel
    panel_w = 320
    panel   = np.ones((h, panel_w, 3), dtype=np.uint8) * 18

    y = 60
    cv2.putText(panel, "DEVIATIONS DETECTED", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    for i, dev in enumerate(deviations):
        color = SEVERITY_COLOR.get(dev.get('severity', 'info'), (200, 200, 200))

        # Severity badge
        cv2.rectangle(panel, (8, y - 14), (90, y + 4), color, -1)
        cv2.putText(panel, dev.get('severity', '?').upper(), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 1)

        # Element name
        cv2.putText(panel, dev.get('element', 'Unknown')[:35], (95, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        y += 20

        # Description (word wrap)
        desc = dev.get('description', '')
        for chunk in [desc[j:j+42] for j in range(0, min(len(desc), 84), 42)]:
            cv2.putText(panel, chunk, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
            y += 16

        # Location
        loc = dev.get('location_in_image', '')
        if loc:
            cv2.putText(panel, f"↳ {loc[:40]}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 100), 1)
            y += 16

        y += 8
        if y > h - 60:
            cv2.putText(panel, f"... +{len(deviations) - i - 1} more", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
            break

    # OK elements
    ok_elems = frame_result.get('elements_confirmed_ok', [])
    if ok_elems and y < h - 80:
        y += 10
        cv2.putText(panel, "CONFIRMED OK:", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 205, 50), 1)
        y += 18
        for el in ok_elems[:4]:
            cv2.putText(panel, f"✓ {el[:38]}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 180, 50), 1)
            y += 16

    # Summary at bottom of panel
    summary = frame_result.get('frame_summary', '')
    if summary:
        summary_y = h - 50
        for chunk in [summary[j:j+40] for j in range(0, min(len(summary), 80), 40)]:
            cv2.putText(panel, chunk, (8, summary_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1)
            summary_y += 16

    # Blend and combine
    img = cv2.addWeighted(overlay, 1.0, img, 0.0, 0)
    output = np.hstack([img, panel])
    cv2.imwrite(output_path, output)


def generate_summary_report(frame_results, element_report, outdir):
    """Generate HTML summary report."""
    severity_emoji = {'critical': '🔴', 'warning': '🟡', 'info': '🔵', 'ok': '🟢'}
    compliance     = element_report.get('overall_compliance_score', 0)
    summary        = element_report.get('executive_summary', '')
    priority       = element_report.get('priority_issues', [])
    elements       = element_report.get('elements', [])

    frame_rows = ""
    for r in frame_results:
        devs = r.get('deviations', [])
        n    = len(devs)
        q    = r.get('image_quality', '?')
        c    = r.get('overall_confidence', 0)
        img  = os.path.basename(r.get('comparison_path', ''))
        frame_rows += f"""
        <tr>
          <td>Frame {r['frame_idx']}</td>
          <td>{n} deviation{'s' if n != 1 else ''}</td>
          <td>{q}</td>
          <td>{c:.0%}</td>
          <td><a href="{img}">View</a></td>
        </tr>"""

    element_rows = ""
    for el in elements:
        sev   = el.get('severity', 'info')
        emoji = severity_emoji.get(sev, '⚪')
        frames_seen = ', '.join(str(f) for f in el.get('seen_in_frames', []))
        element_rows += f"""
        <tr class="{sev}">
          <td>{emoji} {el.get('element', '')}</td>
          <td>{el.get('status', '').upper()}</td>
          <td>{sev.upper()}</td>
          <td>{el.get('description', '')}</td>
          <td>{el.get('recommendation', '')}</td>
          <td>{frames_seen}</td>
        </tr>"""

    priority_html = ''.join(f"<li>{p}</li>" for p in priority)
    compliance_color = '#22c55e' if compliance > 0.8 else '#f59e0b' if compliance > 0.5 else '#ef4444'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IronSLAM Deviation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ font-size: 2rem; color: #38bdf8; margin-bottom: 0.25rem; }}
  .subtitle {{ color: #64748b; margin-bottom: 2rem; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; }}
  .score {{ font-size: 3.5rem; font-weight: bold; color: {compliance_color}; }}
  .score-label {{ color: #94a3b8; font-size: 0.9rem; }}
  h2 {{ color: #38bdf8; font-size: 1.2rem; margin-bottom: 1rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #0f172a; color: #94a3b8; padding: 0.6rem 0.8rem; text-align: left; }}
  td {{ padding: 0.6rem 0.8rem; border-bottom: 1px solid #1e293b; }}
  tr.critical td {{ border-left: 3px solid #ef4444; }}
  tr.warning td {{ border-left: 3px solid #f59e0b; }}
  tr.info td {{ border-left: 3px solid #38bdf8; }}
  tr.ok td {{ border-left: 3px solid #22c55e; }}
  tr:hover td {{ background: #243044; }}
  .priority {{ background: #0f172a; border-radius: 8px; padding: 1rem; }}
  .priority li {{ padding: 0.4rem 0; color: #fbbf24; }}
  a {{ color: #38bdf8; }}
  .summary {{ color: #cbd5e1; line-height: 1.6; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
</style>
</head>
<body>
<h1>⚡ IronSLAM Deviation Report</h1>
<p class="subtitle">Generated {time.strftime('%Y-%m-%d %H:%M')} · {len(frame_results)} frames analyzed</p>

<div class="grid">
  <div class="card">
    <div class="score">{compliance:.0%}</div>
    <div class="score-label">Overall Compliance Score</div>
    <p class="summary" style="margin-top:1rem">{summary}</p>
  </div>
  <div class="card">
    <h2>🚨 Priority Issues</h2>
    <div class="priority"><ol>{priority_html}</ol></div>
  </div>
</div>

<div class="card">
  <h2>📋 Per-Element Analysis</h2>
  <table>
    <tr><th>Element</th><th>Status</th><th>Severity</th><th>Description</th><th>Recommendation</th><th>Frames</th></tr>
    {element_rows}
  </table>
</div>

<div class="card">
  <h2>🎥 Per-Frame Analysis</h2>
  <table>
    <tr><th>Frame</th><th>Deviations</th><th>Image Quality</th><th>Confidence</th><th>Comparison</th></tr>
    {frame_rows}
  </table>
</div>
</body>
</html>"""

    report_path = os.path.join(outdir, 'report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    return report_path


# ═══════════════════════════════════════════════════════════════════════════════
# POINT CLOUD (TRACK A)
# ═══════════════════════════════════════════════════════════════════════════════

def frame_to_pointcloud(image_path, depth_path, ceiling_height_ft=10.0, downsample=3):
    image     = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if image is None or depth_map is None:
        return None, None

    h, w = image.shape[:2]
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fov_h_rad = np.deg2rad(60.0)
    fx = (w / 2.0) / np.tan(fov_h_rad / 2.0)
    cx, cy = w / 2.0, h / 2.0

    depth_norm = depth_map.astype(np.float32) / 255.0
    depth_m    = depth_norm * (ceiling_height_ft * 0.3048)

    u = np.arange(0, w, downsample)
    v = np.arange(0, h, downsample)
    uu, vv = np.meshgrid(u, v)

    mask = depth_norm[vv, uu] > 0.05
    uu, vv = uu[mask], vv[mask]
    d = depth_m[vv, uu]

    X = (uu - cx) * d / fx
    Y = -(vv - cy) * d / fx
    Z = d

    points = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    colors = image_rgb[vv, uu].astype(np.uint8)
    return points, colors


def voxel_downsample(points, colors, voxel_size=0.02):
    if len(points) == 0:
        return points, colors
    voxel_idx = np.floor(points / voxel_size).astype(np.int32)
    seen, keep = {}, []
    for i, vi in enumerate(map(tuple, voxel_idx)):
        if vi not in seen:
            seen[vi] = i
            keep.append(i)
    keep = np.array(keep)
    return points[keep], colors[keep]


def write_ply(filepath, points, colors):
    n = len(points)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {n}\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        xyz = points.astype(np.float32)
        rgb = colors.astype(np.uint8)
        for i in range(n):
            f.write(xyz[i].tobytes())
            f.write(rgb[i].tobytes())


def build_pointcloud(frame_paths, depth_paths, ply_path, ceiling_height_ft, downsample):
    """Simple fused point cloud — no ICP (fast, good enough for visualization)."""
    all_pts, all_col = [], []

    for i, fp in enumerate(frame_paths):
        dp = depth_paths.get(fp)
        if not dp:
            continue
        pts, col = frame_to_pointcloud(fp, dp, ceiling_height_ft, downsample)
        if pts is not None:
            # Offset each frame slightly on X axis to spread views apart
            pts[:, 0] += i * 0.5
            all_pts.append(pts)
            all_col.append(col)

    if not all_pts:
        print("  WARNING: No frames produced point clouds")
        return

    merged_pts = np.vstack(all_pts)
    merged_col = np.vstack(all_col)
    merged_pts, merged_col = voxel_downsample(merged_pts, merged_col, voxel_size=0.025)

    write_ply(ply_path, merged_pts, merged_col)
    size_mb = os.path.getsize(ply_path) / 1e6
    print(f"  Point cloud: {len(merged_pts):,} points → {ply_path} ({size_mb:.1f} MB)")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}"); sys.exit(1)
    if args.video and not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}"); sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)
    work_dir   = os.path.join(args.outdir, '_work')
    frames_dir = os.path.join(work_dir, 'frames')
    depth_dir  = os.path.join(work_dir, 'depth')
    bim_dir    = os.path.join(work_dir, 'bim_renders')
    annot_dir  = os.path.join(args.outdir, 'annotated_frames')

    for d in [frames_dir, depth_dir, bim_dir, annot_dir]:
        os.makedirs(d, exist_ok=True)

    if args.video:
        video_base = os.path.splitext(os.path.basename(args.video))[0]
    else:
        video_base = os.path.splitext(os.path.basename(args.image))[0]
    t0 = time.time()

    

    # ── 1. Extract frames ────────────────────────────────────────────────────
        # Validate inputs
    if not args.image and not args.video:
        print("ERROR: provide --video or --image"); sys.exit(1)
    if args.image and not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}"); sys.exit(1)
    if args.video and not os.path.exists(args.video):
        print(f"ERROR: Video not found: {args.video}"); sys.exit(1)

# ── Single image mode ────────────────────────────────────────────────────
    if args.image:
        section("Single image mode — skipping point cloud")
        sharp_dir = os.path.join(frames_dir, 'sharp')
        os.makedirs(sharp_dir, exist_ok=True)
        dest = os.path.join(sharp_dir, 'frame_0000.png')
        shutil.copy2(args.image, dest)
        sharp_paths = [dest]
        scores      = {dest: laplacian_score(dest)}
        print(f"  Sharpness score: {list(scores.values())[0]:.0f} (no filtering in single image mode)")
        args.skip_pointcloud = True
        args.keyframes = 1

# ── Video mode ───────────────────────────────────────────────────────────
    else:
        section("Step 1/5: Extracting & filtering frames")
        frame_paths = extract_frames(args.video, frames_dir, mode=args.mode)
        sharp_paths, scores = filter_sharp_frames(frame_paths, args.blur_threshold)
        if not sharp_paths:
            print("ERROR: No sharp frames found. Try lowering --blur-threshold")
            sys.exit(1)

    # ── 2. Generate depth maps ───────────────────────────────────────────────
    section("Step 2/5: Generating depth maps")
    depth_paths = generate_depth_maps(sharp_paths, depth_dir, encoder=args.encoder)

    # ── 3. Track A: Point cloud ──────────────────────────────────────────────
    if not args.skip_pointcloud:
        section("Step 3/5: Building point cloud (Track A — visual)")
        ply_path = os.path.join(args.outdir, f'{video_base}_pointcloud.ply')
        build_pointcloud(sharp_paths, depth_paths, ply_path, args.ceiling_height, args.downsample)
    else:
        print("\n  (Skipping point cloud — Track A)")

    # ── 4. Track B: Deviation detection ─────────────────────────────────────
    if not args.skip_detection:
        section("Step 4/5: Deviation detection (Track B — intelligence)")

        # Select keyframes spread across the timeline
        keyframes = select_keyframes(sharp_paths, scores, args.keyframes)
        print(f"  Analyzing {len(keyframes)} keyframes against BIM model...")

        frame_results = []
        for i, kf in enumerate(keyframes):
            print(f"\n  Keyframe {i+1}/{len(keyframes)}: {os.path.basename(kf)}")
            dp = depth_paths.get(kf)

            # Render BIM from approximate viewpoint
            bim_render = os.path.join(bim_dir, f'bim_{i:03d}.png')
            render_bim_wireframe(args.model, kf, dp or kf, bim_render, args.ceiling_height)

            # Gemini comparison
            result = analyze_frame(kf, bim_render, i)
            frame_results.append(result)

            # Annotate keyframe
            annot_path = os.path.join(annot_dir, f'frame_{i:03d}_annotated.png')
            annotate_keyframe(kf, result, annot_path)

        # ── 5. Consolidate + report ──────────────────────────────────────────
        section("Step 5/5: Generating reports")
        print("  Consolidating per-element analysis...")
        element_report = consolidate_elements(frame_results)

        # Save raw JSON
        json_path = os.path.join(args.outdir, 'deviation_report.json')
        with open(json_path, 'w') as f:
            json.dump({'frames': frame_results, 'elements': element_report}, f, indent=2)

        # Save HTML report
        report_path = generate_summary_report(frame_results, element_report, args.outdir)

        # Print priority issues to terminal
        priority = element_report.get('priority_issues', [])
        if priority:
            print("\n  🚨 TOP PRIORITY ISSUES:")
            for i, p in enumerate(priority, 1):
                print(f"    {i}. {p}")

        compliance = element_report.get('overall_compliance_score', 0)
        print(f"\n  Overall compliance: {compliance:.0%}")
        print(f"  Executive summary: {element_report.get('executive_summary', '')}")

    else:
        print("\n  (Skipping deviation detection — Track B)")
        report_path = None
        json_path   = None

    # ── Cleanup ──────────────────────────────────────────────────────────────
    if not args.keep_frames:
        shutil.rmtree(work_dir, ignore_errors=True)

    elapsed = time.time() - t0

    # ── Final summary ────────────────────────────────────────────────────────
    section(f"Pipeline complete in {elapsed:.0f}s")
    print(f"  Output directory: {args.outdir}/\n")
    if not args.skip_pointcloud:
        print(f"  📦 {video_base}_pointcloud.ply   → open in MeshLab / CloudCompare")
    if not args.skip_detection:
        print(f"  🖼  annotated_frames/            → per-keyframe annotations")
        print(f"  📊 report.html                  → open in browser")
        print(f"  📄 deviation_report.json        → raw data")
    print()


if __name__ == '__main__':
    main()