"""
annotate.py

1. Converts pixel bounding boxes → 3D coordinates using the depth map
2. Compares As-Built 3D coords against As-Designed 3D coords
3. Draws annotated output image with color-coded discrepancy labels
"""

import json
import os
import cv2
import numpy as np


# ── Color scheme ─────────────────────────────────────────────────────────────
COLOR_OK         = (50,  205, 50)    # Green  — within tolerance
COLOR_WARNING    = (0,   165, 255)   # Orange — minor discrepancy
COLOR_CRITICAL   = (0,   0,   220)   # Red    — major discrepancy
COLOR_NOT_FOUND  = (180, 180, 180)   # Gray   — not detected in image
COLOR_TEXT_BG    = (20,  20,  20)    # Near-black label background
TOLERANCE_M      = 0.15              # 15cm tolerance = "OK"
WARNING_M        = 0.40              # 40cm = "WARNING", above = "CRITICAL"


def compare_and_draw(image_path, depth_map_path, as_designed_path, as_built_path, out_path, ceiling_height_ft=10.0):
    image     = cv2.imread(image_path)
    depth_map = cv2.imread(depth_map_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if depth_map is None:
        raise FileNotFoundError(f"Could not read depth map: {depth_map_path}")

    h, w = image.shape[:2]

    # Resize depth map to match image if needed
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    with open(as_designed_path, 'r') as f:
        as_designed = json.load(f)
    with open(as_built_path, 'r') as f:
        as_built = json.load(f)

    # ── Convert model scale → real world ─────────────────────────────────────
    # Use the overall scene height from as_designed to calibrate depth scale
    scale_factor = compute_scale_factor(as_designed, ceiling_height_ft)

    discrepancies = []

    for category, designed_data in as_designed.items():
        built_data = as_built.get(category)

        if built_data is None or not built_data.get("visible"):
            discrepancies.append({
                "category": category,
                "status": "not_detected",
                "message": "Not visible in photo",
                "bbox": None,
                "delta_m": None
            })
            continue

        bbox = built_data.get("bbox_pixels")
        if not bbox or len(bbox) < 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Clamp to image bounds
        cx = max(0, min(cx, w - 1))
        cy = max(0, min(cy, h - 1))

        # Get depth at center pixel
        depth_val = float(depth_map[cy, cx])

        # Convert to 3D (simple pinhole approximation)
        built_3d = pixel_to_3d(cx, cy, depth_val, w, h, ceiling_height_ft)

        # Compare against designed center
        designed_center = designed_data["combined_bbox"]["center"]
        designed_3d_normalized = normalize_designed_coords(designed_center, as_designed, ceiling_height_ft)

        delta = compute_delta(built_3d, designed_3d_normalized)

        if delta < TOLERANCE_M:
            status = "ok"
        elif delta < WARNING_M:
            status = "warning"
        else:
            status = "critical"

        discrepancies.append({
            "category": category,
            "status": status,
            "bbox": [x1, y1, x2, y2],
            "built_3d": built_3d,
            "designed_3d": designed_3d_normalized,
            "delta_m": round(delta, 3),
            "message": format_message(category, delta, built_3d, designed_3d_normalized),
            "confidence": built_data.get("confidence", 0.0)
        })

    # ── Draw annotations ──────────────────────────────────────────────────────
    annotated = draw_annotations(image.copy(), discrepancies, w, h)

    # ── Draw legend ───────────────────────────────────────────────────────────
    annotated = draw_legend(annotated, discrepancies)

    cv2.imwrite(out_path, annotated)

    # Save discrepancy report JSON alongside
    report_path = os.path.splitext(out_path)[0] + '_report.json'
    with open(report_path, 'w') as f:
        json.dump(discrepancies, f, indent=2)

    ok_count       = sum(1 for d in discrepancies if d["status"] == "ok")
    warn_count     = sum(1 for d in discrepancies if d["status"] == "warning")
    critical_count = sum(1 for d in discrepancies if d["status"] == "critical")
    nd_count       = sum(1 for d in discrepancies if d["status"] == "not_detected")

    print(f"  Results: {ok_count} OK | {warn_count} warnings | {critical_count} critical | {nd_count} not detected")
    print(f"  Report:  {report_path}")

    return discrepancies


# ── Geometry helpers ──────────────────────────────────────────────────────────

def pixel_to_3d(px, py, depth_val, img_w, img_h, ceiling_height_ft):
    """Convert pixel + depth value to approximate 3D coordinate in feet."""
    ceiling_height_m = ceiling_height_ft * 0.3048
    z = (depth_val / 255.0) * ceiling_height_m

    # Normalize x/y to [-1, 1] then scale by depth
    norm_x = (px - img_w / 2) / (img_w / 2)
    norm_y = (py - img_h / 2) / (img_h / 2)

    fov_scale = 0.7  # approximate horizontal FOV factor
    x = norm_x * z * fov_scale
    y = -norm_y * z * fov_scale  # flip y (image y is downward)

    return {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)}


def compute_scale_factor(as_designed, ceiling_height_ft):
    """Estimate model → real world scale using scene bounding box Y range."""
    all_mins = [v["combined_bbox"]["min"][1] for v in as_designed.values()]
    all_maxs = [v["combined_bbox"]["max"][1] for v in as_designed.values()]
    model_height = max(all_maxs) - min(all_mins)
    if model_height == 0:
        return 1.0
    real_height_m = ceiling_height_ft * 0.3048
    return real_height_m / model_height


def normalize_designed_coords(center, as_designed, ceiling_height_ft):
    """Normalize model coordinates to real-world meters."""
    scale = compute_scale_factor(as_designed, ceiling_height_ft)
    return {
        "x": round(center[0] * scale, 3),
        "y": round(center[2] * scale, 3),  # Z in model → depth (Y in our 3D)
        "z": round(center[1] * scale, 3)   # Y in model → height (Z in our 3D)
    }


def compute_delta(built_3d, designed_3d):
    """Euclidean distance between as-built and as-designed in meters."""
    dx = built_3d["x"] - designed_3d["x"]
    dy = built_3d["y"] - designed_3d["y"]
    dz = built_3d["z"] - designed_3d["z"]
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def format_message(category, delta_m, built, designed):
    dz = built["z"] - designed["z"]
    dx = built["x"] - designed["x"]
    parts = []
    if abs(dz) > TOLERANCE_M:
        direction = "too high" if dz > 0 else "too low"
        parts.append(f"{abs(dz):.2f}m {direction}")
    if abs(dx) > TOLERANCE_M:
        direction = "left" if dx < 0 else "right"
        parts.append(f"{abs(dx):.2f}m {direction}")
    if not parts:
        parts.append(f"Δ{delta_m:.2f}m")
    return f"{category}: {', '.join(parts)}"


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_annotations(image, discrepancies, img_w, img_h):
    overlay = image.copy()

    for disc in discrepancies:
        status = disc["status"]
        bbox   = disc.get("bbox")

        if status == "not_detected" or bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = {"ok": COLOR_OK, "warning": COLOR_WARNING, "critical": COLOR_CRITICAL}.get(status, COLOR_NOT_FOUND)

        # Draw semi-transparent fill
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

    # Blend overlay
    image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)

    for disc in discrepancies:
        status = disc["status"]
        bbox   = disc.get("bbox")

        if status == "not_detected" or bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        color = {"ok": COLOR_OK, "warning": COLOR_WARNING, "critical": COLOR_CRITICAL}.get(status, COLOR_NOT_FOUND)

        # Draw bounding box border
        thickness = 3 if status == "critical" else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Draw corner accents
        corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(image, (cx, cy), (cx + dx * corner_len, cy), color, 3)
            cv2.line(image, (cx, cy), (cx, cy + dy * corner_len), color, 3)

        # Draw label
        label    = disc["message"]
        delta_m  = disc.get("delta_m")
        conf     = disc.get("confidence", 0)
        sub_label = f"Δ {delta_m:.2f}m | conf {conf:.0%}" if delta_m is not None else ""

        draw_label(image, label, sub_label, x1, y1, color)

    return image


def draw_label(image, text, subtext, x, y, color):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    padding    = 6

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    (sw, sh), _ = cv2.getTextSize(subtext, font, font_scale * 0.8, thickness) if subtext else ((0, 0), 0)

    box_w = max(tw, sw) + padding * 2
    box_h = th + (sh + 4 if subtext else 0) + padding * 2

    # Position above bbox, clamp to image edges
    lx = max(0, x)
    ly = max(box_h, y - 4)

    # Background
    cv2.rectangle(image, (lx, ly - box_h), (lx + box_w, ly), COLOR_TEXT_BG, -1)
    cv2.rectangle(image, (lx, ly - box_h), (lx + box_w, ly), color, 1)

    # Main text
    cv2.putText(image, text, (lx + padding, ly - padding - (sh + 4 if subtext else 0)),
                font, font_scale, color, thickness, cv2.LINE_AA)

    # Sub text
    if subtext:
        cv2.putText(image, subtext, (lx + padding, ly - padding),
                    font, font_scale * 0.8, (200, 200, 200), thickness, cv2.LINE_AA)


def draw_legend(image, discrepancies, margin=20):
    h, w = image.shape[:2]

    ok_n   = sum(1 for d in discrepancies if d["status"] == "ok")
    wa_n   = sum(1 for d in discrepancies if d["status"] == "warning")
    cr_n   = sum(1 for d in discrepancies if d["status"] == "critical")
    nd_n   = sum(1 for d in discrepancies if d["status"] == "not_detected")

    entries = [
        (COLOR_OK,        f"Within tolerance ({ok_n})"),
        (COLOR_WARNING,   f"Minor offset <{WARNING_M:.0f}cm ({wa_n})"),
        (COLOR_CRITICAL,  f"Major offset ≥{WARNING_M*100:.0f}cm ({cr_n})"),
        (COLOR_NOT_FOUND, f"Not detected ({nd_n})"),
    ]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    lh         = 24
    pad        = 10
    box_w      = 260
    box_h      = len(entries) * lh + pad * 2 + 20

    lx = w - box_w - margin
    ly = margin

    # Legend background
    cv2.rectangle(image, (lx - pad, ly), (lx + box_w, ly + box_h), COLOR_TEXT_BG, -1)
    cv2.rectangle(image, (lx - pad, ly), (lx + box_w, ly + box_h), (80, 80, 80), 1)

    cv2.putText(image, "AS-BUILT vs AS-DESIGNED", (lx, ly + 16),
                font, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    for i, (color, label) in enumerate(entries):
        ey = ly + 30 + i * lh
        cv2.rectangle(image, (lx, ey - 10), (lx + 14, ey + 2), color, -1)
        cv2.putText(image, label, (lx + 20, ey), font, font_scale, (220, 220, 220), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 6:
        print("Usage: python annotate.py image.jpg depth.png as_designed.json as_built.json out.png")
        sys.exit(1)
    compare_and_draw(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])