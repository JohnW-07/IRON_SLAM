"""
parse_model.py
Extracts named objects and their 3D bounding boxes from OBJ or GLTF files.
Outputs as_designed.json with normalized, grouped object data.
"""

import json
import os
import re
import numpy as np


def parse_and_save(model_path, output_path):
    ext = os.path.splitext(model_path)[1].lower()

    if ext in ('.gltf', '.glb'):
        objects = parse_gltf(model_path)
    elif ext == '.obj':
        objects = parse_obj(model_path)
    else:
        raise ValueError(f"Unsupported model format: {ext}. Use .obj or .gltf/.glb")

    # Normalize + group similar names (e.g. "Duct_001", "Duct_002" → "Duct")
    grouped = group_objects(objects)

    with open(output_path, 'w') as f:
        json.dump(grouped, f, indent=2)

    print(f"  Parsed {len(objects)} objects → {len(grouped)} groups")
    print(f"  Groups: {', '.join(list(grouped.keys())[:10])}{'...' if len(grouped) > 10 else ''}")
    return grouped


def parse_obj(filepath):
    """Parse OBJ file, extracting named groups with vertex bounding boxes."""
    objects = {}
    current_name = "default"
    vertices_by_object = {}

    # First pass: collect all vertices globally (OBJ uses global vertex indices)
    all_vertices = []
    face_groups = {}
    current_group = "default"

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == 'v':
                try:
                    all_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except (ValueError, IndexError):
                    pass

            elif parts[0] in ('g', 'o'):
                # New group/object
                current_group = ' '.join(parts[1:]) if len(parts) > 1 else 'unnamed'
                if current_group not in face_groups:
                    face_groups[current_group] = []

            elif parts[0] == 'f':
                if current_group not in face_groups:
                    face_groups[current_group] = []
                # Parse face vertex indices (handle v, v/vt, v/vt/vn formats)
                for token in parts[1:]:
                    idx = int(token.split('/')[0]) - 1  # OBJ is 1-indexed
                    face_groups[current_group].append(idx)

    all_vertices = np.array(all_vertices) if all_vertices else np.zeros((1, 3))

    objects = {}
    for name, indices in face_groups.items():
        if not indices:
            continue
        # Clamp indices to valid range
        valid_indices = [i for i in indices if 0 <= i < len(all_vertices)]
        if not valid_indices:
            continue
        verts = all_vertices[valid_indices]
        objects[name] = bbox_from_vertices(verts)

    return objects


def parse_gltf(filepath):
    """Parse GLTF/GLB file using pygltflib if available, else fallback JSON parse."""
    try:
        import pygltflib
        return _parse_gltf_pygltflib(filepath)
    except ImportError:
        pass

    # Fallback: parse .gltf (JSON) directly
    if filepath.endswith('.glb'):
        print("  WARNING: pygltflib not installed. GLB parsing limited. Run: pip install pygltflib")
        return {"scene": {"center": [0, 0, 0], "min": [0, 0, 0], "max": [1, 1, 1], "dimensions": [1, 1, 1]}}

    with open(filepath, 'r') as f:
        data = json.load(f)

    objects = {}
    meshes = data.get('meshes', [])
    nodes = data.get('nodes', [])

    # Build node name → mesh index map
    for node in nodes:
        if 'mesh' in node:
            name = node.get('name', f"node_{node['mesh']}")
            mesh = meshes[node['mesh']]
            # We can't decode buffer data without pygltflib, so use node translation as center
            translation = node.get('translation', [0, 0, 0])
            scale = node.get('scale', [1, 1, 1])
            objects[name] = {
                "center": translation,
                "min": [translation[i] - scale[i] for i in range(3)],
                "max": [translation[i] + scale[i] for i in range(3)],
                "dimensions": [scale[i] * 2 for i in range(3)]
            }

    return objects


def _parse_gltf_pygltflib(filepath):
    import pygltflib
    import struct

    gltf = pygltflib.GLTF2().load(filepath)
    objects = {}

    def get_accessor_data(accessor_idx):
        accessor = gltf.accessors[accessor_idx]
        buffer_view = gltf.bufferViews[accessor.bufferView]
        buffer = gltf.buffers[buffer_view.buffer]
        data = gltf.get_data_from_buffer_uri(buffer.uri) if hasattr(buffer, 'uri') and buffer.uri else bytes(gltf._glb_data)
        start = buffer_view.byteOffset + (accessor.byteOffset or 0)
        count = accessor.count
        # VEC3 float
        fmt = f'{count * 3}f'
        raw = struct.unpack_from(fmt, data, start)
        return np.array(raw).reshape(-1, 3)

    for node in (gltf.nodes or []):
        if node.mesh is None:
            continue
        name = node.name or f"node_{node.mesh}"
        mesh = gltf.meshes[node.mesh]
        all_verts = []
        for prim in mesh.primitives:
            if prim.attributes.POSITION is not None:
                try:
                    verts = get_accessor_data(prim.attributes.POSITION)
                    all_verts.append(verts)
                except Exception:
                    pass
        if all_verts:
            combined = np.vstack(all_verts)
            objects[name] = bbox_from_vertices(combined)

    return objects


def bbox_from_vertices(verts):
    """Compute bounding box dict from numpy vertex array."""
    mn = verts.min(axis=0).tolist()
    mx = verts.max(axis=0).tolist()
    center = ((verts.min(axis=0) + verts.max(axis=0)) / 2).tolist()
    dims = (verts.max(axis=0) - verts.min(axis=0)).tolist()
    return {
        "center": [round(v, 4) for v in center],
        "min":    [round(v, 4) for v in mn],
        "max":    [round(v, 4) for v in mx],
        "dimensions": [round(v, 4) for v in dims]
    }


def group_objects(objects):
    """
    Group objects with similar names.
    e.g. "HVAC_Duct_001", "HVAC_Duct_002" → "HVAC_Duct" with list of members + combined bbox.
    """
    groups = {}

    for name, bbox in objects.items():
        # Strip trailing numbers/underscores to get base category
        base = re.sub(r'[_\-\.\s]*\d+$', '', name).strip()
        base = base if base else name

        if base not in groups:
            groups[base] = {
                "members": [],
                "combined_bbox": None
            }

        groups[base]["members"].append({
            "name": name,
            "bbox": bbox
        })

    # Compute combined bbox per group
    for base, group in groups.items():
        all_mins = np.array([m["bbox"]["min"] for m in group["members"]])
        all_maxs = np.array([m["bbox"]["max"] for m in group["members"]])
        combined_min = all_mins.min(axis=0).tolist()
        combined_max = all_maxs.max(axis=0).tolist()
        center = ((np.array(combined_min) + np.array(combined_max)) / 2).tolist()
        dims = (np.array(combined_max) - np.array(combined_min)).tolist()
        groups[base]["combined_bbox"] = {
            "center": [round(v, 4) for v in center],
            "min":    [round(v, 4) for v in combined_min],
            "max":    [round(v, 4) for v in combined_max],
            "dimensions": [round(v, 4) for v in dims],
            "count": len(group["members"])
        }

    return groups


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 3:
        print("Usage: python parse_model.py model.obj output.json")
        sys.exit(1)
    parse_and_save(sys.argv[1], sys.argv[2])
    print(f"Saved to {sys.argv[2]}")