#!/usr/bin/env python3
"""
SMPL-X mesh rendering utilities.

Renders per-person (amodal) part label maps and depth maps from AGORA's
gt_verts and the SMPL-X model faces.

Method:
    Back-to-front painter's algorithm (faces sorted by mean depth). At teacher
    resolution the SMPL-X triangles (20908 faces) are mostly sub-pixel to a few
    pixels, so treating the depth as constant per face is practically
    equivalent to a z-buffer. Checked against a true z-buffer implementation:
    the silhouette IoU with the AGORA masks is 0.79-0.86, the gap being
    clothing and hair rather than rendering error.

Coordinates:
    verts are in camera coordinates (metres, z = depth). Pinhole projection
    u = f*x/z + CX, v = f*y/z + CY (4K reference), scaled by resolution_config.
    The output maps cover an ROI (the person's projected extent, including
    outside the image) in image coordinates of the current resolution, with a
    downscale factor `scale`.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    CX,
    CY,
    PART_GROUPS,
    PART_GROUP_IDS,
    RENDER_MAX_SIDE,
    SMPLX_MODEL_PATH,
    SMPLX_VERT_SEG_PATH,
    resolution_config,
)

# ============================================================
# Lazy asset loading (once per worker process)
# ============================================================

_ASSETS = None  # (faces (F,3) int32, face_part (F,) uint8)


def load_smplx_assets() -> tuple[np.ndarray, np.ndarray]:
    """Load the SMPL-X faces and per-face part labels (cached)

    Returns:
        faces: (F, 3) int32 vertex indices
        face_part: (F,) uint8 part labels (PART_GROUP_IDS: 1=head, 2=torso, 3=limbs, 4=hands)
    """
    global _ASSETS
    if _ASSETS is not None:
        return _ASSETS

    if not SMPLX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"SMPL-X model not found: {SMPLX_MODEL_PATH}\n"
            "Place SMPLX_NEUTRAL.npz (from the SMPL-X model release) in weights/smplx/"
        )
    if not SMPLX_VERT_SEG_PATH.exists():
        raise FileNotFoundError(
            f"SMPL-X vertex segmentation not found: {SMPLX_VERT_SEG_PATH}"
        )

    model = np.load(SMPLX_MODEL_PATH, allow_pickle=True)
    faces = model["f"].astype(np.int64)  # (20908, 3)
    v_template = model["v_template"]  # (10475, 3)
    n_verts = len(v_template)

    with open(SMPLX_VERT_SEG_PATH, "r", encoding="utf-8") as f:
        seg = json.load(f)

    # Vertex -> part label
    vert_part = np.zeros(n_verts, dtype=np.int32)
    for group_name, part_names in PART_GROUPS.items():
        gid = PART_GROUP_IDS[group_name]
        for part_name in part_names:
            vert_part[np.asarray(seg[part_name], dtype=np.int64)] = gid

    # Vertices missing from the segmentation JSON (about 6) take the nearest label
    unassigned = np.where(vert_part == 0)[0]
    if len(unassigned) > 0:
        assigned = np.where(vert_part > 0)[0]
        for ui in unassigned:
            d = np.linalg.norm(v_template[assigned] - v_template[ui], axis=1)
            vert_part[ui] = vert_part[assigned[np.argmin(d)]]

    # Face label = majority of its 3 vertices (vectorized: a value shared by at
    # least two vertices wins; if all differ, the first one)
    fp = vert_part[faces]  # (F, 3)
    face_part = fp[:, 0].copy()
    mask_12 = fp[:, 1] == fp[:, 2]
    face_part[mask_12] = fp[mask_12, 1]
    # fp[:,0]==fp[:,1] or fp[:,0]==fp[:,2] is already correct with fp[:,0]

    _ASSETS = (faces.astype(np.int32), face_part.astype(np.uint8))
    return _ASSETS


# ============================================================
# Per-person rendering
# ============================================================

@dataclass
class SoloRender:
    """Per-person render (amodal, ROI may extend outside the image)

    label_map / depth_map are in ROI-local coordinates:
        image_xy = origin + local_xy * scale
    """
    label_map: np.ndarray   # (H, W) uint8: 0=background, 1=head, 2=torso, 3=limbs, 4=hands
    depth_map: np.ndarray   # (H, W) float32: camera depth z (metres), background = 0
    origin: tuple[float, float]  # image coordinates of the ROI top-left (current resolution)
    scale: float            # ROI-local -> image scale (>= 1 for downscaled renders)


def project_verts(verts: np.ndarray, focal_length: float) -> tuple[np.ndarray, np.ndarray]:
    """Project vertices to pixel coordinates of the current resolution

    Args:
        verts: (V, 3) camera coordinates (metres)
        focal_length: focal length (4K reference)

    Returns:
        uv: (V, 2) pixel coordinates (current resolution)
        z: (V,) camera depth
    """
    z = verts[:, 2]
    z_safe = np.maximum(z, 1e-6)
    u = (focal_length * verts[:, 0] / z_safe + CX) * resolution_config.scale_x
    v = (focal_length * verts[:, 1] / z_safe + CY) * resolution_config.scale_y
    return np.stack([u, v], axis=1), z


def render_person_solo(
    verts: np.ndarray,
    focal_length: float,
    max_side: int = RENDER_MAX_SIDE,
) -> Optional[SoloRender]:
    """Render one person's SMPL-X mesh alone (amodal)

    Args:
        verts: (10475, 3) camera-space vertices (metres)
        focal_length: focal length (4K reference)
        max_side: maximum side of the render

    Returns:
        SoloRender, or None if nothing can be drawn (e.g. behind the camera)
    """
    faces, face_part = load_smplx_assets()

    uv, z = project_verts(verts, focal_length)
    if not np.all(np.isfinite(uv)):
        return None

    # Drop faces with vertices behind the camera
    valid_z = z > 0.05
    face_valid = valid_z[faces].all(axis=1)
    if not face_valid.any():
        return None

    # ROI = projected extent of the used vertices (including outside the image = amodal)
    used_verts = np.unique(faces[face_valid])
    u0 = float(np.floor(uv[used_verts, 0].min())) - 1
    v0 = float(np.floor(uv[used_verts, 1].min())) - 1
    u1 = float(np.ceil(uv[used_verts, 0].max())) + 1
    v1 = float(np.ceil(uv[used_verts, 1].max())) + 1
    roi_w = u1 - u0
    roi_h = v1 - v0
    if roi_w <= 1 or roi_h <= 1:
        return None

    scale = max(1.0, max(roi_w, roi_h) / max_side)
    W = max(2, int(np.ceil(roi_w / scale)))
    H = max(2, int(np.ceil(roi_h / scale)))

    # ROI-local coordinates (fixed point with shift=4 for sub-pixel accuracy)
    SHIFT = 4
    local = (uv - np.array([u0, v0])) / scale
    pts_fixed = np.round(local * (1 << SHIFT)).astype(np.int32)  # (V, 2)

    # Back to front (mean face depth)
    face_idx = np.where(face_valid)[0]
    face_z = z[faces[face_idx]].mean(axis=1)
    order = face_idx[np.argsort(-face_z)]

    label_map = np.zeros((H, W), dtype=np.uint8)
    depth_map = np.zeros((H, W), dtype=np.float32)

    tri_pts = pts_fixed[faces[order]]  # (Fv, 3, 2)
    tri_labels = face_part[order]
    tri_depths = z[faces[order]].mean(axis=1).astype(np.float64)

    for i in range(len(order)):
        pts = tri_pts[i]
        cv2.fillConvexPoly(label_map, pts, int(tri_labels[i]), lineType=cv2.LINE_8, shift=SHIFT)
        cv2.fillConvexPoly(depth_map, pts, float(tri_depths[i]), lineType=cv2.LINE_8, shift=SHIFT)

    return SoloRender(
        label_map=label_map,
        depth_map=depth_map,
        origin=(u0, v0),
        scale=scale,
    )


# ============================================================
# Pseudo-depth conversion
# ============================================================

def pseudo_depth_from_teacher(
    depth_u16: np.ndarray,
    pelvis_depth: float,
    focal_scaled: float,
) -> np.ndarray:
    """Convert a depth teacher (uint16 mm, 0 = background) to pelvis-relative pseudo depth

    Same scale as the body keypoint pseudo depth `(z - z_pelvis) * f_scaled / z_pelvis`
    (the y of keypoints_body_cam): pixel values near a keypoint match that
    keypoint's depth value.

    Args:
        depth_u16: (H, W) uint16 depth teacher (millimetres, 0 = background)
        pelvis_depth: pelvis camera depth z (metres, annotation pelvis_depth)
        focal_scaled: scaled focal length (image_info focal_length)

    Returns:
        (H, W) float32 pseudo depth (pixel-equivalent scale, pelvis = 0, background = NaN)
    """
    out = np.full(depth_u16.shape, np.nan, dtype=np.float32)
    valid = depth_u16 > 0
    if pelvis_depth is None or pelvis_depth <= 0 or not valid.any():
        return out
    z = depth_u16[valid].astype(np.float32) / 1000.0
    out[valid] = (z - pelvis_depth) * focal_scaled / pelvis_depth
    return out


# ============================================================
# Sampling into crop coordinates
# ============================================================

def sample_maps_for_crop(
    solo: SoloRender,
    cx: float,
    cy: float,
    angle_deg: float,
    crop_width: int,
    crop_height: int,
    teacher_scale: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a SoloRender into teacher-resolution maps of a crop

    The crop is "rotate the source image by angle_deg around (cx, cy), then cut
    the centred rectangle" (same definition as crop_rotated_image_from_array).
    Teacher maps are (crop_height / teacher_scale, crop_width / teacher_scale).

    Labels and depth are sampled with nearest neighbour (bilinear depth would
    mix with the background 0 at silhouette edges and produce invalid values).

    Returns:
        label_teacher: (th, tw) uint8
        depth_teacher: (th, tw) float32 (camera depth z, background = 0)
    """
    th = max(1, crop_height // teacher_scale)
    tw = max(1, crop_width // teacher_scale)

    # Crop coordinates of the teacher pixel centres
    u = (np.arange(tw, dtype=np.float64) + 0.5) * (crop_width / tw)
    v = (np.arange(th, dtype=np.float64) + 0.5) * (crop_height / th)
    grid_v, grid_u = np.meshgrid(v, u, indexing="ij")

    # Crop -> rotated image -> source image (same inverse mapping as GPUImageProcessor)
    dx = grid_u - crop_width / 2
    dy = grid_v - crop_height / 2
    angle_rad = angle_deg * np.pi / 180.0
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    src_x = cx + cos_a * dx - sin_a * dy
    src_y = cy + sin_a * dx + cos_a * dy

    # Source image -> SoloRender local coordinates
    lx = (src_x - solo.origin[0]) / solo.scale
    ly = (src_y - solo.origin[1]) / solo.scale

    H, W = solo.label_map.shape
    ix = np.round(lx).astype(np.int64)
    iy = np.round(ly).astype(np.int64)
    inside = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)

    label_teacher = np.zeros((th, tw), dtype=np.uint8)
    depth_teacher = np.zeros((th, tw), dtype=np.float32)
    if inside.any():
        label_teacher[inside] = solo.label_map[iy[inside], ix[inside]]
        depth_teacher[inside] = solo.depth_map[iy[inside], ix[inside]]

    return label_teacher, depth_teacher
