#!/usr/bin/env python3
"""
Keypoint utilities.

Computation of head / body keypoints from SMPL-X and their transformation
into crop coordinates.
"""

import numpy as np

import sys
from pathlib import Path

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    JOINT_INDICES,
    CX,
    CY,
    resolution_config,
    BODY_KEYPOINT_SMPLX_INDICES,
)
from utils.geometry import rotate_point_around_center, find_nearest_vertex_to_line


def project_3d_to_2d(point_3d: np.ndarray, focal_length: float) -> np.ndarray:
    """Project a 3D camera-space point to image coordinates (current resolution)"""
    x, y, z = point_3d
    if z <= 0:
        return np.array([0.0, 0.0])
    u = focal_length * x / z + CX
    v = focal_length * y / z + CY
    # Scale to the current source resolution
    u *= resolution_config.scale_x
    v *= resolution_config.scale_y
    return np.array([u, v])


def compute_head_keypoints(
    joints_3d: np.ndarray,
    verts: np.ndarray,
    joints_2d: np.ndarray,
    focal_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the 11 head keypoints from SMPL-X joints and vertices.

    Args:
        joints_3d: 3D joints (127, 3)
        verts: vertices (10475, 3)
        joints_2d: 2D joints (127, 2)
        focal_length: camera focal length

    Returns:
        keypoints_3d: (11, 3)
        keypoints_2d: (11, 2)
    """
    n_keypoints = 11
    keypoints_3d = np.zeros((n_keypoints, 3))
    keypoints_2d = np.zeros((n_keypoints, 2))

    # Indices of points taken from vertices (projected to 2D below)
    computed_vert_indices = {}

    # Base joints
    head_joint = joints_3d[JOINT_INDICES["head"]]
    neck_joint = joints_3d[JOINT_INDICES["neck"]]
    nose_joint = joints_3d[JOINT_INDICES["nose"]]

    # Head up direction (neck -> head)
    head_up = head_joint - neck_joint

    # Vertices around the head
    head_center = head_joint
    distances_to_head = np.linalg.norm(verts - head_center, axis=1)
    head_mask = distances_to_head < 0.3  # within 30 cm

    # 0: center - head joint
    keypoints_3d[0] = head_joint

    # 1: top - vertex nearest to the neck->head segment extended 3x
    head_top_line_end = head_joint + head_up * 2  # 3x the neck->head vector
    head_top_idx = find_nearest_vertex_to_line(verts, neck_joint, head_top_line_end, head_mask)
    keypoints_3d[1] = verts[head_top_idx]
    computed_vert_indices[1] = head_top_idx

    # 2: left_ear
    keypoints_3d[2] = joints_3d[JOINT_INDICES["left_ear"]]

    # 3: right_ear
    keypoints_3d[3] = joints_3d[JOINT_INDICES["right_ear"]]

    # 4: left_eye
    keypoints_3d[4] = joints_3d[JOINT_INDICES["left_eye"]]

    # 5: right_eye
    keypoints_3d[5] = joints_3d[JOINT_INDICES["right_eye"]]

    # 6: nose
    keypoints_3d[6] = joints_3d[JOINT_INDICES["nose"]]

    # 7: chin - vertex nearest to the 1/4-3/4 section of the nose-neck segment
    chin_idx = find_nearest_vertex_to_line(
        verts, nose_joint, neck_joint, head_mask,
        t_min=0.25, t_max=0.75
    )
    keypoints_3d[7] = verts[chin_idx]
    computed_vert_indices[7] = chin_idx

    # 8: occiput - vertex nearest to the segment from (nose->head extended 3x) to center
    nose_to_head = head_joint - nose_joint
    occiput_line_start = head_joint + nose_to_head * 2  # 3x the nose->head vector (behind the head)
    occiput_idx = find_nearest_vertex_to_line(verts, occiput_line_start, head_joint, head_mask)
    keypoints_3d[8] = verts[occiput_idx]
    computed_vert_indices[8] = occiput_idx

    # 9: vertex - vertex nearest to the segment from B to center, where
    #    B = center + 3 * ((top - center) + (occiput - center))
    center_to_top = keypoints_3d[1] - head_joint
    center_to_occiput = keypoints_3d[8] - head_joint
    combined_vec = center_to_top + center_to_occiput
    vertex_line_start = head_joint + combined_vec * 3
    vertex_idx = find_nearest_vertex_to_line(verts, vertex_line_start, head_joint, head_mask)
    keypoints_3d[9] = verts[vertex_idx]
    computed_vert_indices[9] = vertex_idx

    # 10: neck
    keypoints_3d[10] = neck_joint

    # 2D coordinates: joints are scaled from the native 2D projections
    joints_2d_scaled = joints_2d.copy()
    joints_2d_scaled[:, 0] *= resolution_config.scale_x
    joints_2d_scaled[:, 1] *= resolution_config.scale_y

    keypoints_2d[0] = joints_2d_scaled[JOINT_INDICES["head"]]      # center
    keypoints_2d[2] = joints_2d_scaled[JOINT_INDICES["left_ear"]]
    keypoints_2d[3] = joints_2d_scaled[JOINT_INDICES["right_ear"]]
    keypoints_2d[4] = joints_2d_scaled[JOINT_INDICES["left_eye"]]
    keypoints_2d[5] = joints_2d_scaled[JOINT_INDICES["right_eye"]]
    keypoints_2d[6] = joints_2d_scaled[JOINT_INDICES["nose"]]
    keypoints_2d[10] = joints_2d_scaled[JOINT_INDICES["neck"]]

    # Vertex-based points are projected from 3D
    for kpt_idx, vert_idx in computed_vert_indices.items():
        keypoints_2d[kpt_idx] = project_3d_to_2d(verts[vert_idx], focal_length)

    return keypoints_3d, keypoints_2d


def compute_body_keypoints(
    joints_3d: np.ndarray,
    joints_2d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the 31 body keypoints from the 127 SMPL-X joints.

    All points are actual joints (no vertex search), so this is only index
    selection and scaling. Definition and order: config.BODY_KEYPOINT_DEFS
    (the first 17 points follow COCO-17).

    Args:
        joints_3d: 3D joints (127, 3), camera coordinates
        joints_2d: 2D joints (127, 2), native (4K) pixels

    Returns:
        keypoints_3d: (31, 3) camera coordinates
        keypoints_2d: (31, 2) pixels at the current source resolution
    """
    indices = BODY_KEYPOINT_SMPLX_INDICES
    keypoints_3d = joints_3d[indices].copy()
    keypoints_2d = joints_2d[indices].copy()
    keypoints_2d[:, 0] *= resolution_config.scale_x
    keypoints_2d[:, 1] *= resolution_config.scale_y
    return keypoints_3d, keypoints_2d


def compute_ref_bbox_cropped(
    cropped_kpts: np.ndarray,
    expand_ratio: float,
) -> list:
    """
    Compute a reference bbox (expanded square) from cropped 2D keypoints.

    Generic version of compute_head_bbox_cropped:
    1. bounding rectangle of all points (including out of frame)
    2. square it on the longer side (centre kept)
    3. expand by expand_ratio
    Not clamped to the crop (may extend outside the image).

    Args:
        cropped_kpts: cropped 2D keypoints (N, 2)
        expand_ratio: expansion ratio

    Returns:
        bbox: [x, y, width, height] (square)
    """
    x_min = float(cropped_kpts[:, 0].min())
    y_min = float(cropped_kpts[:, 1].min())
    x_max = float(cropped_kpts[:, 0].max())
    y_max = float(cropped_kpts[:, 1].max())

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    half_size = max(x_max - x_min, y_max - y_min) / 2 * expand_ratio

    return [
        center_x - half_size,
        center_y - half_size,
        half_size * 2,
        half_size * 2,
    ]


def transform_keypoints_with_rotation(
    kpts_2d: np.ndarray,
    cx: float,
    cy: float,
    angle_deg: float,
    crop_width: int,
    crop_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform 2D keypoints into rotated-crop coordinates and compute visibility.

    Rotating the image with cv2.warpAffine moves image points in the opposite
    direction, so the keypoints are rotated by -angle_deg.

    Args:
        kpts_2d: 2D keypoints in the source image (N, 2)
        cx, cy: crop centre (source image coordinates)
        angle_deg: rotation angle (degrees), same value as used for the image
        crop_width, crop_height: crop size

    Returns:
        cropped_kpts: crop coordinates (N, 2)
        visibility: (N,) 1 = inside the crop, 0 = outside
    """
    n_kpts = len(kpts_2d)
    cropped_kpts = np.zeros((n_kpts, 2))
    visibility = np.ones(n_kpts, dtype=np.int32)

    # Rotating the image moves its points the opposite way
    effective_angle = -angle_deg

    for i, (x, y) in enumerate(kpts_2d):
        # Rotate around the crop centre
        rx, ry = rotate_point_around_center(x, y, cx, cy, effective_angle)

        # To crop coordinates (centre maps to (crop_width/2, crop_height/2))
        cropped_x = rx - (cx - crop_width / 2)
        cropped_y = ry - (cy - crop_height / 2)

        cropped_kpts[i] = [cropped_x, cropped_y]

        if cropped_x < 0 or cropped_x >= crop_width or cropped_y < 0 or cropped_y >= crop_height:
            visibility[i] = 0

    return cropped_kpts, visibility


def compute_head_bbox_cropped(
    cropped_kpts: np.ndarray,
    expand_ratio: float = 1.2,
) -> list:
    """
    Compute the head bbox from the cropped head keypoints (all 11 points).

    1. bounding rectangle of all 11 points (including out of frame)
    2. square it on the longer side (centre kept)
    3. expand by expand_ratio

    Not clamped: the head bbox may extend outside the image. The minimum-area
    ignore check is done by the caller (MIN_HEAD_BBOX_AREA).

    Args:
        cropped_kpts: cropped 2D keypoints (11, 2)
        expand_ratio: expansion ratio after squaring

    Returns:
        head_bbox: [x, y, width, height] (square)
    """
    x_min = float(cropped_kpts[:, 0].min())
    y_min = float(cropped_kpts[:, 1].min())
    x_max = float(cropped_kpts[:, 0].max())
    y_max = float(cropped_kpts[:, 1].max())

    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    # Square on the longer side, then expand (centre kept)
    target_size = max(x_max - x_min, y_max - y_min)
    half_size_expanded = target_size / 2 * expand_ratio
    x_min = center_x - half_size_expanded
    y_min = center_y - half_size_expanded
    x_max = center_x + half_size_expanded

    final_size = x_max - x_min
    return [x_min, y_min, final_size, final_size]


def is_any_keypoint_visible(
    cropped_kpts: np.ndarray,
    crop_width: int,
    crop_height: int,
) -> bool:
    """
    Check whether at least one keypoint lies inside the crop.

    Args:
        cropped_kpts: cropped 2D keypoints (N, 2)
        crop_width, crop_height: crop size

    Returns:
        True if any point is inside the image
    """
    for x, y in cropped_kpts:
        if 0 <= x < crop_width and 0 <= y < crop_height:
            return True
    return False
