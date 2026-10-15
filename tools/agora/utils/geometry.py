#!/usr/bin/env python3
"""
Geometry utilities.

Pure mathematical / geometric helpers.
"""

import numpy as np


def rotate_point_around_center(
    x: float, y: float, cx: float, cy: float, angle_deg: float
) -> tuple[float, float]:
    """
    Rotate a point around a centre.

    Args:
        x, y: point to rotate
        cx, cy: rotation centre
        angle_deg: rotation angle (degrees, counter-clockwise positive)

    Returns:
        rotated point (x', y')
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Move the centre to the origin
    dx = x - cx
    dy = y - cy

    # Rotate
    new_x = dx * cos_a - dy * sin_a + cx
    new_y = dx * sin_a + dy * cos_a + cy

    return new_x, new_y


def get_rotated_rect_bounding_box(
    width: int, height: int, angle_deg: float
) -> tuple[float, float]:
    """
    Size of the axis-aligned bounding box of a rotated rectangle.

    Args:
        width, height: rectangle size
        angle_deg: rotation angle (degrees)

    Returns:
        (bounding_width, bounding_height)
    """
    angle_rad = np.deg2rad(abs(angle_deg))
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    bw = width * cos_a + height * sin_a
    bh = width * sin_a + height * cos_a

    return bw, bh


def get_valid_center_range(
    crop_width: int, crop_height: int, angle_deg: float,
    img_width: int, img_height: int
) -> tuple[float, float, float, float] | None:
    """
    Valid range of the centre of a rotated crop (crop stays inside the image).

    Args:
        crop_width, crop_height: crop size
        angle_deg: rotation angle (degrees)
        img_width, img_height: source image size

    Returns:
        (cx_min, cx_max, cy_min, cy_max), or None if no valid range exists
    """
    # Bounding box of the rotated crop
    bw, bh = get_rotated_rect_bounding_box(crop_width, crop_height, angle_deg)

    # cx: bw/2 .. img_width - bw/2
    # cy: bh/2 .. img_height - bh/2
    cx_min = bw / 2
    cx_max = img_width - bw / 2
    cy_min = bh / 2
    cy_max = img_height - bh / 2

    if cx_max < cx_min or cy_max < cy_min:
        return None

    return cx_min, cx_max, cy_min, cy_max


def find_nearest_vertex_to_line(
    verts: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
    mask: np.ndarray = None,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> int:
    """
    Index of the vertex closest to a line segment.

    Args:
        verts: vertices (N, 3)
        line_start: segment start
        line_end: segment end
        mask: vertices to search (None = all)
        t_min: start of the searched section along the segment (0.0-1.0)
        t_max: end of the searched section along the segment (0.0-1.0)

    Returns:
        index of the nearest vertex
    """
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return 0
    line_unit = line_vec / line_len

    if mask is not None:
        search_verts = verts[mask]
        search_indices = np.where(mask)[0]
    else:
        search_verts = verts
        search_indices = np.arange(len(verts))

    # Projection of every vertex onto the line
    to_verts = search_verts - line_start
    projections = np.dot(to_verts, line_unit)

    # Clip the projection to the searched section
    clip_min = line_len * t_min
    clip_max = line_len * t_max
    projections_clipped = np.clip(projections, clip_min, clip_max)
    closest_points = line_start + projections_clipped[:, np.newaxis] * line_unit

    distances = np.linalg.norm(search_verts - closest_points, axis=1)

    best_local_idx = np.argmin(distances)
    return search_indices[best_local_idx]
