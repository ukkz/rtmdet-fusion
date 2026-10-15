#!/usr/bin/env python3
"""
Amodal / visible mask utilities.

Semantics of the AGORA masks (verified on the data):
- individual masks (_00001 ...): amodal (each person rendered alone, including
  parts hidden by other persons or scene objects)
- integrated mask (_00000): colour-ID map of the visible regions (reflects both
  person-person and object occlusion)

This module assigns the colours of the integrated mask to persons and recovers
each person's visible mask. Occlusion teacher = individual (amodal) mask minus
visible mask.

Colour assignment:
    For every unique colour of the integrated mask, compute the fraction of
    that colour's region contained in each person's individual mask
    (containment) and assign the colour to the person with containment > 0.5.
    Anti-aliasing colour variants are absorbed by the same person.
    Matching against the individual mask colour itself is not used: for highly
    occluded persons the two colours can differ (this fails for about 25% of
    persons).
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def _pack_colors(img_bgr: np.ndarray) -> np.ndarray:
    """Pack (H, W, 3) BGR into (H, W) int64"""
    b = img_bgr[:, :, 0].astype(np.int64)
    g = img_bgr[:, :, 1].astype(np.int64)
    r = img_bgr[:, :, 2].astype(np.int64)
    return (b << 16) | (g << 8) | r


def compute_visible_masks(
    integrated_path: Path,
    person_masks: dict[int, np.ndarray],
    min_color_count: int = 10,
    containment_threshold: float = 0.5,
) -> dict[int, Optional[np.ndarray]]:
    """Recover each person's visible mask from the integrated mask

    Args:
        integrated_path: path of the integrated mask (_00000.png)
        person_masks: person_idx -> amodal binary mask ((H, W) uint8, same resolution)
        min_color_count: colours with fewer pixels are ignored (noise)
        containment_threshold: assign a colour only if at least this fraction
            of its region lies inside the person mask

    Returns:
        person_idx -> visible binary mask (uint8 0/1). All None if the integrated
        mask cannot be read
    """
    result: dict[int, Optional[np.ndarray]] = {p: None for p in person_masks}
    if integrated_path is None:
        return result

    integrated = cv2.imread(str(integrated_path), cv2.IMREAD_UNCHANGED)
    if integrated is None or integrated.ndim != 3:
        return result

    packed = _pack_colors(integrated)

    # All unique colours and their pixel counts
    colors, counts = np.unique(packed, return_counts=True)
    valid = counts >= min_color_count
    colors = colors[valid]
    counts = counts[valid]
    if len(colors) == 0:
        return result

    # Colour -> array index (searchsorted; colors is sorted)
    def color_indices(packed_values: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(colors, packed_values)
        idx[idx >= len(colors)] = 0
        return idx

    # Containment from the colour histogram inside each person mask
    # assignment[color_idx] = person_idx or -1
    assignment = np.full(len(colors), -1, dtype=np.int64)
    best_containment = np.zeros(len(colors), dtype=np.float64)

    for p, mask in person_masks.items():
        if mask is None:
            continue
        vals_in = packed[mask > 0]
        if len(vals_in) == 0:
            continue
        c_in, n_in = np.unique(vals_in, return_counts=True)
        idx = color_indices(c_in)
        # Check the searchsorted hit (the mask can contain colours below min_color_count)
        ok = colors[idx] == c_in
        idx = idx[ok]
        n_in = n_in[ok]
        containment = n_in / counts[idx]
        better = containment > best_containment[idx]
        assignment[idx[better]] = p
        best_containment[idx[better]] = containment[better]

    assignment[best_containment < containment_threshold] = -1

    # Visible mask per person
    for p, mask in person_masks.items():
        if mask is None:
            continue
        assigned_colors = colors[assignment == p]
        if len(assigned_colors) == 0:
            result[p] = np.zeros_like(mask)
            continue
        visible = np.isin(packed, assigned_colors)
        # Restrict to the amodal mask (removes stray mis-assigned islands)
        visible &= mask > 0
        result[p] = visible.astype(np.uint8)

    return result


def visible_ratio(
    amodal_mask: Optional[np.ndarray],
    visible_mask: Optional[np.ndarray],
) -> Optional[float]:
    """Visible ratio |visible| / |amodal| (None if either mask is missing)"""
    if amodal_mask is None or visible_mask is None:
        return None
    total = int((amodal_mask > 0).sum())
    if total == 0:
        return None
    return float((visible_mask > 0).sum()) / total
