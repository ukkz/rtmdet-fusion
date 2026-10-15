#!/usr/bin/env python3
"""
Annotation utilities.

Horizontal mirroring of annotations.
"""

import sys
from pathlib import Path

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import BODY_KEYPOINT_SWAP_PAIRS
from utils.image import mirror_mask_rle


def _mirror_keypoints_flat(
    kpts: list,
    elem_size: int,
    swap_pairs: list,
    x_transform,
) -> list:
    """Flip X and swap left/right points in a flat keypoint list

    Args:
        kpts: flat list [x, ...] * N
        elem_size: elements per point (3 or 4)
        swap_pairs: (left_idx, right_idx) pairs to swap
        x_transform: transform applied to x
    """
    for i in range(0, len(kpts), elem_size):
        kpts[i] = x_transform(kpts[i])
    for left_idx, right_idx in swap_pairs:
        ls = left_idx * elem_size
        rs = right_idx * elem_size
        kpts[ls:ls + elem_size], kpts[rs:rs + elem_size] = (
            kpts[rs:rs + elem_size], kpts[ls:ls + elem_size]
        )
    return kpts


def mirror_annotations(
    annotations: list,
    crop_width: int,
) -> None:
    """
    Mirror annotations horizontally (to match a horizontally flipped image).

    X is flipped in every coordinate field:
    - keypoints: 2D [x, y, v, ...] (3 per point) -> x' = crop_width - x
    - keypoints_cam: [x, y, z, v, ...] (4 per point) -> x' = crop_width - x (image X)
    - bbox, head_bbox: [x, y, w, h] -> x' = crop_width - x - w
    - segmentation: RLE mask flipped

    Left/right head keypoints are swapped:
    - left_ear (2) <-> right_ear (3)
    - left_eye (4) <-> right_eye (5)

    Args:
        annotations: annotation list (modified in place)
        crop_width: crop image width
    """
    # Head keypoint index pairs to swap
    swap_pairs = [(2, 3), (4, 5)]

    for ann in annotations:
        # keypoints: [x, y, v, x, y, v, ...] (3 per point)
        kpts = ann["keypoints"]
        for i in range(0, len(kpts), 3):
            kpts[i] = crop_width - kpts[i]  # flip x

        # Swap left/right keypoints (3 elements each)
        for left_idx, right_idx in swap_pairs:
            left_start = left_idx * 3
            right_start = right_idx * 3
            kpts[left_start:left_start+3], kpts[right_start:right_start+3] = \
                kpts[right_start:right_start+3], kpts[left_start:left_start+3]
        ann["keypoints"] = kpts

        # keypoints_cam: [x, y, z, v, x, y, z, v, ...] (4 per point)
        # x = image X (right positive) -> flipped
        # y = depth -> unchanged
        # z = height from the image bottom (up positive) -> unchanged
        # v = visibility -> unchanged
        kpts_cam = ann["keypoints_cam"]
        for i in range(0, len(kpts_cam), 4):
            kpts_cam[i] = crop_width - kpts_cam[i]  # flip x

        # Swap left/right keypoints (4 elements each)
        for left_idx, right_idx in swap_pairs:
            left_start = left_idx * 4
            right_start = right_idx * 4
            kpts_cam[left_start:left_start+4], kpts_cam[right_start:right_start+4] = \
                kpts_cam[right_start:right_start+4], kpts_cam[left_start:left_start+4]
        ann["keypoints_cam"] = kpts_cam

        # bbox: [x, y, width, height]
        bbox = ann["bbox"]
        bbox[0] = crop_width - bbox[0] - bbox[2]  # x' = crop_width - x - width
        ann["bbox"] = bbox

        # head_bbox: [x, y, width, height]
        head_bbox = ann["head_bbox"]
        head_bbox[0] = crop_width - head_bbox[0] - head_bbox[2]
        ann["head_bbox"] = head_bbox

        # segmentation: flip the RLE mask
        if ann.get("segmentation") is not None:
            ann["segmentation"] = mirror_mask_rle(ann["segmentation"], crop_width)

        # ---- Body keypoints, part / occlusion masks ----

        # keypoints_body: [x, y, v] * 31 (image coordinates)
        if "keypoints_body" in ann:
            ann["keypoints_body"] = _mirror_keypoints_flat(
                ann["keypoints_body"], 3, BODY_KEYPOINT_SWAP_PAIRS,
                lambda x: crop_width - x,
            )

        # keypoints_body_cam: [x, y, z, v] * 31 (x = image X, y/z unchanged)
        if "keypoints_body_cam" in ann:
            ann["keypoints_body_cam"] = _mirror_keypoints_flat(
                ann["keypoints_body_cam"], 4, BODY_KEYPOINT_SWAP_PAIRS,
                lambda x: crop_width - x,
            )

        # body_ref_bbox: [x, y, w, h]
        if ann.get("body_ref_bbox") is not None:
            brb = ann["body_ref_bbox"]
            brb[0] = crop_width - brb[0] - brb[2]
            ann["body_ref_bbox"] = brb

        # part_masks: flip the part RLEs (teacher resolution)
        if ann.get("part_masks") is not None:
            ann["part_masks"] = {
                name: (mirror_mask_rle(rle, crop_width) if rle is not None else None)
                for name, rle in ann["part_masks"].items()
            }

        # occlusion_mask: flip the RLE (crop resolution)
        if ann.get("occlusion_mask") is not None:
            ann["occlusion_mask"] = mirror_mask_rle(ann["occlusion_mask"], crop_width)

        # depth_file: shared with the original; the dataset flips it using image.mirrored
