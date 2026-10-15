#!/usr/bin/env python3
"""
Extraction utilities.

Shared logic for a single crop attempt, used by both extracter.py and test.py
so that the interactive preview matches the generated dataset.
"""

import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import cv2
import numpy as np

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    JOINT_INDICES,
    MIN_HEAD_BBOX_AREA,
    resolution_config,
    # Matching thresholds
    MATCHING_IOU_THRESHOLD,
    MATCHING_SCORE_THRESHOLD,
    MATCHING_MIN_COVERAGE,
    # Extraction probabilities
    NO_VALID_PERSON_SKIP_PROB,
    ROTATION_CHANCE,
    # Extraction parameters
    ROTATION_ANGLE_MIN,
    ROTATION_ANGLE_MAX,
    # Body keypoints, part masks, depth, occlusion
    BODY_PELVIS_INDEX,
    BODY_BBOX_EXPAND_RATIO,
    MIN_BODY_BBOX_AREA,
    PART_GROUP_NAMES,
    TEACHER_MAP_SCALE,
    OCCLUSION_MISMATCH_TOLERANCE,
)
from utils.geometry import get_valid_center_range
from utils.image import (
    crop_rotated_image_from_array,
    compute_rotated_cropped_bbox_from_array,
    compute_rotated_cropped_mask_rle_from_array,
    load_mask_2d,
    GPUImageProcessor,
)
from utils.keypoints import (
    compute_head_keypoints,
    compute_body_keypoints,
    compute_ref_bbox_cropped,
    transform_keypoints_with_rotation,
    compute_head_bbox_cropped,
    is_any_keypoint_visible,
)
from utils.path import get_mask_path
from utils.amodal import compute_visible_masks, visible_ratio
from utils.render import render_person_solo, sample_maps_for_crop
from utils.matching import decode_rle_mask, match_masks, MatchResult
from pycocotools import mask as mask_utils

if TYPE_CHECKING:
    import pandas as pd
    from utils.rtmdet_ins import RTMDetInsInference


# Extracts the scan name (subject + pose number)
_SUBJECT_RE = re.compile(r"^(?P<sid>.+?_posed_\d+)")

# Print the "SMPL-X assets missing" warning only once
_render_warned = False


def parse_subject_id(smplx_path: str) -> Optional[str]:
    """Extract the scan ID (subject + pose number) from gt_path_smplx

    e.g. 'smplx_gt/.../rp_tanja_posed_007_0_0.obj' -> 'rp_tanja_posed_007'

    The same subject can wear different clothes in scans with different pose
    numbers, so the Re-ID identity granularity includes the pose number.
    """
    if not smplx_path:
        return None
    stem = Path(smplx_path).name
    if stem.endswith(".obj"):
        stem = stem[:-4]
    m = _SUBJECT_RE.match(stem)
    if m:
        return m.group("sid")
    # Fallback: strip trailing numeric suffixes (_0_0 etc.)
    fallback = re.sub(r"(_\d+)+$", "", stem)
    return fallback if fallback else stem


def _encode_binary_rle(binary_mask: np.ndarray) -> Optional[dict]:
    """Encode a binary mask as COCO RLE (counts=str). Returns None if empty"""
    if not binary_mask.any():
        return None
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def fill_depth_into_clothed(
    depth_t: np.ndarray,
    clothed_mask_t: np.ndarray,
) -> np.ndarray:
    """Extend the naked SMPL-X depth into the clothed mask by nearest-neighbour fill

    The depth teacher must give every pixel of the clothed whole-body mask (ch0)
    a depth, so hair / clothing pixels outside the naked silhouette take the
    depth of the nearest body-surface pixel.

    Args:
        depth_t: (H, W) float32 camera depth z (m), 0 = background (naked silhouette)
        clothed_mask_t: (H, W) clothed amodal mask at teacher resolution (>0 = person)

    Returns:
        Filled depth map. Support = naked silhouette ∪ clothed mask
    """
    valid = depth_t > 0
    if not valid.any():
        return depth_t
    target = (clothed_mask_t > 0) & ~valid
    if not target.any():
        return depth_t

    # Distance transform gives the nearest valid pixel for every invalid one
    invalid_u8 = (~valid).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        invalid_u8, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    ys, xs = np.nonzero(valid)
    lut = np.zeros(int(labels.max()) + 1, dtype=np.float32)
    lut[labels[ys, xs]] = depth_t[ys, xs]

    out = depth_t.copy()
    out[target] = lut[labels[target]]
    return out


def ensure_row_cache(
    row_cache: dict,
    row: "pd.Series",
    img_name: str,
    split: str,
    focal_length: float,
) -> None:
    """Build the expensive per-row (source image) data once and cache it

    Everything that does not depend on the crop parameters is computed here and
    reused across all crop attempts (including retries) on the same image:
    - per-person amodal masks (source resolution)
    - occlusion masks and visible ratios from the integrated mask (_00000)
    - per-person SMPL-X renders (part labels + depth)
    - meta data such as subject_id / kid / occlusion%
    """
    global _render_warned
    if row_cache.get("ready"):
        return

    n_persons = len(row["gt_joints_2d"])
    persons: dict[int, dict] = {}

    has_subject = "gt_path_smplx" in row.index
    has_kid = "kid" in row.index
    has_occ = "occlusion" in row.index

    # Load the amodal masks
    amodal_binary: dict[int, np.ndarray] = {}
    for p in range(n_persons):
        if not row["isValid"][p]:
            continue
        entry: dict = {
            "amodal": None,
            "occ": None,
            "vis_ratio": None,
            "solo": None,
            "subject_id": parse_subject_id(row["gt_path_smplx"][p]) if has_subject else None,
            "kid": bool(row["kid"][p]) if has_kid else None,
            "occlusion_pct": float(row["occlusion"][p]) if has_occ else None,
        }
        mask_path = get_mask_path(img_name, p, split)
        if mask_path is not None:
            mask_2d = load_mask_2d(mask_path)
            if mask_2d is not None:
                entry["amodal"] = mask_2d
                amodal_binary[p] = (mask_2d > 0).astype(np.uint8)
        persons[p] = entry

    # Recover visible masks from the integrated mask -> occlusion masks and visible ratios
    if amodal_binary:
        integrated_path = get_mask_path(img_name, -1, split)  # person_idx=-1 -> _00000
        visible_masks = compute_visible_masks(integrated_path, amodal_binary)
        for p, vis in visible_masks.items():
            if vis is None:
                continue
            amodal = amodal_binary[p]
            persons[p]["vis_ratio"] = visible_ratio(amodal, vis)
            persons[p]["occ"] = ((amodal > 0) & (vis == 0)).astype(np.uint8)

    # Per-person SMPL-X render (disabled with a warning if the assets are missing)
    for p, entry in persons.items():
        try:
            entry["solo"] = render_person_solo(
                np.asarray(row["gt_verts"][p], dtype=np.float64), focal_length
            )
        except FileNotFoundError as e:
            if not _render_warned:
                print(f"\n[WARN] SMPL-X rendering disabled (no part mask / depth teachers): {e}")
                _render_warned = True
            break
        except Exception:
            entry["solo"] = None

    row_cache["persons"] = persons
    row_cache["ready"] = True


def compute_visibility_with_occlusion(
    cropped_kpts: np.ndarray,
    visibility_basic: np.ndarray,
    self_data: dict,
    person_data: list,
    crop_width: int,
    crop_height: int,
) -> np.ndarray:
    """Compute keypoint visibility (0=out of frame, 1=occluded, 2=visible)

    A keypoint is occluded if it lies inside another person's cropped mask and
    that person is closer to the camera (smaller neck depth).
    """
    neck_depth = self_data["neck_depth"]
    visibility = np.zeros(len(cropped_kpts), dtype=np.int32)
    for kpt_idx in range(len(cropped_kpts)):
        if visibility_basic[kpt_idx] == 0:
            visibility[kpt_idx] = 0
            continue
        kpt_x = int(round(cropped_kpts[kpt_idx, 0]))
        kpt_y = int(round(cropped_kpts[kpt_idx, 1]))
        kpt_x = max(0, min(kpt_x, crop_width - 1))
        kpt_y = max(0, min(kpt_y, crop_height - 1))

        is_occluded = False
        for other_data in person_data:
            if other_data is self_data:
                continue
            other_mask = other_data["mask_decoded"]
            other_depth = other_data["neck_depth"]
            if other_mask is None:
                continue
            if other_mask[kpt_y, kpt_x] > 0 and other_depth < neck_depth:
                is_occluded = True
                break

        visibility[kpt_idx] = 1 if is_occluded else 2
    return visibility


@dataclass
class CropResult:
    """Result of a crop attempt"""
    success: bool
    cropped_img: Optional[np.ndarray] = None
    image_info: Optional[dict] = None
    annotations: list = None
    ignore_count: int = 0
    skip_reason: Optional[str] = None
    # RTMDet-Ins matching results
    ins_masks: List[np.ndarray] = field(default_factory=list)  # predicted masks (display)
    match_results: List[MatchResult] = field(default_factory=list)

    def __post_init__(self):
        if self.annotations is None:
            self.annotations = []


def try_single_crop(
    source_img: np.ndarray,
    row: "pd.Series",
    img_name: str,
    split: str,
    focal_length: float,
    crop_width: int,
    crop_height: int,
    angle_deg: float,
    image_id: int = 0,
    anno_id: int = 0,
    force_valid_person: bool = True,
    gpu_processor: Optional[GPUImageProcessor] = None,
    row_cache: Optional[dict] = None,
) -> CropResult:
    """
    Run a single crop attempt.

    Returns only the original (non-mirrored) crop; the caller creates and saves
    the mirrored copy. With gpu_processor the rotation and crop run on the GPU.

    Args:
        source_img: source image (numpy array)
        row: one row of the AGORA pkl DataFrame
        img_name: image file name
        split: train or validation
        focal_length: focal length of the scene (native resolution)
        crop_width: crop width
        crop_height: crop height
        angle_deg: rotation angle (degrees)
        image_id: image ID (COCO)
        anno_id: first annotation ID (COCO)
        force_valid_person: fail the attempt when no valid person is present
        gpu_processor: optional GPU processor holding source_img
        row_cache: per-row cache shared between attempts on the same image

    Returns:
        CropResult (not mirrored)
    """
    # Valid range of the crop centre
    valid_range = get_valid_center_range(
        crop_width, crop_height, angle_deg,
        resolution_config.source_width, resolution_config.source_height
    )

    # No valid range (the rotated crop's bounding box exceeds the source image)
    if valid_range is None:
        return CropResult(success=False, skip_reason="position_not_found")

    cx_min, cx_max, cy_min, cy_max = valid_range

    # Random crop centre within the valid range
    cx = random.uniform(cx_min, cx_max)
    cy = random.uniform(cy_min, cy_max)

    # Crop (GPU or CPU)
    if gpu_processor is not None:
        cropped_img = gpu_processor.crop_rotated(cx, cy, crop_width, crop_height, angle_deg)
    else:
        cropped_img = crop_rotated_image_from_array(
            source_img, cx, cy, crop_width, crop_height, angle_deg
        )
    if cropped_img is None:
        return CropResult(success=False, skip_reason="crop_failed")

    # Image info
    mirrored = False
    image_info = {
        "id": image_id,
        "file_name": "",  # set by the caller
        "width": crop_width,
        "height": crop_height,
        "focal_length": focal_length * resolution_config.scale_x,
        "rotated": float(angle_deg),
        "mirrored": mirrored,
        "cx": cx,
        "cy": cy,
    }

    # Annotations
    crop_annotations = []
    ignore_count = 0
    current_anno_id = anno_id
    n_persons = len(row["gt_joints_2d"])

    # Per-row cache (masks, renders, meta data; shared between attempts)
    if row_cache is None:
        row_cache = {}
    ensure_row_cache(row_cache, row, img_name, split, focal_length)
    cached_persons = row_cache["persons"]

    # Pass 1: per-person keypoints and masks
    person_data = []
    for person_idx in range(n_persons):
        if not row["isValid"][person_idx]:
            continue

        joints_3d = row["gt_joints_3d"][person_idx]
        joints_2d = row["gt_joints_2d"][person_idx]
        verts = row["gt_verts"][person_idx]

        # Head keypoints
        kpts_3d, kpts_2d = compute_head_keypoints(
            joints_3d, verts, joints_2d, focal_length
        )

        # To crop coordinates (visibility: 0=out of frame, 1=in frame)
        cropped_kpts, visibility_basic = transform_keypoints_with_rotation(
            kpts_2d, cx, cy, angle_deg, crop_width, crop_height
        )

        # Body keypoints (31)
        body_kpts_3d, body_kpts_2d = compute_body_keypoints(joints_3d, joints_2d)
        cropped_body_kpts, body_visibility_basic = transform_keypoints_with_rotation(
            body_kpts_2d, cx, cy, angle_deg, crop_width, crop_height
        )

        # Skip the person only if every head and body keypoint is out of frame
        # (partially visible persons are kept)
        head_in_frame = is_any_keypoint_visible(cropped_kpts, crop_width, crop_height)
        body_in_frame = is_any_keypoint_visible(cropped_body_kpts, crop_width, crop_height)
        if not head_in_frame and not body_in_frame:
            continue

        cache_p = cached_persons.get(person_idx, {})

        # Mask RLE and decoded mask (amodal, from the row cache)
        mask_2d = cache_p.get("amodal")
        segmentation_rle = None
        mask_decoded = None
        body_bbox = None
        if mask_2d is not None:
            body_bbox, _ = compute_rotated_cropped_bbox_from_array(
                mask_2d, cx, cy, angle_deg, crop_width, crop_height
            )
            segmentation_rle = compute_rotated_cropped_mask_rle_from_array(
                mask_2d, cx, cy, angle_deg, crop_width, crop_height
            )
            if segmentation_rle is not None:
                mask_decoded = mask_utils.decode(segmentation_rle)

        neck_depth = float(joints_3d[JOINT_INDICES["neck"], 2])

        person_data.append({
            "person_idx": person_idx,
            "kpts_3d": kpts_3d,
            "cropped_kpts": cropped_kpts,
            "visibility_basic": visibility_basic,
            "body_kpts_3d": body_kpts_3d,
            "cropped_body_kpts": cropped_body_kpts,
            "body_visibility_basic": body_visibility_basic,
            "head_in_frame": head_in_frame,
            "body_in_frame": body_in_frame,
            "cache": cache_p,
            "mask_decoded": mask_decoded,
            "segmentation_rle": segmentation_rle,
            "body_bbox": body_bbox,
            "neck_depth": neck_depth,
        })

    # Pass 2: occlusion-aware visibility and annotation fields
    for data in person_data:
        cropped_kpts = data["cropped_kpts"]
        kpts_3d = data["kpts_3d"]
        cache_p = data["cache"]

        # visibility: 0=out of frame, 1=occluded, 2=visible
        visibility = compute_visibility_with_occlusion(
            cropped_kpts, data["visibility_basic"],
            data, person_data, crop_width, crop_height,
        )
        body_visibility = compute_visibility_with_occlusion(
            data["cropped_body_kpts"], data["body_visibility_basic"],
            data, person_data, crop_width, crop_height,
        )

        # COCO-style 2D keypoints (x, y, v)
        keypoints_flat = []
        for i in range(len(cropped_kpts)):
            keypoints_flat.extend([
                float(cropped_kpts[i, 0]),
                float(cropped_kpts[i, 1]),
                int(visibility[i]),
            ])

        # Unified head keypoints (x, y, z, v)
        # x: image X (absolute pixels)
        # y: pseudo depth (center.y = 0, pixel-equivalent scale)
        # z: flipped image Y (absolute pixels, z = H - y)
        keypoints_cam = np.zeros((len(cropped_kpts), 3))
        keypoints_cam[:, 0] = cropped_kpts[:, 0]
        keypoints_cam[:, 2] = crop_height - cropped_kpts[:, 1]

        z_center = kpts_3d[0, 2]
        if z_center > 0:
            focal_length_scaled = focal_length * resolution_config.scale_x
            for i in range(len(kpts_3d)):
                z_point = kpts_3d[i, 2]
                keypoints_cam[i, 1] = (z_point - z_center) * focal_length_scaled / z_center
        else:
            keypoints_cam[:, 1] = 0

        keypoints_cam_flat = []
        for i in range(len(keypoints_cam)):
            keypoints_cam_flat.extend([
                float(keypoints_cam[i, 0]),
                float(keypoints_cam[i, 1]),
                float(keypoints_cam[i, 2]),
                int(visibility[i]),
            ])

        # Number of visible keypoints (visibility == 2 only)
        num_keypoints = int((visibility == 2).sum())

        body_bbox = data["body_bbox"]
        segmentation_rle = data["segmentation_rle"]
        mask_decoded = data["mask_decoded"]  # clothed amodal mask of this person (crop resolution)

        # Head bbox (all 11 points incl. out of frame, square x1.2, not clamped)
        head_bbox = compute_head_bbox_cropped(cropped_kpts)

        # Head ignore flag: head bbox too small or head completely out of frame
        head_area = head_bbox[2] * head_bbox[3]
        should_ignore = (head_area < MIN_HEAD_BBOX_AREA) or (not data["head_in_frame"])
        if should_ignore:
            ignore_count += 1

        # ---- Body keypoints ----
        cropped_body_kpts = data["cropped_body_kpts"]
        body_kpts_3d = data["body_kpts_3d"]

        # COCO-style 2D (x, y, v)
        body_keypoints_flat = []
        for i in range(len(cropped_body_kpts)):
            body_keypoints_flat.extend([
                float(cropped_body_kpts[i, 0]),
                float(cropped_body_kpts[i, 1]),
                int(body_visibility[i]),
            ])

        # Unified (x, y, z, v): x=image X, y=pelvis-relative pseudo depth, z=flipped image Y
        pelvis_z = float(body_kpts_3d[BODY_PELVIS_INDEX, 2])
        body_cam = np.zeros((len(cropped_body_kpts), 3))
        body_cam[:, 0] = cropped_body_kpts[:, 0]
        body_cam[:, 2] = crop_height - cropped_body_kpts[:, 1]
        if pelvis_z > 0:
            focal_length_scaled = focal_length * resolution_config.scale_x
            body_cam[:, 1] = (
                (body_kpts_3d[:, 2] - pelvis_z) * focal_length_scaled / pelvis_z
            )
        body_cam_flat = []
        for i in range(len(body_cam)):
            body_cam_flat.extend([
                float(body_cam[i, 0]),
                float(body_cam[i, 1]),
                float(body_cam[i, 2]),
                int(body_visibility[i]),
            ])

        # Body reference bbox (all 31 points incl. out of frame, square x1.2, not clamped)
        body_ref_bbox = compute_ref_bbox_cropped(
            cropped_body_kpts, BODY_BBOX_EXPAND_RATIO
        )
        body_ignore = (
            body_ref_bbox[2] * body_ref_bbox[3] < MIN_BODY_BBOX_AREA
            or not data["body_in_frame"]
        )

        # ---- Part masks and depth teacher (naked SMPL-X render, amodal) ----
        part_masks = None
        depth_teacher = None
        solo = cache_p.get("solo")
        if solo is not None:
            label_t, depth_t = sample_maps_for_crop(
                solo, cx, cy, angle_deg, crop_width, crop_height, TEACHER_MAP_SCALE
            )
            if (label_t > 0).any():
                part_masks = {}
                for gid, gname in enumerate(PART_GROUP_NAMES, start=1):
                    part_masks[gname] = _encode_binary_rle(label_t == gid)
                # Fill the depth into every pixel of the clothed mask (ch0) so that
                # hair / clothing pixels get the nearest body-surface depth
                if mask_decoded is not None:
                    clothed_t = cv2.resize(
                        mask_decoded,
                        (depth_t.shape[1], depth_t.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    depth_t = fill_depth_into_clothed(depth_t, clothed_t)
                # Stored as uint16 millimetres (0 = background); extracter writes the PNG
                depth_teacher = np.clip(
                    depth_t * 1000.0, 0, 65535
                ).astype(np.uint16)

        # ---- Occlusion teacher (clothed amodal minus visible, from the integrated mask) ----
        occlusion_rle = None
        occlusion_valid = False
        vis_ratio_v = cache_p.get("vis_ratio")
        occlusion_pct = cache_p.get("occlusion_pct")
        occ_src = cache_p.get("occ")
        if occ_src is not None and vis_ratio_v is not None and occlusion_pct is not None:
            # Fail-safe: invalidate the teacher if the visible ratio disagrees
            # with the pkl occlusion percentage
            occlusion_valid = (
                abs((1.0 - vis_ratio_v) * 100.0 - occlusion_pct)
                <= OCCLUSION_MISMATCH_TOLERANCE
            )
            if occlusion_valid:
                occlusion_rle = compute_rotated_cropped_mask_rle_from_array(
                    occ_src, cx, cy, angle_deg, crop_width, crop_height
                )  # None if no occluded pixel is inside the crop (valid: no occlusion)

        if body_bbox is None:
            body_bbox = head_bbox.copy()

        annotation = {
            "id": current_anno_id,
            "image_id": image_id,
            "category_id": 1,
            "keypoints": keypoints_flat,
            "keypoints_cam": keypoints_cam_flat,
            "num_keypoints": num_keypoints,
            "segmentation": segmentation_rle,
            "bbox": body_bbox,
            "head_bbox": head_bbox,
            "area": body_bbox[2] * body_bbox[3],
            "iscrowd": 0,
            "ignore": should_ignore,
            "keypoints_body": body_keypoints_flat,
            "keypoints_body_cam": body_cam_flat,
            "num_body_keypoints": int((body_visibility == 2).sum()),
            "body_ref_bbox": body_ref_bbox,
            "body_ignore": body_ignore,
            "pelvis_depth": pelvis_z,
            "part_masks": part_masks,
            "occlusion_mask": occlusion_rle,
            "occlusion_valid": occlusion_valid,
            "subject_id": cache_p.get("subject_id"),
            "kid": cache_p.get("kid"),
            "occlusion_pct": occlusion_pct,
            "_depth_teacher": depth_teacher,  # replaced by depth_file once extracter writes the PNG
        }
        crop_annotations.append(annotation)
        current_anno_id += 1

    # Is there at least one valid person?
    valid_person_count = sum(1 for ann in crop_annotations if not ann["ignore"])

    # No valid person: retry with a given probability
    if force_valid_person and valid_person_count == 0 and random.random() < NO_VALID_PERSON_SKIP_PROB:
        return CropResult(success=False, skip_reason="no_valid_person")

    return CropResult(
        success=True,
        cropped_img=cropped_img,
        image_info=image_info,
        annotations=crop_annotations,
        ignore_count=ignore_count,
    )


def select_crop_params(force_rotation: bool = False) -> tuple[int, int, float]:
    """
    Pick random crop parameters.

    Args:
        force_rotation: always apply a rotation

    Returns:
        (crop_width, crop_height, angle_deg)
    """
    valid_sizes = resolution_config.get_valid_crop_sizes()
    crop_size = random.choice(valid_sizes)
    crop_width = crop_size["width"]
    crop_height = crop_size["height"]

    # Rotation angle (applied with probability ROTATION_CHANCE)
    if force_rotation or random.random() < ROTATION_CHANCE:
        angle_deg = random.uniform(ROTATION_ANGLE_MIN, ROTATION_ANGLE_MAX)
        if random.random() < 0.5:
            angle_deg = -angle_deg
    else:
        angle_deg = 0.0

    return crop_width, crop_height, angle_deg


def run_ins_inference_and_match(
    result: CropResult,
    rtmdet_ins: "RTMDetInsInference",
    iou_threshold: float = MATCHING_IOU_THRESHOLD,
    score_threshold: float = MATCHING_SCORE_THRESHOLD,
    min_coverage: float = MATCHING_MIN_COVERAGE,
) -> CropResult:
    """
    Run RTMDet-Ins on the crop and match its masks to the GT masks.

    Args:
        result: result of try_single_crop
        rtmdet_ins: RTMDet-Ins instance
        iou_threshold: IoU threshold
        score_threshold: matching score threshold
        min_coverage: minimum coverage

    Returns:
        CropResult with ins_masks, match_results and per-annotation matched_ins_id set
    """
    if not result.success or result.cropped_img is None:
        return result

    if rtmdet_ins is None:
        return result

    try:
        detections = rtmdet_ins.infer(result.cropped_img)
    except Exception:
        return result

    if not detections:
        return result

    ins_masks = [det['mask'] for det in detections if det.get('mask') is not None]
    result.ins_masks = ins_masks  # kept for display

    if not ins_masks:
        return result

    # GT masks
    gt_masks = []
    anno_idx_to_gt_idx = {}
    gt_idx_to_anno_idx = {}

    for anno_idx, anno in enumerate(result.annotations):
        if "segmentation" in anno and anno["segmentation"] is not None:
            try:
                gt_mask = decode_rle_mask(anno["segmentation"])
                gt_idx = len(gt_masks)
                gt_masks.append(gt_mask)
                anno_idx_to_gt_idx[anno_idx] = gt_idx
                gt_idx_to_anno_idx[gt_idx] = anno_idx
            except Exception:
                pass

    if not gt_masks:
        return result

    # Matching
    match_results = match_masks(
        gt_masks=gt_masks,
        ins_masks=ins_masks,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        min_coverage=min_coverage,
    )

    # GT -> Ins lookup
    gt_to_ins_map = {}
    for match_result in match_results:
        if match_result.matched:
            gt_to_ins_map[match_result.gt_idx] = match_result.ins_idx

    # Set matched_ins_id on every annotation
    for anno_idx, anno in enumerate(result.annotations):
        if anno_idx in anno_idx_to_gt_idx:
            gt_idx = anno_idx_to_gt_idx[anno_idx]
            if gt_idx in gt_to_ins_map:
                anno["matched_ins_id"] = gt_to_ins_map[gt_idx]
            else:
                anno["matched_ins_id"] = None
        else:
            anno["matched_ins_id"] = None

    # Convert gt_idx in match_results to annotation indices (display)
    for match_result in match_results:
        if match_result.matched and match_result.gt_idx in gt_idx_to_anno_idx:
            match_result.gt_idx = gt_idx_to_anno_idx[match_result.gt_idx]

    result.match_results = match_results

    return result


def check_no_ins_matched_persons(result: CropResult) -> bool:
    """
    Check whether RTMDet-Ins matched no person at all.

    Args:
        result: CropResult after run_ins_inference_and_match

    Returns:
        True: no matched person (no detection or no match)
        False: at least one person matched
    """
    if not result.success:
        return True

    matched_count = 0
    for anno in result.annotations:
        matched_ins_id = anno.get("matched_ins_id")
        if matched_ins_id is not None:
            matched_count += 1

    return matched_count == 0
