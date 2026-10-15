#!/usr/bin/env python3
"""
Path and file name utilities.

Builds AGORA image / mask paths and looks up the focal length per scene.
All paths are resolved against resolution_config.agora_dir.
"""

import sys
from pathlib import Path
from typing import Optional

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from config import (
    SCENE_FOCAL_LENGTHS,
    DEFAULT_FOCAL_LENGTH,
    resolution_config,
)


def get_focal_length(img_path: str) -> float:
    """Extract the scene name from an image path and return its focal length"""
    # e.g. ag_validationset_renderpeople_bfh_hdri_50mm_5_15_00001.png (validation)
    # e.g. ag_trainset_renderpeople_body_hdri_50mm_5_10_00013.png (train)
    # e.g. ag_trainset_renderpeople_bfh_archviz_5_10_cam02_01060.png
    parts = img_path.split("_")

    # The element after "bfh" or "body" is the scene name
    for prefix in ["bfh", "body"]:
        try:
            idx = parts.index(prefix)
            scene = parts[idx + 1]
            return SCENE_FOCAL_LENGTHS.get(scene, DEFAULT_FOCAL_LENGTH)
        except (ValueError, IndexError):
            continue

    return DEFAULT_FOCAL_LENGTH


def get_image_path(img_name: str, split: str, subdir: Optional[int] = None) -> Path:
    """Return the source image path"""
    agora_dir = resolution_config.agora_dir
    stem = Path(img_name).stem
    img_filename = f"{stem}{resolution_config.image_suffix}.png"
    if split == "train" and subdir is not None:
        return agora_dir / split / resolution_config.images_dir / str(subdir) / img_filename
    return agora_dir / split / resolution_config.images_dir / img_filename


def get_mask_path(img_name: str, person_idx: int, split: str) -> Optional[Path]:
    """
    Return the mask file path for an image and person index.

    Args:
        img_name: image file name (e.g. ag_trainset_renderpeople_body_hdri_50mm_5_10_00013.png)
        person_idx: person index (0-based); -1 selects the integrated mask (_00000)
        split: dataset split (train/validation)

    Returns:
        Path to the mask file, or None if not found
    """
    stem = Path(img_name).stem
    parts = stem.split("_")

    # The frame number is the last element (5 digits)
    frame_num = parts[-1]
    frame_num_6digit = frame_num.zfill(6)

    # Build the scene prefix: up to camXX if present, otherwise drop the frame number.
    # Image file name patterns:
    #   body: ag_trainset_renderpeople_body_hdri_50mm_5_10_00013
    #   bfh:  ag_trainset_renderpeople_bfh_archviz_5_10_cam02_00000
    cam_id = None
    if parts[-2].startswith("cam"):
        # bfh style: keep the camera ID
        cam_id = parts[-2]
        full_prefix = "_".join(parts[:-2])
    else:
        # body style
        full_prefix = "_".join(parts[:-1])

    # Mask directory name = prefix without the parameter part that follows the scene name
    # e.g. ag_trainset_renderpeople_body_hdri_50mm_5_10 -> ag_trainset_renderpeople_body_hdri
    # e.g. ag_trainset_renderpeople_bfh_archviz_5_10 -> ag_trainset_renderpeople_bfh_archviz
    dir_parts = full_prefix.split("_")
    if "bfh" in dir_parts:
        bfh_idx = dir_parts.index("bfh")
        dir_name = "_".join(dir_parts[:bfh_idx + 2])
    elif "body" in dir_parts:
        body_idx = dir_parts.index("body")
        dir_name = "_".join(dir_parts[:body_idx + 2])
    else:
        dir_name = full_prefix

    mask_dir = resolution_config.agora_dir / split / resolution_config.masks_dir / dir_name

    if not mask_dir.exists():
        return None

    # Mask file name. _00000 is the integrated mask of all persons;
    # individual persons start at _00001.
    mask_person_idx = person_idx + 1
    person_idx_5digit = str(mask_person_idx).zfill(5)

    if cam_id is not None:
        # bfh style: {full_prefix}_mask_{cam}_{frame_6digit}_{person_5digit}{suffix}.png
        mask_filename = f"{full_prefix}_mask_{cam_id}_{frame_num_6digit}_{person_idx_5digit}{resolution_config.mask_suffix}.png"
    else:
        # body style: {full_prefix}_mask_{frame_6digit}_{person_5digit}{suffix}.png
        mask_filename = f"{full_prefix}_mask_{frame_num_6digit}_{person_idx_5digit}{resolution_config.mask_suffix}.png"

    mask_path = mask_dir / mask_filename
    if mask_path.exists():
        return mask_path

    return None


def find_image_subdir(img_name: str, split: str) -> Optional[int]:
    """Find the numbered subdirectory (0-9) holding a train image"""
    if split != "train":
        return None

    for subdir in range(10):
        img_path = get_image_path(img_name, split, subdir)
        if img_path.exists():
            return subdir
    return None
