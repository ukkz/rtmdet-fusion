#!/usr/bin/env python3
"""
Utilities for the AGORA extraction tools.
"""

import sys
from pathlib import Path

# Add the script directory to the import path
_script_dir = Path(__file__).parent.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from utils.geometry import (
    rotate_point_around_center,
    get_rotated_rect_bounding_box,
    get_valid_center_range,
    find_nearest_vertex_to_line,
)
from utils.image import (
    crop_rotated_image_from_array,
    save_image_jpeg,
    get_jpeg_encoder_name,
    mirror_image_array,
    compute_rotated_cropped_bbox_from_array,
    compute_rotated_cropped_mask_rle_from_array,
    mirror_mask_rle,
    GPUImageProcessor,
    get_image_processor_name,
)
from utils.keypoints import (
    project_3d_to_2d,
    compute_head_keypoints,
    compute_body_keypoints,
    transform_keypoints_with_rotation,
    compute_head_bbox_cropped,
    compute_ref_bbox_cropped,
    is_any_keypoint_visible,
)
from utils.path import (
    get_focal_length,
    get_image_path,
    get_mask_path,
    find_image_subdir,
)
from utils.annotation import (
    mirror_annotations,
)
from utils.extraction import (
    CropResult,
    try_single_crop,
    select_crop_params,
    run_ins_inference_and_match,
    check_no_ins_matched_persons,
)
from utils.widgets import (
    ImageWidget,
    Person3DWidget,
    update_3d_views,
    build_info_lines,
)

__all__ = [
    # geometry
    "rotate_point_around_center",
    "get_rotated_rect_bounding_box",
    "get_valid_center_range",
    "find_nearest_vertex_to_line",
    # image
    "crop_rotated_image_from_array",
    "save_image_jpeg",
    "get_jpeg_encoder_name",
    "mirror_image_array",
    "compute_rotated_cropped_bbox_from_array",
    "compute_rotated_cropped_mask_rle_from_array",
    "mirror_mask_rle",
    "GPUImageProcessor",
    "get_image_processor_name",
    # keypoints
    "project_3d_to_2d",
    "compute_head_keypoints",
    "compute_body_keypoints",
    "transform_keypoints_with_rotation",
    "compute_head_bbox_cropped",
    "compute_ref_bbox_cropped",
    "is_any_keypoint_visible",
    # path
    "get_focal_length",
    "get_image_path",
    "get_mask_path",
    "find_image_subdir",
    # annotation
    "mirror_annotations",
    # extraction
    "CropResult",
    "try_single_crop",
    "select_crop_params",
    "run_ins_inference_and_match",
    "check_no_ins_matched_persons",
    # widgets
    "ImageWidget",
    "Person3DWidget",
    "update_3d_views",
    "build_info_lines",
]
