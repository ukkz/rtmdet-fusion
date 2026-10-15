#!/usr/bin/env python3
"""
Settings for the AGORA extraction tools.

Central place for constants, paths and extraction parameters.
"""

import sys
from pathlib import Path

# Make configs.default importable
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Shared settings from configs/default.py
from configs.default import (
    PROJECT_ROOT,
    AGORA_SOURCE_DIR,
    DATA_DIR,
    # Matching thresholds (re-exported)
    MATCHING_IOU_THRESHOLD,  # noqa: F401
    MATCHING_SCORE_THRESHOLD,  # noqa: F401
    MATCHING_MIN_COVERAGE,  # noqa: F401
)

# ============================================================
# Paths
# ============================================================

# AGORA source data (default; overridable with --agora-dir, see ResolutionConfig)
AGORA_DIR = AGORA_SOURCE_DIR

# Output (data/train/, data/validation/ and the annotation JSON files)
CROPPED_DIR = DATA_DIR
OUTPUT_DIR = DATA_DIR


# ============================================================
# Resolution
# ============================================================

# Native resolution (reference frame of the SMPL-X projections)
ORIG_SIZE = (3840, 2160)

# Camera principal point (native resolution)
CX, CY = ORIG_SIZE[0] / 2, ORIG_SIZE[1] / 2


# ============================================================
# Crop sizes (width x height), sampled uniformly
# ============================================================

CROP_SIZES = [
    # 1:1
    {"width": 512, "height": 512},
    {"width": 640, "height": 640},
    {"width": 720, "height": 720},
    {"width": 800, "height": 800},
    {"width": 960, "height": 960},
    {"width": 1024, "height": 1024},
    {"width": 1280, "height": 1280},
    {"width": 1440, "height": 1440},
    {"width": 1600, "height": 1600},
    # 4:3
    {"width": 640, "height": 480},
    {"width": 854, "height": 640},
    {"width": 960, "height": 720},
    {"width": 1024, "height": 768},
    {"width": 1280, "height": 960},
    {"width": 1440, "height": 1080},
    {"width": 1600, "height": 1200},
    # 3:4
    {"width": 480, "height": 640},
    {"width": 640, "height": 854},
    {"width": 720, "height": 960},
    {"width": 768, "height": 1024},
    {"width": 960, "height": 1280},
    {"width": 1080, "height": 1440},
    {"width": 1200, "height": 1600},
    # 16:9
    {"width": 854, "height": 480},
    {"width": 1138, "height": 640},
    {"width": 1280, "height": 720},
    {"width": 1422, "height": 800},
    {"width": 1706, "height": 960},
    # 9:16
    {"width": 480, "height": 854},
    {"width": 640, "height": 1138},
    {"width": 720, "height": 1280},
    {"width": 800, "height": 1422},
    {"width": 960, "height": 1706},
]

# Crops per source image (each crop is saved as original + mirror)
CROPS_PER_IMAGE = 1


# ============================================================
# Filters
# ============================================================

# Random seed (reproducibility)
RANDOM_SEED = 42

# Minimum head bbox area (px^2); roughly below 32x32 is marked ignore
MIN_HEAD_BBOX_AREA = 1000


# ============================================================
# Matching thresholds
# ============================================================
# Imported from configs/default.py:
# - MATCHING_IOU_THRESHOLD = 0.3
# - MATCHING_SCORE_THRESHOLD = 0.4
# - MATCHING_MIN_COVERAGE = 0.7


# ============================================================
# Extraction probabilities
# ============================================================

# Probability of retrying a crop that contains no valid person
NO_VALID_PERSON_SKIP_PROB = 1.00

# Probability of retrying a crop in which RTMDet-Ins matches no person
# (no detection, or detections that match no GT mask)
NO_INS_MATCH_SKIP_PROB = 1.00

# Probability of applying a rotation
ROTATION_CHANCE = 0.25


# ============================================================
# Extraction parameters
# ============================================================

# Maximum retries per crop (afterwards the crop is skipped)
MAX_CROP_RETRIES = 100

# Rotation angle range (degrees, random sign)
ROTATION_ANGLE_MIN = 5
ROTATION_ANGLE_MAX = 55


# ============================================================
# SMPL-X joints
# ============================================================

JOINT_INDICES = {
    "neck": 12,
    "head": 15,
    "jaw": 22,
    "nose": 55,
    "right_eye": 56,
    "left_eye": 57,
    "right_ear": 58,
    "left_ear": 59,
}

# Head keypoints (11) and their skeleton are defined in configs/default.py.
# How each point is derived from SMPL-X (see utils/keypoints.py):
#   0 center    joint 15
#   1 top       vertex nearest to the neck->center segment extended 3x
#   2 left_ear  joint 59        3 right_ear  joint 58
#   4 left_eye  joint 57        5 right_eye  joint 56
#   6 nose      joint 55
#   7 chin      vertex nearest to the 1/4-3/4 section of the nose-neck segment
#   8 occiput   vertex nearest to the segment from (nose->center extended 3x) to center
#   9 vertex    vertex nearest to the segment from
#               center + 3 * ((top - center) + (occiput - center)) to center
#  10 neck      joint 12
from configs.default import KEYPOINT_NAMES, SKELETON  # noqa: E402, F401


# ============================================================
# Body keypoints (31)
# ============================================================
# Defined in configs/default.py (shared with the model); re-exported here.

from configs.default import (  # noqa: E402, F401
    BODY_KEYPOINT_DEFS,
    BODY_KEYPOINT_NAMES,
    BODY_KEYPOINT_SMPLX_INDICES,
    NUM_BODY_KEYPOINTS,
    BODY_PELVIS_INDEX,
    BODY_KEYPOINT_SWAP_PAIRS,
    BODY_SKELETON,
)

# GT body reference bbox: bounding square of all 31 points x this ratio
BODY_BBOX_EXPAND_RATIO = 1.2

# Minimum body bbox area (px^2); smaller persons get body_ignore
MIN_BODY_BBOX_AREA = 3000


# ============================================================
# SMPL-X assets and rendering
# ============================================================

# SMPL-X model (face topology) and per-vertex part segmentation.
# Not distributed with this repository; obtain them under their own licenses.
SMPLX_MODEL_PATH = PROJECT_ROOT / "weights" / "smplx" / "SMPLX_NEUTRAL.npz"
SMPLX_VERT_SEG_PATH = PROJECT_ROOT / "weights" / "smplx" / "smplx_vert_segmentation.json"

# Part groups (27 parts of smplx_vert_segmentation -> 4 groups).
# Mask channels: ch1 = whole naked body (label > 0), ch2 = head, ch3 = torso,
# ch4 = limbs, ch5 = hands
PART_GROUPS = {
    "head": ["head", "leftEye", "rightEye", "eyeballs"],
    "torso": ["neck", "spine", "spine1", "spine2", "hips",
              "leftShoulder", "rightShoulder"],
    "limbs": ["leftArm", "leftForeArm", "rightArm", "rightForeArm",
              "leftUpLeg", "leftLeg", "leftFoot", "leftToeBase",
              "rightUpLeg", "rightLeg", "rightFoot", "rightToeBase"],
    "hands": ["leftHand", "leftHandIndex1", "rightHand", "rightHandIndex1"],
}
PART_GROUP_IDS = {"head": 1, "torso": 2, "limbs": 3, "hands": 4}
PART_GROUP_NAMES = ["head", "torso", "limbs", "hands"]  # order of labels 1..4

# Mirroring swaps no part labels (left and right share a group)

# Maximum side of the per-person render (larger persons are rendered downscaled)
RENDER_MAX_SIDE = 512

# Teacher map resolution: crop resolution / TEACHER_MAP_SCALE
TEACHER_MAP_SCALE = 4

# Occlusion teacher fail-safe: allowed gap (percentage points) between the
# occlusion ratio derived from the integrated mask and the pkl occlusion value
OCCLUSION_MISMATCH_TOLERANCE = 20.0


# ============================================================
# Camera focal length per scene
# ============================================================

SCENE_FOCAL_LENGTHS = {
    "archviz": 1920.0,
    "hdri": 5333.3,
    # everything else uses DEFAULT_FOCAL_LENGTH
}
DEFAULT_FOCAL_LENGTH = 2986.7


# ============================================================
# Per-process source data state
# ============================================================

class ResolutionConfig:
    """Source data settings: AGORA location and source resolution.

    Module-level state is not inherited by spawned worker processes, so every
    process (main and each worker) must call init() with explicit values.
    """

    def __init__(self):
        # Defaults (4K)
        self.agora_dir = Path(AGORA_DIR)
        self.resolution = "4K"
        self.source_width = 3840
        self.source_height = 2160
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.images_dir = "images4K"
        self.masks_dir = "masks4K"
        self.mask_suffix = ""
        self.image_suffix = ""

    def init(self, resolution: str, agora_dir: Path) -> None:
        """
        Initialize the settings for this process.

        Args:
            resolution: "4K" or "720p"
            agora_dir: AGORA dataset root
        """
        self.agora_dir = Path(agora_dir)
        self.resolution = resolution

        if resolution == "4K":
            self.source_width = 3840
            self.source_height = 2160
            self.scale_x = 1.0
            self.scale_y = 1.0
            self.images_dir = "images4K"
            self.masks_dir = "masks4K"
            self.image_suffix = ""
            self.mask_suffix = ""
        else:  # 720p
            self.source_width = 1280
            self.source_height = 720
            self.scale_x = 1280 / 3840  # 1/3
            self.scale_y = 720 / 2160   # 1/3
            self.images_dir = "images"
            self.masks_dir = "masks"
            self.image_suffix = "_1280x720"
            self.mask_suffix = "_1280x720"

    def get_valid_crop_sizes(self) -> list:
        """Return only the crop sizes that fit in the current source resolution"""
        return [
            cs for cs in CROP_SIZES
            if cs["width"] <= self.source_width and cs["height"] <= self.source_height
        ]


# Global instance
resolution_config = ResolutionConfig()


# ============================================================
# Display
# ============================================================

# Display padding (to show keypoints outside the image)
DISPLAY_PADDING = 100

# Head keypoint colours of the dataset viewers. Kept local (and different from
# configs.default.KEYPOINT_COLORS_BGR used by the model viewer) so that the BGR
# and RGB tables below stay in sync.
KEYPOINT_COLORS_BGR = [
    (255, 0, 0),     # center - blue
    (0, 255, 0),     # top - green
    (255, 0, 255),   # left_ear - magenta
    (200, 0, 200),   # right_ear
    (255, 255, 0),   # left_eye - cyan
    (200, 200, 0),   # right_eye
    (0, 0, 255),     # nose - red
    (0, 128, 255),   # chin - orange
    (0, 200, 0),     # occiput - dark green
    (128, 128, 128), # vertex - grey
    (0, 255, 255),   # neck - yellow
]

# Same colours as RGB in [0, 1] (matplotlib)
KEYPOINT_COLORS_RGB = [
    (0, 0, 1),       # center - blue
    (0, 1, 0),       # top - green
    (1, 0, 1),       # left_ear - magenta
    (0.8, 0, 0.8),   # right_ear
    (0, 1, 1),       # left_eye - cyan
    (0, 0.8, 0.8),   # right_eye
    (1, 0, 0),       # nose - red
    (1, 0.5, 0),     # chin - orange
    (0, 0.8, 0),     # occiput - dark green
    (0.5, 0.5, 0.5), # vertex - grey
    (1, 1, 0),       # neck - yellow
]

# Body skeleton and keypoint colours come from configs/default.py
from configs.default import (  # noqa: E402, F401
    BODY_KEYPOINT_COLORS_BGR,
    BODY_KEYPOINT_COLORS_RGB,
    PERSON_COLORS_BGR,
)

# Part mask colours (BGR): label 1=head, 2=torso, 3=limbs, 4=hands
PART_COLORS_BGR = {
    1: (0, 0, 255),     # head: red
    2: (0, 200, 0),     # torso: green
    3: (255, 128, 0),   # limbs: blue
    4: (0, 255, 255),   # hands: yellow
}
