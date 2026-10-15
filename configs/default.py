"""RTMDet-Fusion default configuration"""

import os
from pathlib import Path

# ==================================================
# Paths
# ==================================================

# Project root (the rtmdet-fusion checkout)
PROJECT_ROOT = Path(__file__).parent.parent

# Root of the AGORA download (source data for tools/agora/).
# Override with the AGORA_DIR environment variable.
AGORA_SOURCE_DIR = Path(os.environ.get("AGORA_DIR", PROJECT_ROOT / "datasets" / "agora"))

# Generated training data (output of tools/agora/extracter.py)
DATA_DIR = PROJECT_ROOT / "data"

# Training annotations and images
TRAIN_ANN_PATH = DATA_DIR / "head_keypoints_train.json"
VAL_ANN_PATH = DATA_DIR / "head_keypoints_validation.json"
TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
VAL_IMG_DIR = DATA_DIR / "validation" / "images"

# Pretrained RTMDet-Ins weights
RTMDET_INS_TINY_WEIGHTS = PROJECT_ROOT / "weights" / "rtmdet-ins_tiny.pth"
RTMDET_INS_S_WEIGHTS = PROJECT_ROOT / "weights" / "rtmdet-ins_s.pth"

# Working directory (checkpoints, TensorBoard logs)
WORK_DIR = PROJECT_ROOT / "work_dirs" / "main"

# ==================================================
# Head keypoints (11 points)
# ==================================================

NUM_KEYPOINTS = 11

# Keypoint names (index order)
KEYPOINT_NAMES = [
    'center',     # 0: head center (reference point of the pseudo-depth axis)
    'top',        # 1: top of the head
    'left_ear',   # 2
    'right_ear',  # 3
    'left_eye',   # 4
    'right_eye',  # 5
    'nose',       # 6
    'chin',       # 7
    'occiput',    # 8: back of the head
    'vertex',     # 9: crown, between top and occiput
    'neck',       # 10
]

# Skeleton connections (drawing)
SKELETON = [
    (1, 0),   # top -> center
    (0, 2),   # center -> left_ear
    (0, 3),   # center -> right_ear
    (4, 5),   # left_eye -> right_eye
    (6, 0),   # nose -> center
    (7, 6),   # chin -> nose
    (0, 8),   # center -> occiput
    (1, 9),   # top -> vertex
    (9, 8),   # vertex -> occiput
    (0, 9),   # center -> vertex
    (0, 10),  # center -> neck
]

# ==================================================
# Body keypoints (31 points)
# ==================================================
# The first 17 points follow the COCO-17 order. The pseudo-depth reference is pelvis (index 17).
# Each entry: (name, index in the 127-joint SMPL-X layout).
# This file is the single source of truth; tools/agora/config.py imports from here.

BODY_KEYPOINT_DEFS = [
    # --- COCO-17 compatible part (0-16) ---
    ("nose", 55),
    ("left_eye", 57),
    ("right_eye", 56),
    ("left_ear", 59),
    ("right_ear", 58),
    ("left_shoulder", 16),
    ("right_shoulder", 17),
    ("left_elbow", 18),
    ("right_elbow", 19),
    ("left_wrist", 20),
    ("right_wrist", 21),
    ("left_hip", 1),
    ("right_hip", 2),
    ("left_knee", 4),
    ("right_knee", 5),
    ("left_ankle", 7),
    ("right_ankle", 8),
    # --- Extension (17-30) ---
    ("pelvis", 0),        # 17: pseudo-depth reference point
    ("spine2", 6),        # 18
    ("neck", 12),         # 19
    ("head", 15),         # 20: same joint as the head keypoint 'center'
    ("left_collar", 13),  # 21
    ("right_collar", 14), # 22
    ("left_big_toe", 60),  # 23
    ("right_big_toe", 63), # 24
    ("left_heel", 62),     # 25
    ("right_heel", 65),    # 26
    ("left_thumb_tip", 66),   # 27
    ("right_thumb_tip", 71),  # 28
    ("left_middle_tip", 68),  # 29
    ("right_middle_tip", 73), # 30
]

BODY_KEYPOINT_NAMES = [name for name, _ in BODY_KEYPOINT_DEFS]
BODY_KEYPOINT_SMPLX_INDICES = [idx for _, idx in BODY_KEYPOINT_DEFS]
NUM_BODY_KEYPOINTS = len(BODY_KEYPOINT_DEFS)  # 31
BODY_PELVIS_INDEX = 17  # position of pelvis in the output array

# Left/right pairs swapped on horizontal mirroring (output array indices)
BODY_KEYPOINT_SWAP_PAIRS = [
    (1, 2),    # eye
    (3, 4),    # ear
    (5, 6),    # shoulder
    (7, 8),    # elbow
    (9, 10),   # wrist
    (11, 12),  # hip
    (13, 14),  # knee
    (15, 16),  # ankle
    (21, 22),  # collar
    (23, 24),  # big_toe
    (25, 26),  # heel
    (27, 28),  # thumb_tip
    (29, 30),  # middle_tip
]

# Body skeleton (drawing and bone losses)
BODY_SKELETON = [
    # Face (COCO part)
    [0, 1], [0, 2], [1, 3], [2, 4],
    # Arms
    [5, 7], [7, 9], [6, 8], [8, 10],
    # Torso (COCO part)
    [5, 6], [11, 12], [5, 11], [6, 12],
    # Legs
    [11, 13], [13, 15], [12, 14], [14, 16],
    # Extension: pelvis, spine, neck, head
    [17, 11], [17, 12], [17, 18], [18, 19], [19, 20], [0, 20],
    # Extension: collarbones
    [19, 21], [19, 22], [21, 5], [22, 6],
    # Extension: feet
    [15, 23], [15, 25], [16, 24], [16, 26],
    # Extension: fingers
    [9, 27], [9, 29], [10, 28], [10, 30],
]

# Bone links for the bone losses (same as the skeleton)
BODY_BONE_LINKS = [tuple(pair) for pair in BODY_SKELETON]


def _body_kpt_color_bgr(name: str) -> tuple:
    """Display color of a body keypoint (left = orange, right = green, center = cyan/special)"""
    if name in ("nose", "head"):
        return (0, 0, 255)      # red
    if name == "pelvis":
        return (255, 255, 255)  # white (reference point)
    if name.startswith("left_"):
        if "tip" in name:
            return (0, 100, 255)   # dark orange (fingertips)
        return (0, 165, 255)       # orange
    if name.startswith("right_"):
        if "tip" in name:
            return (0, 200, 0)     # dark green (fingertips)
        return (80, 255, 80)       # green
    return (255, 255, 0)           # cyan (center: spine2, neck)


BODY_KEYPOINT_COLORS_BGR = [_body_kpt_color_bgr(n) for n in BODY_KEYPOINT_NAMES]
BODY_KEYPOINT_COLORS_RGB = [
    (c[2] / 255.0, c[1] / 255.0, c[0] / 255.0) for c in BODY_KEYPOINT_COLORS_BGR
]

# ==================================================
# Model
# ==================================================

# Input image size
IMG_SIZE = 640

# Maximum number of instances (fixed slot count of the exported outputs)
MAX_INSTANCES = 5

# Expansion applied to a reference bbox before keypoint decoding.
# 1.0 = none (annotations already include the margin)
BBOX_EXPAND_RATIO = 1.0

# FPN strides
FPN_STRIDES = [8, 16, 32]

# Grid size of the per-instance bbox feature sampling
# (G x G points; 14 x 14 = 196 points makes use of the P3/P4 resolution)
GRID_SIZE = 14

# SimCC bins of the head keypoints (coordinate classification resolution).
# 64 bins over [-BIN_MARGIN, 1 + BIN_MARGIN]: ~0.02 spacing, about 2 px for a 100 px head bbox.
# The training target (sigma=0.03) is a Gaussian roughly 4 bins wide.
NUM_BINS = 64

# SimCC bins of the body keypoints (twice the head; the body reference bbox is
# 3-4 times larger in pixels than the head bbox)
NUM_BINS_BODY = 128

# Margin bins: the bin range is extended to [-BIN_MARGIN, 1 + BIN_MARGIN] so keypoints
# slightly outside the reference bbox can be represented (head and body)
BIN_MARGIN = 0.125

# Prediction range of the body reference bbox head
# (Ins bbox center +/- BODY_BBOX_RANGE_SCALE / 2 x ins_size).
# For people cut off by the frame the body reference bbox extends beyond the Ins bbox
# (the visible part) and outside the image, so the range is wider than the head (1.25).
BODY_BBOX_RANGE_SCALE = 2.0

# GT body reference bbox used for training: the bounding rectangle of all 31 keypoints
# x BODY_REF_BBOX_EXPAND, built in the dataset (not squared; a square box around a
# standing person makes nearly half of the grid features background).
# The square body_ref_bbox stored in the annotations is not used for training.
BODY_REF_BBOX_EXPAND = 1.1

# Weight of the coordinate losses for out-of-frame (visibility=0) keypoints.
# AGORA provides true coordinates even outside the frame, so they are supervised with a
# low weight; whether a point is in frame is learned separately by the in_frame head.
OOF_COORD_WEIGHT = 0.3

# RTMDet detection score threshold for training-time matching.
# Lower than INS_SCORE_THR so that distant or occluded people are also used for training;
# the Hungarian matching (IoU/Coverage) acts as the quality gate.
TRAIN_INS_SCORE_THR = 0.05

# Resolutions for multi-scale training (multiples of 32)
MULTISCALE_SIZES = [256, 320, 416, 512, 640]

# ==================================================
# Training
# ==================================================

# Default training parameters (batch size 48, frozen RTMDet-Ins backbone)
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 48
DEFAULT_LR = 5e-4
WARMUP_EPOCHS = 5  # learning rate warmup
WEIGHT_DECAY = 0.05

# Person class ID (COCO)
PERSON_CLASS_ID = 0

# ==================================================
# Mask matching (Hungarian)
# ==================================================

# IoU threshold (match unconditionally at or above this)
MATCHING_IOU_THRESHOLD = 0.3

# Combined score threshold (used below the IoU threshold)
MATCHING_SCORE_THRESHOLD = 0.4

# Minimum coverage (used below the IoU threshold to rescue a match)
# Coverage = Intersection / Ins_area (how much of the Ins mask lies inside the GT)
MATCHING_MIN_COVERAGE = 0.7

# ==================================================
# RTMDet-Ins detection
# ==================================================

# Instance detection score threshold (RTMDet-Ins test_cfg)
INS_SCORE_THR = 0.3

# Minimum bbox size (pixels)
INS_MIN_BBOX_SIZE = 100

# ==================================================
# RTMDet-Ins model configs
# ==================================================

# --- tiny ---
RTMDET_INS_TINY_CFG = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=80,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss',
            loss_weight=2.0,
            eps=5e-6,
            reduction='mean'
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=INS_MIN_BBOX_SIZE,
        score_thr=INS_SCORE_THR,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5
    )
)

# --- s ---
RTMDET_INS_S_CFG = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.33,
        widen_factor=0.5,
        channel_attention=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[128, 256, 512],
        out_channels=128,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU', inplace=True)
    ),
    bbox_head=dict(
        type='RTMDetInsSepBNHead',
        num_classes=80,
        in_channels=128,
        stacked_convs=2,
        feat_channels=128,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32]
        ),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(
            type='DiceLoss',
            loss_weight=2.0,
            eps=5e-6,
            reduction='mean'
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=INS_MIN_BBOX_SIZE,
        score_thr=INS_SCORE_THR,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100,
        mask_thr_binary=0.5
    )
)

# ==================================================
# Drawing
# ==================================================

# Head keypoint colors (BGR), index order
KEYPOINT_COLORS_BGR = [
    (0, 255, 0),      # center - green (reference point)
    (255, 0, 0),      # top - blue
    (255, 0, 255),    # left_ear - magenta
    (200, 0, 200),    # right_ear
    (255, 255, 0),    # left_eye - cyan
    (200, 200, 0),    # right_eye
    (0, 0, 255),      # nose - red
    (0, 128, 255),    # chin - orange
    (128, 128, 128),  # occiput - gray
    (0, 255, 255),    # vertex - yellow
    (255, 200, 0),    # neck - light blue
]

# Per-person colors (BGR)
PERSON_COLORS_BGR = [
    (0, 255, 0),      # green
    (255, 0, 0),      # blue
    (0, 165, 255),    # orange
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (0, 255, 255),    # yellow
    (128, 0, 255),    # purple
    (255, 128, 0),    # light blue
]

# ==================================================
# RTMDet-Ins variant selection
# ==================================================

RTMDET_VARIANTS = {
    'tiny': {
        'weights': RTMDET_INS_TINY_WEIGHTS,
        'in_channels': 96,  # FPN output channels
    },
    's': {
        'weights': RTMDET_INS_S_WEIGHTS,
        'in_channels': 128,  # FPN output channels
    },
}

def get_rtmdet_config(variant: str = 's') -> dict:
    """Return the RTMDet-Ins model config of a variant

    Args:
        variant: 'tiny' or 's'

    Returns:
        dict: RTMDet config dictionary
    """
    variant = variant.lower()
    if variant not in RTMDET_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(RTMDET_VARIANTS.keys())}")

    if variant == 'tiny':
        return RTMDET_INS_TINY_CFG
    else:
        return RTMDET_INS_S_CFG

def get_rtmdet_weights(variant: str = 's') -> Path:
    """Return the path of the pretrained RTMDet-Ins weights of a variant"""
    variant = variant.lower()
    if variant not in RTMDET_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(RTMDET_VARIANTS.keys())}")
    return RTMDET_VARIANTS[variant]['weights']

def get_rtmdet_in_channels(variant: str = 's') -> int:
    """Return the FPN output channel count of a variant"""
    variant = variant.lower()
    if variant not in RTMDET_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(RTMDET_VARIANTS.keys())}")
    return RTMDET_VARIANTS[variant]['in_channels']
