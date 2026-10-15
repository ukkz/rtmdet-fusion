"""RTMDet-Fusion datasets"""

from .keypoint_dataset import KeypointDataset, collate_fn

__all__ = [
    'KeypointDataset',
    'collate_fn',
]
