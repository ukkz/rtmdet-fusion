"""RTMDet-Fusion utilities"""

from .bbox import expand_bbox
from .device import get_device, setup_device_optimizations
from .versioning import (
    get_code_version,
    is_release_version,
    parse_major,
    check_checkpoint_version,
    version_filename_part,
)

__all__ = [
    # bbox
    'expand_bbox',
    # device
    'get_device',
    'setup_device_optimizations',
    # versioning
    'get_code_version',
    'is_release_version',
    'parse_major',
    'check_checkpoint_version',
    'version_filename_part',
]
