"""Patch mmcv's ops/__init__.py to disable imports of excluded ops.

Works together with the other patch scripts: Python imports of the ops whose
sources were excluded from the build are commented out.

Usage:
    python patch_mmcv_init.py

Environment variables:
    MMCV_BUILD_DIR: path to the mmcv source checkout
                    (default: a sibling directory of this repository named "mmcv-build")
"""
import os
import re

_REPO_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MMCV_BUILD_DIR = os.environ.get("MMCV_BUILD_DIR") or os.path.join(_REPO_PARENT, "mmcv-build")
INIT_PY = os.path.join(MMCV_BUILD_DIR, "mmcv", "ops", "__init__.py")

# Ops to exclude (keep in sync with patch_pybind.py)
EXCLUDED_OPS = [
    # StyleGAN
    "upfirdn2d",
    "fused_bias_leakyrelu",
    "bias_act",
    "filtered_lrelu",

    # 3D / PointNet
    "gather_points",
    "group_points",
    "stack_group_points",
    "roipoint_pool3d",
    "three_interpolate",
    "three_nn",
    "knn",
    "iou3d",
    "furthest_point_sample",
    "ball_query",
    "stack_ball_query",
    "dynamic_point_to_voxel",
    "hard_voxelize",
    "dynamic_voxelize",
    "points_in_boxes",
    "roiaware_pool3d",
    "chamfer_distance",

    # Sparse conv
    "get_indice_pairs",
    "indice_conv",
    "fused_indice_conv",
    "indice_maxpool",
    "sparse_conv",
    "sparse_modules",
    "sparse_pool",
    "sparse_structure",
    "scatter_points",
    "voxelize",

    # Others
    "tin_shift",
    "assign_score_withk",
]

# Import line patterns:
#   from .xxx import yyy
#   from .xxx import (yyy, zzz)
IMPORT_PATTERN = re.compile(r'^from \.([\w_]+) import')


def should_exclude_line(line: str) -> tuple[bool, str]:
    """Return whether this line imports an excluded op."""
    match = IMPORT_PATTERN.match(line.strip())
    if not match:
        return False, ""

    module_name = match.group(1).lower()

    for op in EXCLUDED_OPS:
        op_lower = op.lower()
        # Match the module name against the op name
        if op_lower in module_name or module_name in op_lower:
            return True, op

    return False, ""


def patch_init():
    """Comment out imports of excluded ops in __init__.py."""

    with open(INIT_PY, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already patched
    if "# PATCHED_INIT:" in content:
        print("__init__.py is already patched.")
        return

    lines = content.split('\n')
    new_lines = []
    excluded_count = 0
    in_multiline_import = False
    multiline_excluded = False

    for line in lines:
        # Continuation line of a multi-line import
        if in_multiline_import:
            if multiline_excluded:
                new_lines.append(f"# {line}")
            else:
                new_lines.append(line)
            # A closing parenthesis ends the multi-line import
            if ')' in line:
                in_multiline_import = False
                multiline_excluded = False
            continue

        should_exclude, matched_op = should_exclude_line(line)

        if should_exclude:
            # Comment the line out
            new_lines.append(f"# PATCHED_INIT: excluded ({matched_op})")
            new_lines.append(f"# {line}")
            excluded_count += 1
            # Multi-line import (parenthesis opened but not closed)?
            if '(' in line and ')' not in line:
                in_multiline_import = True
                multiline_excluded = True
        else:
            new_lines.append(line)

    # The __all__ list may also need manual cleanup; only a warning is printed below.

    # Header comment
    header = [
        "# PATCHED_INIT: This file has been patched for RTMDet-only build",
        "# Excluded ops that are not needed for RTMDet",
        "# Original file backed up as __init__.py.bak",
        "",
    ]

    patched_content = '\n'.join(header + new_lines)

    # Back up the original
    backup_path = INIT_PY + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Write the patched file
    with open(INIT_PY, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("Patched __init__.py successfully!")
    print(f"  - Excluded {excluded_count} imports")
    print(f"\nExcluded ops: {', '.join(EXCLUDED_OPS[:5])}... ({len(EXCLUDED_OPS)} total)")
    print("\nNote: You may need to manually clean up __all__ list if there are import errors.")


if __name__ == "__main__":
    patch_init()
