"""Patch mmcv's setup.py to exclude source files that fail to build.

Modifies <MMCV_BUILD_DIR>/setup.py so that CUDA sources of ops RTMDet does not
need are filtered out before the extension is built.

Usage:
    python patch_mmcv.py

Environment variables:
    MMCV_BUILD_DIR: path to the mmcv source checkout
                    (default: a sibling directory of this repository named "mmcv-build")
"""
import os
import re

_REPO_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MMCV_BUILD_DIR = os.environ.get("MMCV_BUILD_DIR") or os.path.join(_REPO_PARENT, "mmcv-build")
SETUP_PY = os.path.join(MMCV_BUILD_DIR, "setup.py")

# File-name patterns of ops that RTMDet does not need and that tend to fail to build
# (StyleGAN, 3D, etc.).
# Keep in sync with EXCLUDED_OPS in patch_pybind.py.
EXCLUDE_PATTERNS = [
    # StyleGAN
    "filtered_lrelu",
    "fused_bias_act",
    "bias_act",
    "upfirdn2d",

    # 3D / PointNet
    "gather_points",
    "group_points",
    "roipoint_pool3d",
    "three_interpolate",
    "three_nn",
    "knn",
    "iou3d",
    "furthest_point_sample",
    "ball_query",
    "voxeliz",          # voxelize, voxelization
    "scatter_points",
    "points_in_boxes",
    "roiaware_pool3d",
    "chamfer_distance",

    # Sparse conv
    "sparse",           # sparse_indice, sparse_conv, etc.
    "spconv",
    "indice_conv",

    # Rotated ops are kept (used internally by mmdet):
    # "box_iou_rotated", "nms_rotated", "roi_align_rotated", etc.

    # Others
    "tin_shift",
    "assign_score_withk",
]

def patch_setup_py():
    """Apply the file-exclusion patch to setup.py."""

    with open(SETUP_PY, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already patched
    if "# PATCHED: exclude problematic files" in content:
        print("setup.py is already patched.")
        return

    # Build the exclusion list literal
    exclude_list = ", ".join([f'"{p}"' for p in EXCLUDE_PATTERNS])

    # Code injected into setup.py
    patch_code = f'''
# PATCHED: exclude problematic files for Windows/CUDA 12.8+ build
EXCLUDE_PATTERNS = [{exclude_list}]

def filter_problematic_files(file_list):
    """Drop source files that tend to fail to build."""
    filtered = []
    excluded = []
    for f in file_list:
        fname = os.path.basename(f).lower()
        should_exclude = any(pat.lower() in fname for pat in EXCLUDE_PATTERNS)
        if should_exclude:
            excluded.append(f)
        else:
            filtered.append(f)
    if excluded:
        print(f"[PATCH] Excluding {{len(excluded)}} problematic files:")
        for f in excluded:
            print(f"  - {{os.path.basename(f)}}")
        # Save the exclusion list to a file
        with open("excluded_files.txt", "w") as ef:
            for f in excluded:
                ef.write(f + "\\n")
    return filtered

'''

    # Insert the patch code after the imports
    import_end = content.find("EXT_TYPE = ''")
    if import_end == -1:
        print("ERROR: Could not find insertion point in setup.py")
        return

    patched_content = content[:import_end] + patch_code + content[import_end:]

    # op_files is assigned in several places; apply the filter right before
    # the extension is created ("ext_ops = extension(").
    ext_ops_pattern = r"(\s+)(ext_ops = extension\()"
    replacement = r"\1op_files = filter_problematic_files(op_files)\n\1\2"

    patched_content = re.sub(ext_ops_pattern, replacement, patched_content)

    # Back up the original
    backup_path = SETUP_PY + ".bak"
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Backup saved to: {backup_path}")

    # Write the patched file
    with open(SETUP_PY, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("Patched setup.py successfully!")
    print(f"Excluded patterns: {len(EXCLUDE_PATTERNS)}")
    print("Patterns:", ", ".join(EXCLUDE_PATTERNS))


if __name__ == "__main__":
    patch_setup_py()
