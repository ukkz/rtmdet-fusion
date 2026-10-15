"""Patch mmcv's pybind.cpp to disable declarations and bindings of excluded ops.

Comments out the function declarations and the m.def(...) bindings of the ops
whose sources are excluded from the build.

Usage:
    python patch_pybind.py

Environment variables:
    MMCV_BUILD_DIR: path to the mmcv source checkout
                    (default: a sibling directory of this repository named "mmcv-build")
"""
import os
import re

_REPO_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MMCV_BUILD_DIR = os.environ.get("MMCV_BUILD_DIR") or os.path.join(_REPO_PARENT, "mmcv-build")
PYBIND_CPP = os.path.join(MMCV_BUILD_DIR, "mmcv", "ops", "csrc", "pytorch", "pybind.cpp")

# Ops to exclude entirely (clearly not needed by RTMDet)
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
    "furthest_point_sampling",
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

    # Rotated ops are kept (used internally by mmdet, e.g. active_rotated_filter):
    # "box_iou_rotated", "nms_rotated", etc.

    # Others
    "tin_shift",
    "assign_score_withk",
]


def patch_pybind():
    """Comment out excluded functions in pybind.cpp."""

    with open(PYBIND_CPP, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already patched
    if "// PATCHED_V2:" in content:
        print("pybind.cpp is already patched (v2).")
        return

    lines = content.split('\n')
    new_lines = []
    excluded_decl_count = 0
    excluded_bind_count = 0
    in_pybind_module = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect the PYBIND11_MODULE section
        if 'PYBIND11_MODULE(' in line:
            in_pybind_module = True
            new_lines.append(line)
            i += 1
            continue

        # Does this line refer to an excluded function?
        should_exclude = False
        matched_op = None

        for op in EXCLUDED_OPS:
            if not in_pybind_module:
                # Declarations: void xxx_forward, void xxx_backward, Tensor xxx, std::vector xxx, ...
                if re.search(rf'\b{op}\b', line, re.IGNORECASE):
                    should_exclude = True
                    matched_op = op
                    break
            else:
                # Bindings: m.def("xxx_forward", ...)
                if re.search(rf'm\.def\s*\(\s*"[^"]*{op}', line, re.IGNORECASE):
                    should_exclude = True
                    matched_op = op
                    break

        if should_exclude:
            # A statement may span several lines; collect it entirely
            block_lines = []
            paren_count = 0
            started = False

            while i < len(lines):
                current_line = lines[i]
                block_lines.append(current_line)

                # Count parentheses
                for char in current_line:
                    if char == '(':
                        paren_count += 1
                        started = True
                    elif char == ')':
                        paren_count -= 1

                # Statement ends with a semicolon
                if ';' in current_line:
                    if not in_pybind_module or (started and paren_count <= 0):
                        break

                i += 1

            # Comment the statement out
            new_lines.append(f"// PATCHED_V2: excluded ({matched_op})")
            for bl in block_lines:
                new_lines.append(f"// {bl}")

            if in_pybind_module:
                excluded_bind_count += 1
            else:
                excluded_decl_count += 1
        else:
            new_lines.append(line)

        i += 1

    # Header comment
    header = [
        "// PATCHED_V2: This file has been patched for RTMDet-only build",
        "// Excluded ops that are not needed for RTMDet to reduce build complexity",
        "// Original file backed up as pybind.cpp.bak",
        "",
    ]

    patched_content = '\n'.join(header + new_lines)

    # Back up the original
    backup_path = PYBIND_CPP + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Write the patched file
    with open(PYBIND_CPP, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("Patched pybind.cpp successfully!")
    print(f"  - Excluded {excluded_decl_count} function declarations")
    print(f"  - Excluded {excluded_bind_count} pybind bindings")
    print(f"\nExcluded ops: {', '.join(EXCLUDED_OPS[:5])}... ({len(EXCLUDED_OPS)} total)")


if __name__ == "__main__":
    patch_pybind()
