"""Patch mmcv's cudabind.cpp to disable CUDA kernel launchers of excluded ops.

Comments out the declarations, implementations and REGISTER_DEVICE_IMPL entries
in cudabind.cpp that correspond to the CUDA sources excluded by patch_mmcv.py.

Usage:
    python patch_cudabind.py

Environment variables:
    MMCV_BUILD_DIR: path to the mmcv source checkout
                    (default: a sibling directory of this repository named "mmcv-build")
"""
import os

_REPO_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MMCV_BUILD_DIR = os.environ.get("MMCV_BUILD_DIR") or os.path.join(_REPO_PARENT, "mmcv-build")
CUDABIND_CPP = os.path.join(MMCV_BUILD_DIR, "mmcv", "ops", "csrc", "pytorch", "cuda", "cudabind.cpp")

# Ops to exclude (function-name patterns) - keep in sync with patch_pybind.py
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


def should_exclude_line(line: str) -> tuple[bool, str]:
    """Return whether this line refers to an excluded op."""
    line_lower = line.lower()
    for op in EXCLUDED_OPS:
        op_lower = op.lower()
        # Match both snake_case and CamelCase function names,
        # e.g. assign_score_withk / AssignScoreWithK, ball_query / BallQuery
        camel_pattern = ''.join(word.capitalize() for word in op.split('_'))

        if op_lower in line_lower or camel_pattern.lower() in line_lower:
            return True, op
    return False, ""


def patch_cudabind():
    """Comment out excluded functions in cudabind.cpp."""

    with open(CUDABIND_CPP, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if already patched
    if "// PATCHED_CUDABIND:" in content:
        print("cudabind.cpp is already patched.")
        return

    lines = content.split('\n')
    new_lines = []
    excluded_count = 0
    i = 0

    while i < len(lines):
        line = lines[i]
        should_exclude, matched_op = should_exclude_line(line)

        if should_exclude:
            # Comment out the whole block: a function declaration, a function
            # definition or a REGISTER_DEVICE_IMPL entry
            block_lines = []
            brace_count = 0
            paren_count = 0
            in_block = False
            is_register_device_impl = 'REGISTER_DEVICE_IMPL' in line

            while i < len(lines):
                current_line = lines[i]
                block_lines.append(current_line)

                # Count braces
                brace_count += current_line.count('{')
                brace_count -= current_line.count('}')

                # Count parentheses (for REGISTER_DEVICE_IMPL)
                paren_count += current_line.count('(')
                paren_count -= current_line.count(')')

                if '{' in current_line:
                    in_block = True

                # End of block:
                # 1. function definition: a block enclosed in { }
                # 2. function declaration: ends with ; (when not a definition)
                # 3. REGISTER_DEVICE_IMPL: ends when the parentheses close

                if in_block and brace_count == 0:
                    # End of a function definition
                    i += 1
                    break
                elif is_register_device_impl and paren_count == 0:
                    # End of REGISTER_DEVICE_IMPL (parentheses closed)
                    i += 1
                    break
                elif not in_block and not is_register_device_impl and ';' in current_line:
                    # End of a function declaration
                    i += 1
                    break

                i += 1

            # Comment the block out
            new_lines.append(f"// PATCHED_CUDABIND: excluded ({matched_op})")
            for bl in block_lines:
                new_lines.append(f"// {bl}")
            excluded_count += 1
        else:
            new_lines.append(line)
            i += 1

    # Header comment
    header = [
        "// PATCHED_CUDABIND: This file has been patched for RTMDet-only build",
        "// Excluded CUDA kernel launchers that are not needed for RTMDet",
        "// Original file backed up as cudabind.cpp.bak",
        "",
    ]

    patched_content = '\n'.join(header + new_lines)

    # Back up the original
    backup_path = CUDABIND_CPP + ".bak"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Backup saved to: {backup_path}")
    else:
        print(f"Backup already exists: {backup_path}")

    # Write the patched file
    with open(CUDABIND_CPP, 'w', encoding='utf-8') as f:
        f.write(patched_content)

    print("Patched cudabind.cpp successfully!")
    print(f"  - Excluded {excluded_count} blocks")
    print(f"\nExcluded ops: {', '.join(EXCLUDED_OPS[:5])}... ({len(EXCLUDED_OPS)} total)")


if __name__ == "__main__":
    patch_cudabind()
