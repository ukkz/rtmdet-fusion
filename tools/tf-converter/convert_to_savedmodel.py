#!/usr/bin/env python3
"""Convert exported ONNX models to TensorFlow SavedModel (onnx2tf).

Usage:
    uv run python convert_to_savedmodel.py --version v2.0.0                        # all variants/sizes
    uv run python convert_to_savedmodel.py --version v2.0.0 --variant s --size 640

Input:  models/rtmdet_fusion_{variant}_{size}_{version}.onnx
        (models/ is a symlink to the repository's onnx_models/)
Output: output/savedmodel_{variant}_{size}_{version}/ (.tflite files are produced as a by-product)

onnx2tf decides layout transpositions by comparing ONNX and TF outputs on test data, so a
representative image with at least one person in it is required (--sample-image).
"""

import argparse
import os
import sys
import types
from pathlib import Path
from typing import Optional


def _ensure_litert_importable() -> None:
    """Make `ai_edge_litert` importable so that onnx2tf can be imported.

    onnx2tf imports `ai_edge_litert.interpreter` unconditionally, but the package has no wheel
    for some platforms (e.g. macOS arm64). SavedModel conversion does not use the interpreter,
    so a stub that only fails when instantiated is sufficient.
    """
    try:
        import ai_edge_litert.interpreter  # noqa: F401
    except ImportError:
        class Interpreter:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("ai-edge-litert is not available on this platform")

        package = types.ModuleType("ai_edge_litert")
        interpreter = types.ModuleType("ai_edge_litert.interpreter")
        interpreter.Interpreter = Interpreter
        package.interpreter = interpreter
        sys.modules["ai_edge_litert"] = package
        sys.modules["ai_edge_litert.interpreter"] = interpreter


_ensure_litert_importable()

import onnx2tf  # noqa: E402


# Supported variants and input sizes
SUPPORTED_VARIANTS = ["s", "tiny"]
SUPPORTED_SIZES = [256, 320, 416, 512, 640]

DEFAULT_SAMPLE_IMAGE = Path(__file__).parent / "../../examples/web-bench/public/test.jpg"

# Sample data that onnx2tf reads from the current directory during conversion.
# The download source used by onnx2tf 1.28.8 is gone, so it is generated from the sample
# image in the same format (20x128x128x3 float32, RGB, 0-1) when missing.
SAMPLE_DATA_NAME = "calibration_image_sample_data_20x128x128x3_float32.npy"

METADATA_FILE = "rtmdet_fusion_metadata.json"


def load_sample_image(path: Path):
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(
            f"Sample image not found: {path}\n"
            "Pass an image that contains at least one person with --sample-image."
        )
    return img


def ensure_onnx2tf_sample_data(work_dir: Path, sample_image: Path) -> None:
    """Generate onnx2tf's sample data if it does not exist."""
    path = work_dir / SAMPLE_DATA_NAME
    if path.exists():
        return
    import cv2
    import numpy as np

    img = load_sample_image(sample_image)
    rng = np.random.default_rng(0)
    h, w = img.shape[:2]
    crops = []
    for _ in range(20):
        s = int(rng.integers(min(h, w) // 3, min(h, w) + 1))
        y = int(rng.integers(0, h - s + 1))
        x = int(rng.integers(0, w - s + 1))
        crop = cv2.resize(img[y:y + s, x:x + s], (128, 128), interpolation=cv2.INTER_AREA)
        crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    np.save(path, np.stack(crops).astype(np.float32) / 255.0)
    print(f"Generated onnx2tf sample data: {path}")


def build_gridsample_replace_json(onnx_path: Path, out_path: Path) -> Path:
    """Generate a param_replacement JSON that transposes the grid input of GridSample nodes.

    onnx2tf treats the 4D grid (N, Gh, Gw, 2) as NCHW and converts it to NHWC, which turns it
    into (N, Gh, 2, Gw). A [0, 1, 3, 2] transpose right before GridSample restores it.
    Node names depend on the exporter, so the JSON is generated from the graph every time.
    ONNX files exported with the default matrix-multiplication sampling contain no GridSample.
    """
    import json

    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    operations = [
        {
            "op_name": node.name,
            "param_target": "inputs",
            "param_name": node.input[1],
            "pre_process_transpose_perm": [0, 1, 3, 2],
        }
        for node in graph.node if node.op_type == "GridSample"
    ]
    out_path.write_text(json.dumps({"format_version": 1, "operations": operations}, indent=2))
    print(f"Generated GridSample transpose rules: {len(operations)} nodes -> {out_path}")
    return out_path


def build_representative_input(sample_image: Path, size: int, out_path: Path) -> Path:
    """Create the representative input onnx2tf uses to validate the conversion.

    onnx2tf feeds test data to both ONNX and TF during conversion and picks the direction of
    layout transpositions whose outputs match. Its default test data is normalized to 0-1, which
    is almost black for this model (it takes 0-255 input), so no person is detected. The
    keypoint heads then receive all-zero inputs, every transposition looks like a match and a
    wrong one can be chosen (keypoints were off by 20px or more). Therefore a real image that
    produces detections is given in the model's input format (BGR, 0-255, NHWC, SxS).
    """
    import cv2
    import numpy as np

    img = load_sample_image(sample_image)
    img = cv2.resize(img, (size, size))  # same non-uniform resize as at inference time
    np.save(out_path, img.astype(np.float32)[None])  # (1, S, S, 3)
    return out_path


def write_onnx_metadata(onnx_path: Path, output_dir: Path) -> None:
    """Save the ONNX metadata_props (version and export settings) as JSON in the SavedModel folder.

    SavedModel / tfjs do not carry over ONNX metadata, so tfjs-converter merges this JSON into
    userDefinedMetadata of model.json.
    """
    import json

    import onnx

    model = onnx.load(str(onnx_path), load_external_data=False)
    props = {p.key: p.value for p in model.metadata_props if p.key.startswith("rtmdet_fusion_")}
    (output_dir / METADATA_FILE).write_text(json.dumps(props, ensure_ascii=False, indent=2))
    print(f"Saved ONNX metadata: {props}")


def convert_model(onnx_path: Path, output_dir: Path, replace_json: Optional[Path],
                  representative_input: Path) -> None:
    """Convert a single ONNX model to SavedModel."""
    print(f"\n{'='*60}")
    print(f"Input: {onnx_path}")
    print(f"Output: {output_dir}")
    print("Converting...")

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=False,
        param_replacement_file=str(replace_json) if replace_json and replace_json.exists() else None,
        output_nms_with_dynamic_tensor=True,
        # The representative input is raw 0-255 values (mean=0, std=1)
        custom_input_op_name_np_data_path=[
            ["images", str(representative_input), [[[[0.0, 0.0, 0.0]]]], [[[[1.0, 1.0, 1.0]]]]]
        ],
    )

    print(f"Done: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="ONNX to SavedModel converter")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version suffix of the ONNX file names (e.g. v2.0.0)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        nargs="*",
        default=None,
        choices=SUPPORTED_VARIANTS,
        help=f"Variants to convert (multiple allowed, default: all). Choices: {SUPPORTED_VARIANTS}",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs="*",
        default=None,
        help=f"Input sizes to convert (multiple allowed, default: all). Choices: {SUPPORTED_SIZES}",
    )
    parser.add_argument(
        "--sample-image",
        type=Path,
        default=DEFAULT_SAMPLE_IMAGE,
        help="Image with at least one person, used as representative input "
             "(default: examples/web-bench/public/test.jpg)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    sample_image = args.sample_image.resolve()
    # onnx2tf reads its sample data from the current directory
    os.chdir(script_dir)
    ensure_onnx2tf_sample_data(script_dir, sample_image)
    models_dir = script_dir / "models"
    output_base_dir = script_dir / "output"

    variants = args.variant if args.variant else SUPPORTED_VARIANTS
    sizes = args.size if args.size else SUPPORTED_SIZES
    sizes = [s for s in sizes if s in SUPPORTED_SIZES]
    if not sizes:
        print("No valid sizes to convert.")
        return

    print(f"Targets: variants={variants}, sizes={sizes}")

    converted = []
    for variant in variants:
        for size in sizes:
            name = f"{variant}_{size}_{args.version}"
            onnx_path = models_dir / f"rtmdet_fusion_{name}.onnx"
            if not onnx_path.exists():
                print(f"Warning: ONNX file not found: {onnx_path}")
                continue

            output_dir = output_base_dir / f"savedmodel_{name}"
            output_dir.mkdir(parents=True, exist_ok=True)
            replace_json = build_gridsample_replace_json(
                onnx_path, output_base_dir / f"replace_{name}.json"
            )
            representative_input = build_representative_input(
                sample_image, size, output_base_dir / f"representative_input_{size}.npy"
            )

            convert_model(onnx_path, output_dir, replace_json, representative_input)
            write_onnx_metadata(onnx_path, output_dir)
            converted.append(name)

    print(f"\n{'='*60}")
    print(f"Converted: {len(converted)}/{len(variants) * len(sizes)} models")
    for name in converted:
        print(f"  - {output_base_dir / f'savedmodel_{name}'}")


if __name__ == "__main__":
    main()
