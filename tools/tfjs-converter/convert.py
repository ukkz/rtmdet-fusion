#!/usr/bin/env python3
"""Convert SavedModels produced by tf-converter to TensorFlow.js.

Usage:
    uv run python convert.py --version v2.0.0                          # all variants/sizes, fp16
    uv run python convert.py --version v2.0.0 --variant s --size 640
    uv run python convert.py --version v2.0.0 --quantize uint8         # smallest size
    uv run python convert.py --version v2.0.0 --quantize none          # fp32

Input:  ../tf-converter/output/savedmodel_{variant}_{size}_{version}/
Output: examples/web-bench/public/tfjs_model/{variant}_{size}_{version}/ (not tracked by git)
"""

import argparse
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Supported variants and input sizes
SUPPORTED_VARIANTS = ["s", "tiny"]
SUPPORTED_SIZES = [256, 320, 416, 512, 640]


def fix_structured_output_keys(savedmodel_path: Path, output_path: Path, signature: str) -> None:
    """Reorder structuredOutputKeys in model.json to the signature's output order.

    tfjs predict() assigns structuredOutputKeys[i] to the i-th output of the signature, but
    tensorflowjs writes structuredOutputKeys in alphabetical order, so output names and
    contents get mixed up (e.g. bboxes ending up in appearance_features). The mapping
    "output tensor name -> output name" is derived from the SavedModel signature and the
    keys are reordered after checking the shapes.
    """
    import json

    import tensorflow as tf

    fn = tf.saved_model.load(str(savedmodel_path)).signatures[signature]
    # The concrete function outputs Identity_N follow structured_outputs flattened in name order
    keys = sorted(fn.structured_outputs)
    name_to_key = {t.name: k for t, k in zip(fn.outputs, keys)}
    # Outputs renamed to their real names by optimization (appear without ':0' in model.json)
    name_to_key.update({f"{k}:0": k for k in keys})
    name_to_key.update({k: k for k in keys})
    key_shape = {k: [int(d) for d in fn.structured_outputs[k].shape] for k in keys}

    model_json_path = output_path / "model.json"
    model_json = json.loads(model_json_path.read_text())
    fixed = []
    for tensor_name, info in model_json["signature"]["outputs"].items():
        key = name_to_key[tensor_name]
        shape = [int(d["size"]) for d in info["tensorShape"]["dim"]]
        if shape != key_shape[key]:
            raise ValueError(f"Output shape mismatch: {tensor_name} -> {key} {shape} != {key_shape[key]}")
        fixed.append(key)
    if sorted(fixed) != keys:
        raise ValueError(f"Incomplete output name mapping: {fixed}")
    model_json["userDefinedMetadata"]["structuredOutputKeys"] = fixed
    model_json_path.write_text(json.dumps(model_json))
    print(f"Reordered structuredOutputKeys to the signature order ({len(fixed)} outputs)")


def merge_rtmdet_metadata(savedmodel_path: Path, output_path: Path) -> None:
    """Merge the ONNX metadata (version and export settings) left by tf-converter into
    userDefinedMetadata of model.json, so that tfjs consumers can read them."""
    import json

    meta_path = savedmodel_path / "rtmdet_fusion_metadata.json"
    if not meta_path.exists():
        return
    props = json.loads(meta_path.read_text())
    model_json_path = output_path / "model.json"
    model_json = json.loads(model_json_path.read_text())
    model_json.setdefault("userDefinedMetadata", {}).update(props)
    model_json_path.write_text(json.dumps(model_json))
    print(f"Merged metadata into model.json: {props}")


def convert_model(
    input_path: Path,
    output_path: Path,
    signature: str,
    quantize: str,
) -> bool:
    """Convert a single SavedModel to TensorFlow.js."""
    import tensorflowjs as tfjs

    print(f"\n{'='*60}")
    print(f"Converting SavedModel: {input_path}")
    print(f"Output: {output_path}")
    print(f"Signature: {signature}")
    print(f"Quantization: {quantize}")

    # Build quantization map
    quantization_dtype_map = None
    if quantize != "none":
        quantization_dtype_map = {quantize: "*"}

    tfjs.converters.convert_tf_saved_model(
        str(input_path),
        str(output_path),
        signature_def=signature,
        quantization_dtype_map=quantization_dtype_map,
        use_structured_outputs_names=True,  # keep the SavedModel output names
    )
    fix_structured_output_keys(input_path, output_path, signature)
    merge_rtmdet_metadata(input_path, output_path)

    print("Conversion completed!")

    # Show output files
    if output_path.exists():
        files = list(output_path.iterdir())
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        print(f"Output files: {len(files)} files, {total_size / 1024 / 1024:.1f} MB")

    return True


def main():
    parser = argparse.ArgumentParser(description="SavedModel to TensorFlow.js converter")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Version suffix of the SavedModel names (e.g. v2.0.0, same as tf-converter --version)",
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
        "--signature",
        type=str,
        default="serving_default",
        help="Signature name (default: serving_default)",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="float16",
        choices=["float16", "uint8", "uint16", "none"],
        help="Quantization type (default: float16)",
    )
    args = parser.parse_args()

    savedmodel_base_dir = BASE_DIR / "../tf-converter/output"
    output_base_dir = BASE_DIR / "../../examples/web-bench/public/tfjs_model"

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
            input_path = savedmodel_base_dir / f"savedmodel_{name}"
            if not input_path.exists():
                print(f"Warning: SavedModel not found: {input_path}")
                continue

            output_path = output_base_dir / name
            # Recreate the directory so that stale shards (group1-shardXofN.bin) do not remain
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True)

            if convert_model(input_path, output_path, args.signature, args.quantize):
                converted.append(name)

    print(f"\n{'='*60}")
    print(f"Converted: {len(converted)}/{len(variants) * len(sizes)} models")
    for name in converted:
        print(f"  - {output_base_dir / name}")


if __name__ == "__main__":
    main()
