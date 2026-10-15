# ONNX → TensorFlow / TensorFlow.js converters

Two independent uv projects convert the exported ONNX models for the web:

```
onnx_models/rtmdet_fusion_{variant}_{size}_{version}.onnx
    │  tools/tf-converter   (onnx2tf)
    ▼
tools/tf-converter/output/savedmodel_{variant}_{size}_{version}/   (+ .tflite by-products)
    │  tools/tfjs-converter (tensorflowjs, fp16 by default)
    ▼
examples/web-bench/public/tfjs_model/{variant}_{size}_{version}/
```

All outputs are build artifacts and are not tracked by git. Tested on macOS (Apple Silicon).

## Setup

```bash
# ONNX -> SavedModel (onnx2tf 1.28.8 / TensorFlow 2.16, pinned by uv.lock)
cd tools/tf-converter
uv sync

# SavedModel -> TensorFlow.js (installed manually for Apple Silicon compatibility)
cd ../tfjs-converter
uv venv
uv pip install tensorflowjs==4.22.0
```

## Convert

`models/` is a symlink to the repository's `onnx_models/`; export the ONNX models there first
(see `docs/export.md`).

```bash
cd tools/tf-converter
PATH="$PWD/.venv/bin:$PATH" .venv/bin/python convert_to_savedmodel.py --version v2.0.0 --variant s --size 640

cd ../tfjs-converter
.venv/bin/python convert.py --version v2.0.0 --variant s --size 640
```

- `PATH` must include the venv because onnx2tf invokes the `onnxsim` command.
- `convert_to_savedmodel.py` needs an image with at least one person as representative input
  (`--sample-image`, default `examples/web-bench/public/test.jpg`, which is not included in the
  repository: place an image there or pass `--sample-image`). onnx2tf picks layout
  transpositions by comparing ONNX and TF outputs; with an image without people the keypoint
  heads receive all-zero inputs and a wrong transposition can be chosen silently.
- `convert.py` fixes `structuredOutputKeys` in `model.json` so that `model.predict()` returns
  the same output names as the ONNX model, and copies the `rtmdet_fusion_*` metadata into
  `userDefinedMetadata`.
