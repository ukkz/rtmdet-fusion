# AGENTS.md

Guide for contributors and coding agents working on this repository.
What the project is and how to use it is described in [README.md](README.md); design and
procedures are in [docs/](docs/) (Japanese).

## Repository map

| Path | Content |
|------|---------|
| `configs/default.py` | Single source of truth for constants (keypoint definitions, bins, RTMDet-Ins configs, paths) |
| `rtmdet_fusion/models/` | Model (`fusion_model.py`), CoreML-specific modules (`fusion_model_coreml.py`), encoders, losses |
| `rtmdet_fusion/datasets/` | Training dataset |
| `rtmdet_fusion/utils/export_contract.py` | Export output names, order and metadata keys (shared by all exporters) |
| `rtmdet_fusion/utils/versioning.py` | Version embedding and checks |
| `tools/agora/` | Training data extraction from AGORA |
| `tools/train.py`, `tools/viewer.py` | Training and the inference viewer |
| `tools/onnx_exporter.py`, `tools/coreml_exporter_native.py` | Exporters |
| `tools/tf-converter/`, `tools/tfjs-converter/` | ONNX → SavedModel → TensorFlow.js (independent uv projects) |
| `examples/web-bench/` | Browser test app (verification only) |
| `scripts/mmcv-build/` | mmcv source build for Windows + Blackwell GPUs |

## Environment

- Use `uv` for everything Python (`uv run python ...`). Never call `python` / `pip` directly.
- `torch` / `torchvision` are pinned to the PyTorch CUDA 12.8 index in `pyproject.toml`, which has
  no macOS wheels. On macOS always run `uv run --no-sync ...` (the bundled `.envrc` sets
  `UV_NO_SYNC=1` for direnv users) and install torch and the other dependencies manually.
- MMEngine / MMDetection are installed with `mim` and mmcv is built from source, outside
  `pyproject.toml`. `uv sync` removes them; reinstall afterwards. Never run `mim install mmcv`
  (it replaces the source build).
- Do not change the project name/version, dependencies or package indexes in `pyproject.toml`
  casually: `uv.lock` goes out of sync, uv re-resolves torch, and the source-built mmcv no longer
  matches the installed torch. Comments and `description` are safe to edit.
- The converters (`tools/tf-converter`, `tools/tfjs-converter`) and `examples/web-bench` have their
  own environments; see their READMEs (web-bench also has its own `AGENTS.md`).
- Line endings are LF (enforced by `.gitattributes`, `*.bat` excepted).
- Do not commit data, weights or build artifacts (`data/`, `weights/`, `work_dirs/`,
  `onnx_models/`, converter outputs). AGORA and SMPL-X must not be redistributed.

## Language

| What | Language |
|------|----------|
| `README.ja.md` (master) | Japanese, polite form (です/ます), no space between Japanese and ASCII text (keep the spaces around bold) |
| `README.md` and other `README.<lang>.md` | Translations of `README.ja.md` |
| `docs/` | Japanese |
| Everything else: `AGENTS.md`, code comments, docstrings, CLI / log messages, tool READMEs, commit messages | English |

- Update all README languages in the same change. When `README.ja.md` has been edited by hand,
  it is the newest; translate its changes into the other languages.
- Commit messages start with a type prefix (`feat:`, `fix:`, `docs:`, `refactor:`, `perf:`, `chore:`).

## Versioning

The only version axis is the git tag (SemVer `vX.Y.Z`). There are no separate model generation
numbers.

| Change | Bump |
|--------|------|
| Checkpoint-incompatible architecture change (state_dict keys / shapes) or export contract change | MAJOR |
| Training recipe or loss change (checkpoint-compatible, results change) | MINOR |
| Changes that do not affect training results (fixes, tools, docs) | PATCH |

**Tag first, then train.** A model is trained from a clean commit that already carries its
version tag:

1. Commit the training code, then tag the commit locally (`git tag -a vX.Y.Z -m "..."`).
2. Train. `tools/train.py` embeds `git describe --tags --dirty` into the checkpoint and refuses
   to start unless it is exactly a release tag. Experimental runs must pass `--experiment`;
   their checkpoints carry the describe string (e.g. `v2.0.0-3-gabc1234`) and are not releases.
3. If the model is adopted, push the tag (`git push origin vX.Y.Z`). If it is rejected, delete
   the local tag and its checkpoints; the number can be reused. Never push a tag before adoption.

Artifacts inherit the checkpoint's version (the code that produced the weights, not the HEAD at
export time):

| Artifact | Version location | File name |
|----------|------------------|-----------|
| pth | top-level `'version'` key | (none; distinguished by work dir) |
| ONNX | `metadata_props['rtmdet_fusion_version']` | `rtmdet_fusion_{variant}_{size}_{version}.onnx` |
| CoreML | `user_defined_metadata['rtmdet_fusion_version']` | `rtmdet_fusion_coreml_{variant}_{size}_{version}.mlpackage` |
| tfjs | `model.json` `userDefinedMetadata.rtmdet_fusion_version` | `{variant}_{size}_{version}/` |

Do not change the file name patterns or the `rtmdet_fusion_*` metadata keys; downstream
applications depend on them. New artifact generators must follow the same rules.

## Export contract

- Input: `images`, BGR, 0–255, not normalized, NCHW `(1, 3, S, S)` (NHWC for TF / tfjs).
- Outputs: 19 tensors with fixed slots; `valid_count` gives the number of valid detections.
- `rtmdet_fusion/utils/export_contract.py` is the source of truth for names, order and metadata;
  [docs/output-contract.md](docs/output-contract.md) is the consumer-facing specification.
  Any change to names, order, shapes or meaning is a MAJOR change and must update both.

## Export-path constraints

The inference path is written so that it converts cleanly to ONNX (opset 18, dynamo), CoreML and
onnx2tf → TensorFlow.js. When touching code used by the exporters, keep these rules and check
[docs/export.md](docs/export.md):

- No data-dependent control flow or dynamic shapes; fixed `MAX_INSTANCES` slots and a static,
  tensor-only NMS (no `NonMaxSuppression` op).
- Avoid ops onnx2tf mishandles: concat of conv outputs with computed coordinates (split the
  weights and sum `bmm` results instead), `ReduceL2` (`torch.norm` / `F.normalize` → sum + sqrt),
  selections from 3D tensors (flatten to 2D first), partially broadcast batched matmul
  (`expand` batch dimensions first).
- After changing the export path, verify numerically through ONNX, CoreML and tfjs against
  PyTorch, not only with `onnx_exporter.py --verify`.
- `fusion_model_coreml.py` mirrors the main modules for CoreML; keep both in sync.

## Change checklist

1. Constants in `configs/default.py` → check every reference.
2. Annotation format → `tools/agora/` (extraction) and `rtmdet_fusion/datasets/keypoint_dataset.py`.
3. Model structure → `tools/train.py`, `tools/viewer.py`, both exporters and `fusion_model_coreml.py`.
4. Checkpoint format → loaders and exporters (bump MAJOR if incompatible).
5. RTMDet variant handling (`tiny` / `s`, auto-detected from checkpoint shapes) → train, viewer
   and exporters.
6. Keep the references list in the README files when reorganizing documents.
