# Building mmcv with CUDA ops on Windows (Blackwell / CUDA 12.8)

Unofficial helper scripts that build **mmcv 2.1.0 with CUDA extensions from source** on Windows
for RTX 50-series (Blackwell, `sm_120`) GPUs. Only the ops needed by RTMDet / MMDetection are
built; ops that fail to compile with CUDA 12.8 (and that RTMDet does not use) are excluded.

These scripts are provided as-is. They are the configuration that worked for us; other
combinations of GPU, CUDA, compiler and PyTorch versions may need adjustments.

## Why a source build is needed

1. Blackwell GPUs (`sm_120`) require a PyTorch build against CUDA 12.8 or later.
2. Official mmcv wheels are not provided for recent PyTorch / CUDA combinations, so mmcv must
   be built against the installed PyTorch.
3. Some mmcv ops (StyleGAN, 3D/PointNet and sparse-convolution ops) fail to compile with
   CUDA 12.8 on Windows. RTMDet does not need them, so they are patched out before the build.

On Linux a plain source build (`MMCV_WITH_OPS=1 pip install -e .` in an mmcv checkout) is
usually sufficient; these scripts target Windows.

## Tested configuration

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 5070 Ti (Blackwell, sm_120) |
| CUDA Toolkit | 12.8 |
| PyTorch | 2.10.0 nightly (cu128) |
| Compiler | Visual Studio 2022 Community (C++ desktop workload) |
| Python | 3.12 (managed by uv) |
| mmcv | 2.1.0 (built from source) |
| mmengine | 0.10.x |
| mmdet | 3.3.0 |

## Files

```
scripts/mmcv-build/
├── build.bat           # Main build script
├── patch_mmcv.py       # Patches setup.py (excludes .cu sources)
├── patch_pybind.py     # Patches pybind.cpp (excludes declarations and bindings)
├── patch_cudabind.py   # Patches cudabind.cpp (excludes CUDA kernel launchers)
└── patch_mmcv_init.py  # Patches ops/__init__.py (excludes Python imports)
```

## Prerequisites

1. **mmcv source** — clone it next to this repository (the default location):

   ```bat
   cd <parent directory of rtmdet-fusion>
   git clone https://github.com/open-mmlab/mmcv.git mmcv-build
   cd mmcv-build
   git checkout v2.1.0
   ```

2. **CUDA Toolkit 12.8** — [download](https://developer.nvidia.com/cuda-12-8-0-download-archive).
   Default path: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`
3. **Visual Studio 2022** with the "Desktop development with C++" workload.
4. **PyTorch built for CUDA 12.8** in this repository's `.venv` (`uv sync`; the index is
   configured in `pyproject.toml`), plus MMEngine and MMDetection:

   ```bat
   uv sync
   uv run mim install mmengine mmdet
   ```

## Build

Run from a **"x64 Native Tools Command Prompt for VS 2022"** (or a Command Prompt where
`vcvars64.bat` has been called), not from PowerShell:

```bat
cd <path to rtmdet-fusion>\scripts\mmcv-build
build.bat
```

The build takes roughly 10–30 minutes. mmcv is installed into `.venv` in editable mode
(`pip install -e`), so **keep the mmcv source directory** after the build.

> Do not run `mim install mmcv` afterwards: it replaces the source build with an incompatible
> package.

### Settings

`build.bat` reads the following environment variables (defaults in parentheses):

| Variable | Meaning |
|----------|---------|
| `MMCV_BUILD_DIR` | mmcv source checkout (`..\mmcv-build` relative to this repository) |
| `MMCV_CUDA_HOME` | CUDA Toolkit root (`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`) |
| `MMCV_VCVARS` | `vcvars64.bat` of Visual Studio 2022 (Community edition path) |

The patch scripts also honor `MMCV_BUILD_DIR`.

### Re-running the build

On the first run each patch script saves the original file as `*.bak`. On later runs
`build.bat` restores the originals from these backups before patching again, so the patches
are never applied twice.

## Excluded ops

Excluded because RTMDet does not use them and they fail to build with CUDA 12.8:

- **StyleGAN**: `upfirdn2d`, `fused_bias_leakyrelu`, `bias_act`, `filtered_lrelu`
- **3D / PointNet**: `gather_points`, `group_points`, `roipoint_pool3d`, `three_interpolate`,
  `three_nn`, `knn`, `iou3d`, `furthest_point_sample`, `ball_query`, `points_in_boxes`,
  `roiaware_pool3d`, `chamfer_distance`, `hard_voxelize`, `dynamic_voxelize`
- **Sparse convolution**: `get_indice_pairs`, `indice_conv`, `fused_indice_conv`,
  `indice_maxpool`, `sparse_conv`, `sparse_modules`, `sparse_pool`, `sparse_structure`,
  `scatter_points`
- **Others**: `tin_shift`, `assign_score_withk`

To add or remove ops, edit the exclusion lists in **all** patch scripts and keep them in sync.

Ops used by RTMDet / MMDetection are kept, e.g. `nms`, `softnms`, `nms_rotated`, `roi_align`,
`roi_pool`, `deform_conv`, `modulated_deform_conv`, `box_iou_rotated`, `active_rotated_filter`,
`ms_deform_attn`, `sigmoid_focal_loss`, `bbox_overlaps`, `sync_bn`, `carafe`.

## Verification

```python
import mmcv
print(f'mmcv: {mmcv.__version__}')  # 2.1.0

from mmcv.ops import nms
print('mmcv.ops.nms: OK')

from mmdet.utils import register_all_modules
register_all_modules()
print('mmdet: OK')
```

## Troubleshooting

| Error | What to check |
|-------|---------------|
| `CUDA_HOME environment variable is not set` | CUDA 12.8 is installed and `MMCV_CUDA_HOME` points to it |
| `Could not find Visual Studio 2022` | Visual Studio 2022 is installed and `MMCV_VCVARS` points to `vcvars64.bat` |
| `sm_120 is not compatible` | PyTorch is a CUDA 12.8 build (`torch.version.cuda` prints `12.8`) |
| `xxx_forward miss in module _ext` | Patches were not applied correctly; restore from the `.bak` files and rebuild |
| Linker error `LNK1120: unresolved externals` | Same as above (`pybind.cpp` / `cudabind.cpp` patches) |

## Links

- [mmcv](https://github.com/open-mmlab/mmcv)
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
