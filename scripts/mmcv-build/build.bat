@echo off
REM ============================================================
REM mmcv Build Script for RTMDet-Fusion
REM
REM Builds mmcv 2.1.0 from source with CUDA extensions
REM Requires: Visual Studio 2022, CUDA 12.8, PyTorch nightly (cu128)
REM ============================================================

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..\..
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe
set VENV_PIP=%PROJECT_ROOT%\.venv\Scripts\pip.exe

REM Overridable settings (set these before running to change the defaults)
REM   MMCV_BUILD_DIR : mmcv source checkout (default: sibling directory "mmcv-build" of this repository)
REM   MMCV_CUDA_HOME : CUDA Toolkit root (default: CUDA 12.8 default install path)
REM   MMCV_VCVARS    : vcvars64.bat of Visual Studio 2022 (default: Community edition)
if not defined MMCV_BUILD_DIR for %%I in ("%PROJECT_ROOT%\..\mmcv-build") do set "MMCV_BUILD_DIR=%%~fI"
if not defined MMCV_CUDA_HOME set "MMCV_CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
if not defined MMCV_VCVARS set "MMCV_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo mmcv source: %MMCV_BUILD_DIR%

REM Set up the Visual Studio 2022 developer environment
call "%MMCV_VCVARS%"
if errorlevel 1 (
    echo ERROR: Could not find Visual Studio 2022.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Current PyTorch/CUDA version:
echo ============================================================
"%VENV_PYTHON%" -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

echo.
echo ============================================================
echo Setting environment variables for CUDA build...
echo ============================================================
set CUDA_HOME=%MMCV_CUDA_HOME%
set CUDA_PATH=%CUDA_HOME%
set PATH=%CUDA_HOME%\bin;%PATH%
set MMCV_WITH_OPS=1
set FORCE_CUDA=1
set DISTUTILS_USE_SDK=1
set MAX_JOBS=4

cd /d "%MMCV_BUILD_DIR%"
echo Working directory: %CD%

echo.
echo ============================================================
echo Cleaning previous build artifacts and caches...
echo ============================================================
if exist build rmdir /s /q build
if exist mmcv.egg-info rmdir /s /q mmcv.egg-info
if exist mmcv\_ext*.pyd del /f mmcv\_ext*.pyd
if exist mmcv\*.so del /f mmcv\*.so
REM Ninja build cache
if exist "%TEMP%\torch_extensions" rmdir /s /q "%TEMP%\torch_extensions"
REM pip cache for mmcv
"%VENV_PIP%" cache remove mmcv 2>nul
echo Cache cleared.

echo.
echo ============================================================
echo Uninstalling existing mmcv...
echo ============================================================
"%VENV_PIP%" uninstall mmcv -y 2>nul

echo.
echo ============================================================
echo Restoring and patching pybind.cpp to exclude unnecessary ops...
echo ============================================================
REM Restore from backup if exists
if exist "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\pybind.cpp.bak" (
    echo Restoring pybind.cpp from backup...
    copy /y "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\pybind.cpp.bak" "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\pybind.cpp"
)
if exist "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\cuda\cudabind.cpp.bak" (
    echo Restoring cudabind.cpp from backup...
    copy /y "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\cuda\cudabind.cpp.bak" "%MMCV_BUILD_DIR%\mmcv\ops\csrc\pytorch\cuda\cudabind.cpp"
)
REM Also restore setup.py from backup if exists
if exist "%MMCV_BUILD_DIR%\setup.py.bak" (
    echo Restoring setup.py from backup...
    copy /y "%MMCV_BUILD_DIR%\setup.py.bak" "%MMCV_BUILD_DIR%\setup.py"
)

echo.
echo Applying patches...
REM Set MMCV_BUILD_DIR for patch scripts
set MMCV_BUILD_DIR=%MMCV_BUILD_DIR%
REM Apply patch for setup.py (exclude .cu files)
"%VENV_PYTHON%" "%SCRIPT_DIR%patch_mmcv.py"
REM Apply patch for pybind.cpp (exclude function declarations and bindings)
"%VENV_PYTHON%" "%SCRIPT_DIR%patch_pybind.py"
REM Apply patch for cudabind.cpp (exclude CUDA kernel launchers)
"%VENV_PYTHON%" "%SCRIPT_DIR%patch_cudabind.py"
REM Apply patch for __init__.py (exclude Python imports)
"%VENV_PYTHON%" "%SCRIPT_DIR%patch_mmcv_init.py"
if errorlevel 1 (
    echo WARNING: One or more patches may have failed, continuing anyway...
)

echo.
echo ============================================================
echo Building mmcv with CUDA extensions...
echo This may take 10-30 minutes...
echo ============================================================
echo.

"%VENV_PYTHON%" setup.py build_ext --inplace 2>&1

if errorlevel 1 (
    echo.
    echo ============================================================
    echo build_ext had errors. Checking excluded files...
    echo ============================================================
    if exist excluded_files.txt (
        echo.
        echo Excluded files:
        type excluded_files.txt
    )
    echo.
    echo Attempting pip install anyway...
)

echo.
echo ============================================================
echo Installing mmcv in development mode...
echo ============================================================
"%VENV_PIP%" install -e . --no-build-isolation

echo.
echo ============================================================
echo Verifying installation...
echo ============================================================
"%VENV_PYTHON%" -c "import mmcv; print(f'mmcv: {mmcv.__version__}')"
"%VENV_PYTHON%" -c "from mmcv.ops import nms; print('mmcv.ops.nms: OK')"
"%VENV_PYTHON%" -c "from mmdet.utils import register_all_modules; register_all_modules(); print('mmdet: OK')"

echo.
echo ============================================================
echo Build completed!
echo ============================================================
pause