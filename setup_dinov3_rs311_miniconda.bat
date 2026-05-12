@echo off
setlocal
REM DINOv3 ArcGIS Toolkit v2.15-rs311 - external Miniconda validation environment setup
REM Run from Anaconda Prompt or CMD where conda is available.

set ENV_NAME=dinov3-rs311
set REPO_DIR=C:\src\dinov3
set TOOLKIT_DIR=%~dp0

echo [1/7] Creating %ENV_NAME% with Python 3.11...
call conda create -n %ENV_NAME% python=3.11 -y
if errorlevel 1 goto :err

call conda activate %ENV_NAME%
if errorlevel 1 goto :err

echo [2/7] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :err

echo [3/7] Installing PyTorch CUDA 12.1 wheels. Change cu121 if your standard environment uses another CUDA wheel index.
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto :err

echo [4/7] Installing DINOv3 toolkit validation dependencies...
python -m pip install -r "%TOOLKIT_DIR%requirements_dinov3_rs311.txt"
if errorlevel 1 goto :err

echo [5/7] Checking local DINOv3 repo folder: %REPO_DIR%
if not exist "%REPO_DIR%\setup.py" (
  echo ERROR: DINOv3 repo was not found at %REPO_DIR%.
  echo Clone it first with clone_dinov3_repo_windows.ps1 or run:
  echo   git clone https://github.com/facebookresearch/dinov3.git C:\src\dinov3
  goto :err
)

echo [6/7] Installing local DINOv3 repo in editable mode...
cd /d "%REPO_DIR%"
python -m pip install -e .
if errorlevel 1 goto :err

echo [7/7] Basic import check...
python -c "import sys, torch, dinov3; print(sys.version); print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('dinov3 import OK')"
if errorlevel 1 goto :err

echo.
echo DONE. Use: conda activate %ENV_NAME%
exit /b 0

:err
echo.
echo FAILED. Review the error above.
exit /b 1
