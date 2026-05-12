@echo off
REM Run this inside the ArcGIS Pro Deep Learning Python environment.
REM Example:
REM   conda activate arcgispro-py3-DL
REM   install_missing_dinov3_deps_arcgispro.bat

python -m pip install torchmetrics
python -c "import torchmetrics; print('torchmetrics', torchmetrics.__version__)"
