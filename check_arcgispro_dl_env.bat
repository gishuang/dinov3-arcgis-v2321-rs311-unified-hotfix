@echo off
REM Run this inside ArcGIS Pro Deep Learning Python environment, e.g. arcgispro-py3-DL.
REM This does not create or modify the environment except optional torchmetrics installation below.

python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import arcpy; print('arcpy OK')"
python -c "import arcgis; print('arcgis', arcgis.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python -c "import numpy; print('numpy', numpy.__version__)"

echo If fallback torch.hub.load complains about torchmetrics, run:
echo   python -m pip install torchmetrics
