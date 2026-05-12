# 03. ArcGIS Pro Deep Learning 環境

## 建議環境

- ArcGIS Pro 3.6.x
- Python 3.13.x
- Deep Learning Libraries 3.6
- PyTorch 2.5.1 / CUDA 12.6 由 Esri deep-learning-essentials 提供

## 基本檢查

```bat
"C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL\python.exe" -u -c "import arcpy; print('arcpy OK', arcpy.GetInstallInfo()['Version']); import arcgis; print('arcgis OK', arcgis.__version__)"
```

測試 `arcgis.learn`：

```bat
"C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL\python.exe" -u -c "from arcgis.learn import prepare_data, ModelExtension; print('arcgis.learn OK')"
```

## spconv.pytorch crash 與 2DTEST clone

部分 ArcGIS Pro 3.6 Deep Learning clone 可能出現：

```text
import spconv.pytorch
→ -1073741819 / 0xC0000005 Access Violation
```

而 `arcgis.learn` 初始化時可能間接載入 `spconv.pytorch`，造成 `arcgis.learn` import crash。

對 DINOv3 2D pixel classification 而言，`spconv` 並非必要。建議建立 2D-only clone 測試環境：

```bat
"C:\Program Files\ArcGIS\Pro\bin\Python\Scripts\conda.exe" create --clone arcgispro-py3-DL -n arcgispro-py3-DL-2DTEST -y
```

停用 spconv：

```bat
set SP=C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\Lib\site-packages
ren "%SP%\spconv" "_spconv_DISABLED"
for /d %D in ("%SP%\spconv-*.dist-info") do ren "%D" "_%~nxD_DISABLED"
```

再測：

```bat
"C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe" -u -c "import arcpy; print('arcpy OK', arcpy.GetInstallInfo()['Version']); import arcgis; print('arcgis OK', arcgis.__version__); from arcgis.learn import prepare_data, ModelExtension; print('arcgis.learn OK')"
```

成功後，工具欄位 `ArcGIS Pro DL Python Executable` 請指定：

```text
C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe
```

## 注意

`arcgispro-py3-DL-2DTEST` 不建議用於 3D deep learning、point cloud detection、mmdet3d 或 spconv-based workflow。原本的 `arcgispro-py3-DL` 請保留。
