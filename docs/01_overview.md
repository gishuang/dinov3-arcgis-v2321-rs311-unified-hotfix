# 01. 架構與版本說明

## 版本

`v2.32.1-rs311-unified-hotfix`

此版是在 v2.32 統一版上修正 CLI parser 漏掉 `max` 選項的 hotfix。v2.32.1 允許：

```text
--input-scale max
--training-input-scale max
```

因此 ArcGIS Pro 工具介面中的 `Inference Input Scale = max` 與 `Training Input Scale = max` 不會再被 argparse 擋下。

## 架構概念

工具分成三個層次：

1. **ArcGIS Pro Python Toolbox (`.pyt`)**
   - 提供 UI 參數。
   - 不直接 import `torch / CUDA / DINOv3`。
   - 以 external subprocess 呼叫訓練 script。

2. **訓練核心 (`dinov3_arcgis_v2.py`, `train_dinov3_model_v2.py`)**
   - 使用 `arcgis.learn.prepare_data` 讀取 ArcGIS Export Training Data。
   - 建立 DINOv3 segmentation network。
   - 支援 3-band / 4-band / spectral adapter。
   - 輸出 checkpoint、EMD、DLPK、metrics。

3. **推論 PRF (`dinov3_inference_v2.py`)**
   - 打包在 DLPK 中。
   - ArcGIS Pro `Classify Pixels Using Deep Learning` 會載入此 PRF。
   - 支援 repo snapshot、checkpoint channel probe、class value mapping、radiometry override 與 diagnostic output modes。

## 為何需要 external subprocess

ArcGIS Pro GUI 程序直接載入 `torch / CUDA / DINOv3 / arcgis.learn` 時，若 native DLL 有衝突，可能導致 ArcGIS Pro crash。external subprocess 可把風險隔離在子程序，並保留 log。

## 已知實測關鍵點

- `arcgis.learn` 可被 `spconv.pytorch` native crash 影響。
- DINOv3 SAT-493M `vitl16` 權重可能是 4-channel patch embedding。
- 16-bit 影像推論應注意 DN scale，而不是轉 8-bit stretch。
- 推論輸出需將 model raw index 對應回 ArcGIS EMD class value。
