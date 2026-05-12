# DINOv3 ArcGIS Toolkit v2.32.1-rs311-unified-hotfix

本版整理 v2.24~v2.31 的修正，目的為讓 ArcGIS Pro 3.6.x / Python 3.13 的 DINOv3 SAT-493M 語意分割訓練與 DLPK 推論流程更穩定。

## 本版重點

- 統一工具箱、訓練核心、PRF、EMD、manifest 與 repair script 的版本標籤為 `v2.32.1-rs311-unified-hotfix`。
- `DINOv3_ArcGIS_Trainer_v2.pyt` 每個參數均加入 `description`，ArcGIS Pro 參數列左側資訊圖示可顯示簡易說明。
- 保留 external subprocess 架構，避免 ArcGIS Pro GUI 直接 import torch / CUDA / DINOv3。
- 保留 safe-exit：訓練輸出 DLPK 後使用安全結束策略，避免 Python finalization native crash 將已成功訓練標成失敗。
- 保留 PRF 修正：repo snapshot、4-channel backbone probe、output dtype、class value mapping、radiometry override、parallel timeout-safe、diagnostic output modes。
- Inference Input Scale 與 Training Input Scale 新增/保留 `max` 模式，可搭配 Max Input Value，例如 15000。

## 建議環境

- ArcGIS Pro 3.6.x
- Deep Learning Libraries 3.6
- Python 3.13 ArcGIS Pro DL clone，例如：
  `C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe`
- 若 `arcgis.learn` 被 `spconv.pytorch` crash 影響，請使用已停用 `spconv` 的 2DTEST clone 進行 2D pixel classification。

## 建議訓練參數

### 4-band SAT-493M / RGB+NIR

```text
Extract Bands: 0,1,2,3
Input Adapter: auto
Input Channels: 空白
Training Input Scale: fastai
Inference Input Scale: 0_65535 或 max
Max Input Value: 視資料 DN，例如 15000
Batch Size: 4 或 8
Decoder Channels: 256
Execution Mode: external_subprocess
```

### NoData / background 為 0 且比例過高

```text
Loss Mode: focal
Ignore Index: 0
Class Weights: 0,2,3,3,2,5,2
Epochs: 20 起跳
```

## 既有 DLPK 修補

若已訓練好舊版 DLPK，可用：

```bat
C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe ^
  D:\python程式資料區\dinov3_arcgis_v2_32_rs311_unified_toolkit\repair_existing_dlpk_v2_32.py ^
  --input D:\temp\Dinov3_model_test2026\D0508\dinov3_sat493m_seg_v2_deployment\dinov3_sat493m_seg_v2.dlpk ^
  --output D:\temp\Dinov3_model_test2026\D0508\dinov3_sat493m_seg_v2_deployment\dinov3_sat493m_seg_v2_v232.dlpk ^
  --repo-dir C:\src\dinov3
```

## 推論建議

Classify Pixels Using Deep Learning：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale auto
```

若 16-bit 影像全部偏向 0，可測：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale max;max_input_value 15000
```

Environments：

```text
Processor Type: GPU
Parallel Processing Factor: 0
```

## 診斷輸出模式

推論時可臨時使用：

```text
output_mode raw_index
output_mode class_value
output_mode confidence
output_mode nonzero_class_value;background_index 0
```

用來判斷模型是否真的全部預測 0，或只是 class value / NoData 顯示問題。
