# 05. DLPK 推論流程

## 1. ArcGIS Pro 工具

使用：

```text
Classify Pixels Using Deep Learning
```

`Model Definition` 選擇訓練輸出的 DLPK：

```text
<Output Root>\<Model Name>_deployment\<Model Name>.dlpk
```

## 2. 建議 Arguments

先以保守設定測試小範圍：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale auto
```

16-bit 影像若全部偏向 0，可測：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale max;max_input_value 15000
```

其他常測最大值：

```text
max_input_value 4095
max_input_value 10000
max_input_value 15000
max_input_value 16383
max_input_value 65535
```

## 3. Environments

建議：

```text
Processor Type: GPU
Parallel Processing Factor: 0
```

DINOv3 ViT-L 模型較大，ArcGIS parallel workers 同時載入模型可能造成 GPU 競爭與 timeout。

## 4. 診斷輸出模式

若輸出全部為 0，可使用 v2.32.1 PRF 診斷模式：

```text
output_mode raw_index
output_mode class_value
output_mode confidence
output_mode nonzero_class_value;background_index 0
```

範例：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale auto;output_mode raw_index;debug_first_tile true
```

判讀：

- `raw_index` 全 0：模型 logits 真的偏向 class 0。
- `class_value` 全 0：可能是 class mapping 或 class 0 顯示為 NoData。
- `confidence` 很低：推論影像 radiometry 或 band order 可能不一致。
- `nonzero_class_value` 有結果：背景類太強，需調整訓練 loss / class weights。

## 5. 既有 DLPK 修補

若舊 DLPK 缺少新版 PRF，可用：

```bat
C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe ^
  D:\python程式資料區\dinov3_arcgis_v2_32_1_rs311_unified_hotfix_toolkit\repair_existing_dlpk_v2_32_1.py ^
  --input D:\temp\Dinov3_model_test2026\D0508\dinov3_sat493m_seg_v2_deployment\dinov3_sat493m_seg_v2.dlpk ^
  --output D:\temp\Dinov3_model_test2026\D0508\dinov3_sat493m_seg_v2_deployment\dinov3_sat493m_seg_v2_v2321.dlpk ^
  --repo-dir C:\src\dinov3
```
