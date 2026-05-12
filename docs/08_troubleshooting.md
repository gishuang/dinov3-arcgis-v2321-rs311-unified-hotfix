# 08. Troubleshooting

## 1. `invalid choice: max`

症狀：

```text
train_dinov3_model_v2.py: error: argument --input-scale: invalid choice: 'max'
```

原因：v2.32 CLI parser 漏掉 `max` 選項。

解法：使用 v2.32.1 hotfix。

## 2. `arcgis.learn` import access violation

症狀：

```text
import arcgis.learn
→ -1073741819 / 0xC0000005
```

若 `spconv.pytorch` 也 crash，建議建立 `arcgispro-py3-DL-2DTEST` 並停用 spconv。DINOv3 2D segmentation 不需要 spconv。

## 3. DLPK 初始化：repo snapshot not found

症狀：

```text
FileNotFoundError: DINOv3 repo snapshot not found: ...\dinov3_repo
```

解法：使用 v2.32.1 重新訓練輸出 DLPK，或用 `repair_existing_dlpk_v2_32_1.py --repo-dir C:\src\dinov3` 修補既有 DLPK。

## 4. DLPK 初始化：3-channel / 4-channel mismatch

症狀：

```text
size mismatch for backbone.patch_embed.proj.weight
checkpoint [1024,4,16,16], current model [1024,3,16,16]
```

解法：使用 v2.32.1 PRF；它會從 checkpoint probe backbone input channels。

## 5. Raster output byte size mismatch

症狀：

```text
incoming 50176 bytes, expected 100352 bytes
```

原因：PRF output dtype 與 ArcGIS 預期 dtype 不一致。

解法：使用 v2.32.1 output dtype-safe PRF。

## 6. Parallel processing job timed out

建議：

```text
Parallel Processing Factor: 0
batch_size 1
```

DINOv3 ViT-L 模型大，多個 parallel worker 可能同時載入模型造成 GPU 競爭。

## 7. 可以推論但全是 0

排查順序：

1. 用 `output_mode raw_index` 看 raw prediction 是否全 0。
2. 用 `output_mode confidence` 看模型是否低信心。
3. 用 `input_scale max;max_input_value 15000` 測 16-bit scale。
4. 檢查 `per_class_metrics.csv` 的 class 0 support 是否過高。
5. 重訓時使用 `focal`、`class weights`、必要時 `ignore_index=0`。

## 8. Safe exit 後仍顯示 Failed

若 log 已有：

```text
Training and packaging complete.
DLPK TO USE IN ARCGIS PRO:
```

代表訓練與打包其實已完成。v2.32.1 已加入 safe exit 與 log-based success guard，正常情況不應再把成功結果標為 Failed。
