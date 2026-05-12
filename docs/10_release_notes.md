# 10. Release Notes

## v2.32.1-rs311-unified-hotfix

修正：

- `train_dinov3_model_v2.py` CLI parser 加入 `max`：
  - `--input-scale max`
  - `--training-input-scale max`
- 保留 v2.32 的 unified versioning 與參數 help 說明。
- 保留已驗證的 ArcGIS Pro 3.6.x / Python 3.13 external subprocess 架構。

## v2.32-rs311-unified

整理：

- 統一 `.pyt`、training core、PRF、EMD、manifest、repair script 版本標籤。
- 在 ArcGIS Pro 參數資訊圖示加入說明文字。
- 保留 DLPK 推論端修正：repo snapshot、4-channel backbone probe、output dtype-safe、class mapping、radiometry override、diagnostic output modes、parallel timeout-safe。

## 重要歷程摘要

- 多資料夾訓練輸入。
- 16-bit / 4-band / spectral adapter 架構。
- external subprocess crash-safe。
- ArcGIS Pro DL clone 中 `spconv.pytorch` crash 的 2DTEST workaround。
- DINOv3 SAT-493M 4-channel backbone checkpoint 與 PRF 推論端 channel mismatch 修正。
- DLPK 推論 output dtype mismatch 修正。
- 推論全 0 時的 class mapping / radiometry / diagnostic mode 修正。
