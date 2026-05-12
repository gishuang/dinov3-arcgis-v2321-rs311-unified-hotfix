# DINOv3 ArcGIS Toolkit v2.32.1-rs311-unified-hotfix

此工具包將 DINOv3 SAT-493M backbone 整合到 ArcGIS Pro 的語意分割訓練與 DLPK 推論流程，目標是支援航遙測 8-bit / 16-bit、3-band / 4-band 影像資料，並以 ArcGIS Pro `Classify Pixels Using Deep Learning` 執行推論。

> 狀態：實驗整合版。此版已收斂到 `v2.32.1-rs311-unified-hotfix`，保留 external subprocess、safe exit、4-channel backbone runtime probe、DLPK repo snapshot、output dtype fix、class value mapping、radiometry override、diagnostic inference modes 與 parallel timeout-safe 等修正。

## 專案重點

- ArcGIS Pro GUI 不直接 import `torch / CUDA / DINOv3`，訓練改用 `external_subprocess` 執行，降低 ArcGIS Pro crash 機率。
- 支援多個 ArcGIS Export Training Data folder 作為同一次訓練來源。
- 支援 SAT-493M 4-channel backbone，`Extract Bands=0,1,2,3` 時可走 4-channel native path。
- 支援 16-bit 推論尺度控制：`input_scale auto / 0_65535 / max` 與 `max_input_value`。
- 支援診斷推論輸出：`class_value / raw_index / confidence / nonzero_class_value`。
- 針對 ArcGIS Pro 3.6.x + Python 3.13 + Deep Learning Libraries 3.6 的實測環境整理。

## 重要限制

- 本 repo 不應包含 DINOv3 官方權重 `.pth`、原始影像、訓練資料、輸出模型或 DLPK 成果。請自行存放於本機路徑，例如 `C:\weights` 與 `D:\temp`。
- `arcgispro-py3-DL` 若因 `spconv.pytorch` 造成 `arcgis.learn` import crash，建議建立 2D-only clone 並停用 `spconv`，詳見 `docs/03_arcgis_pro_dl_environment.md`。
- 若使用 `max_input_value=15000`，要確認此值與推論影像 DN 分布相符；它不是 8-bit 顯示 stretch，而是模型輸入 float tensor 的固定正規化參數。

## 快速開始

1. 建立外部 DINOv3 驗證環境 `dinov3-rs311`。
2. Clone DINOv3 repo 到 `C:\src\dinov3`。
3. 準備 DINOv3 SAT-493M 權重，例如 `C:\weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth`。
4. 建立 ArcGIS Pro Deep Learning 2D 測試環境，例如 `arcgispro-py3-DL-2DTEST`。
5. 在 ArcGIS Pro 載入 `DINOv3_ArcGIS_Trainer_v2.pyt`。
6. 訓練時使用：

```text
Execution Mode: external_subprocess
ArcGIS Pro DL Python Executable:
C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe
Extract Bands: 0,1,2,3
Input Adapter: auto
Decoder Channels: 256
Training Input Scale: fastai
Inference Input Scale: 0_65535 或 max
```

7. 推論時使用 ArcGIS Pro `Classify Pixels Using Deep Learning`，建議：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale auto
```

若 16-bit 影像全部偏 0，可改測：

```text
padding 0;batch_size 1;tta none;confidence_threshold 0;use_half false;input_scale max;max_input_value 15000
```

## 文件

- `docs/01_overview.md`：架構與版本說明
- `docs/02_miniconda_dinov3_env.md`：外部 DINOv3 rs311 環境
- `docs/03_arcgis_pro_dl_environment.md`：ArcGIS Pro DL clone 與 spconv 問題處理
- `docs/04_training_workflow.md`：訓練流程
- `docs/05_inference_workflow.md`：DLPK 推論流程
- `docs/06_parameter_reference.md`：ArcGIS Pro 工具參數說明
- `docs/07_16bit_multiband_guidance.md`：16-bit / 4-band / hyperspectral 評估
- `docs/08_troubleshooting.md`：錯誤排除
- `docs/09_github_publish.md`：GitHub 發布步驟
- `docs/10_release_notes.md`：版本紀錄

## 建議 repository 名稱

```text
dinov3-arcgis-v2321-rs311-unified-hotfix
```

## License

尚未指定授權。若要公開發布，請先確認 DINOv3、Esri ArcGIS API/Deep Learning Libraries、PyTorch 及專案內所有第三方依賴之授權條款，再加入合適的 LICENSE。
