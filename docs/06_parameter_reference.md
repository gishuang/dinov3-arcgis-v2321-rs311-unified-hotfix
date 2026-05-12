# 06. ArcGIS Pro 工具參數完整說明

此章節對應 `.pyt` 每個欄位前方資訊圖示的說明文字。

| 參數 | 建議值 | 說明 |
|---|---|---|
| Input Training Data Folder(s) | 多選 Export Training Data folder | ArcGIS Pro `Export Training Data For Deep Learning` 的輸出資料夾。可多選。多資料夾需 class schema 一致。 |
| Output Root Folder | `D:\temp\...` | 訓練成果輸出根目錄。工具會建立 `_training_logs` 與 `_deployment`。 |
| Local DINOv3 Repository Folder | `C:\src\dinov3` | DINOv3 官方 repo 本機路徑。DLPK 打包時會建立 repo snapshot。 |
| DINOv3 Backbone Weights (.pth) | `C:\weights\...sat493m...pth` | DINOv3 backbone 權重。SAT-493M 4-channel 權重通常搭配 `Extract Bands=0,1,2,3`。 |
| Model Name | `dinov3_sat493m_seg_v2` | 模型名稱，建議英數與底線。 |
| Chip Size | `224` | chip 大小。ViT-L/16 建議可被 16 整除。 |
| Batch Size | `4`、`8`、穩定後 `16` | 每次送入 GPU 的 chip 數。過大會增加 VRAM 需求。 |
| Epochs | `20` | 訓練輪數。快速 smoke test 可用 3~5。 |
| Learning Rate | `0.0001` | backbone 凍結時常用 1e-4。 |
| Validation Split Percentage | `0.2` | 驗證比例。 |
| Random Seed | `42` | 固定資料分割與訓練隨機性。 |
| Freeze Backbone | `true` | 初期建議凍結 DINOv3 backbone，只訓練 decoder/adapter。 |
| Decoder Type | `fpn_lite` | 分割 decoder。建議 `fpn_lite`。 |
| Decoder Channels | `256` | decoder feature width，不是 class count 或 band count。 |
| Decoder Dropout | `0.1` | dropout 比例。 |
| Loss Mode | `ce` 或 `focal` | 背景比例高時建議 `focal`。 |
| Class Weights | `0.5,1,1,3,2,5,2` | 逗號分隔，長度需等於 class 數。用於提高少數類別權重。 |
| Focal Gamma | `2` | focal loss gamma。 |
| Ignore Index | `-100` 或 `0` | 訓練時忽略的 label。若 0 是 NoData 且不想學背景，可設 0。 |
| Use Early Stopping | `true` | 啟用 early stopping。 |
| Early Stopping Patience | `5` | validation 未改善時等待輪數。 |
| Evaluate After Training | `true` | 訓練後輸出 validation metrics。 |
| Max Validation Batches | 空白 | 限制驗證 batch 數。空白表示完整驗證。 |
| Extract Bands | `0,1,2,3` | 輸入波段索引，0 起算。4-band SAT-493M 建議 `0,1,2,3`。 |
| Inference Input Scale | `auto`、`0_65535`、`max` | DLPK 推論時 raw pixel 轉 float 的尺度。 |
| Max Input Value | `15000` | `input_scale=max` 時使用。需依 16-bit DN 分布調整。 |
| Model Padding | `0` | 推論 tile padding。 |
| Confidence Threshold | `0` | 信心門檻。語意分割初期建議 0，不過濾。 |
| Run Environment Check | `true` | 訓練前檢查 repo、weights、torch forward。 |
| Aggressive Repo Trim | `true` | 打包 DLPK 時精簡 DINOv3 repo snapshot。 |
| Input Channels | 空白 | 空白時等於 Extract Bands 數。 |
| Input Adapter | `auto` | 自動判斷是否需要 learned 1x1/3x3 spectral adapter。 |
| Training Input Scale | `fastai` | ArcGIS `prepare_data` 常建議使用 fastai。若確定 raw DN tensor，才用 `max`。 |
| Execution Mode | `external_subprocess` | 建議值。避免 ArcGIS Pro GUI 直接 import heavy modules。 |
| ArcGIS Pro DL Python Executable | `...\arcgispro-py3-DL-2DTEST\python.exe` | 指定可 import `arcgis.learn` 的 ArcGIS Pro DL clone。 |
