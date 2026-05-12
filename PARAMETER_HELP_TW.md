# v2.32 ArcGIS Pro 工具參數資訊說明

本檔內容與 `.pyt` 內各參數的 `description` 對應，供使用者在 ArcGIS Pro 參數列左側資訊圖示以外快速查閱。

| 參數 | 說明 |
|---|---|
| Input Training Data Folder(s) | 選擇 ArcGIS Pro 的 Export Training Data For Deep Learning 輸出資料夾，可多選。每個資料夾應使用一致的 class schema、chip size 與 label 格式。 |
| Output Root Folder | 模型訓練成果輸出根目錄。工具會在此建立 training_logs 與 deployment 資料夾。 |
| Local DINOv3 Repository Folder | 本機 DINOv3 官方 repository 路徑，例如 `C:\src\dinov3`。 |
| DINOv3 Backbone Weights (.pth) | DINOv3 backbone 權重 `.pth`。SAT-493M 4-channel 權重通常搭配 `Extract Bands=0,1,2,3`。 |
| Model Name | 輸出的模型名稱，會用於資料夾與檔名。建議使用英數與底線。 |
| Chip Size | 訓練 chip 大小。ViT-L/16 需可被 16 整除。 |
| Batch Size | 每次訓練送入 GPU 的 chip 數。ViT-L 建議先用 2、4 或 8。 |
| Epochs | 訓練輪數。快速測試可用 3~5；正式訓練建議 20 起跳。 |
| Learning Rate | 學習率。Freeze Backbone=True 時常用 0.0001。 |
| Validation Split Percentage | 驗證資料比例，0.2 表示 20%。 |
| Random Seed | 隨機種子，用於資料分割與重現性。 |
| Freeze Backbone | 是否凍結 DINOv3 backbone。建議先 True。 |
| Decoder Type | 分割 decoder 類型，建議 `fpn_lite`。 |
| Decoder Channels | decoder feature width，不是類別數或波段數。建議 256。 |
| Decoder Dropout | dropout 比例，常用 0.1。 |
| Loss Mode | `ce` 或 `focal`。背景比例高時建議 `focal`。 |
| Class Weights | 類別權重，以逗號分隔，例如 `0,2,3,3,2,5,2`。 |
| Focal Gamma | focal loss gamma，常用 2.0。 |
| Ignore Index | 訓練時忽略的 label 值。若 0 是 NoData/background，可設 0。 |
| Use Early Stopping | 是否啟用 early stopping。 |
| Early Stopping Patience | early stopping 等待輪數。 |
| Evaluate After Training | 訓練後計算 validation metrics。 |
| Max Validation Batches | 限制驗證 batch 數；空白表示完整驗證。 |
| Extract Bands | 輸入波段索引，以 0 起算，例如 `0,1,2,3`。 |
| Inference Input Scale | 推論像素尺度：`auto`、`0_255`、`0_65535`、`max` 等。 |
| Max Input Value | `input_scale=max` 時使用的固定最大值，例如 15000。 |
| Model Padding | 推論 tile padding，通常 0。 |
| Confidence Threshold | 信心門檻，0 表示不過濾。 |
| Run Environment Check | 訓練前檢查環境與 forward。 |
| Aggressive Repo Trim | 打包 DLPK 時精簡 DINOv3 repo snapshot。 |
| Input Channels | 實際輸入 channel 數，空白時等於 Extract Bands 數。 |
| Input Adapter | `auto`、`none`、`learned_1x1`、`learned_3x3`；建議 `auto`。 |
| Training Input Scale | 訓練 batch 尺度，arcgis.learn 通常用 `fastai`。 |
| Execution Mode | 建議 `external_subprocess`。 |
| ArcGIS Pro DL Python Executable | 外部訓練 Python.exe，建議指定能 import arcgis.learn 的 DL clone。 |
