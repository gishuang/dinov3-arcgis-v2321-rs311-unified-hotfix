# 04. 訓練流程

## 1. 準備 ArcGIS 訓練資料

使用 ArcGIS Pro `Export Training Data For Deep Learning` 建立訓練資料。若分多個資料夾管理類別，例如：

```text
D:\temp\forest_IPCC_IPCCName\IPCCName_01
D:\temp\forest_IPCC_IPCCName\IPCCName_02
...
D:\temp\forest_IPCC_IPCCName\IPCCName_06
```

可以在 v2.32.1 的 `Input Training Data Folder(s)` 多選。

## 2. 資料一致性要求

多資料夾必須盡量一致：

- chip size 一致，例如 224。
- label 格式一致。
- class schema 一致。
- class value 不應在不同資料夾代表不同類別。
- 影像 band order 一致。

## 3. 建議訓練參數：4-band SAT-493M

```text
Input Training Data Folder(s): 選取多個 Export Training Data folder
Output Root Folder: D:\temp\Dinov3_model_test2026\D0511_2
Local DINOv3 Repository Folder: C:\src\dinov3
DINOv3 Backbone Weights: C:\weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
Chip Size: 224
Batch Size: 4 或 8；確認穩定後可試 16
Epochs: 20
Learning Rate: 0.0001
Freeze Backbone: true
Decoder Type: fpn_lite
Decoder Channels: 256
Loss Mode: focal
Extract Bands: 0,1,2,3
Input Adapter: auto
Training Input Scale: fastai
Inference Input Scale: max 或 0_65535
Max Input Value: 依推論影像 DN，例如 15000
Execution Mode: external_subprocess
ArcGIS Pro DL Python Executable: C:\Users\user\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL-2DTEST\python.exe
```

## 4. 背景 / NoData 比例過高

如果 class 0 是 background 或 NoData，且比例過高，建議：

```text
Loss Mode: focal
Ignore Index: 0
Class Weights: 0,2,3,3,2,5,2
```

若仍希望 0 類也保留一些學習訊號，可改成：

```text
Ignore Index: -100
Class Weights: 0.5,1,1,3,2,5,2
```

## 5. 訓練輸出

完成後會輸出：

```text
<Output Root>\<Model Name>_training_logs\fit_report.json
<Output Root>\<Model Name>_training_logs\validation_metrics.json
<Output Root>\<Model Name>_training_logs\epoch_progress.csv
<Output Root>\<Model Name>_training_logs\per_class_metrics.csv
<Output Root>\<Model Name>_deployment\<Model Name>.pth
<Output Root>\<Model Name>_deployment\<Model Name>.emd
<Output Root>\<Model Name>_deployment\<Model Name>.dlpk
```

## 6. 判讀訓練結果

- `accuracy` 容易受背景比例影響，不應單獨判斷好壞。
- `dice` 與 `per_class_metrics.csv` 更能反映語意分割效果。
- 若 `NoData / Background` support 過高，請重新平衡樣本或調整 loss / class weights。
