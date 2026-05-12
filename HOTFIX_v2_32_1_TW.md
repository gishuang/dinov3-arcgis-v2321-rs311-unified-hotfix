# DINOv3 ArcGIS v2.32.1-rs311-unified-hotfix

此版修正 v2.32 的 CLI parser 問題：

- `.pyt` 參數已允許 `Inference Input Scale = max`。
- 但 `train_dinov3_model_v2.py` 的 `--input-scale` choices 漏掉 `max`，造成 `invalid choice: max`。
- 本版已將 `--input-scale` 與 `--training-input-scale` 加入 `max`。

建議：

- 若只想讓推論端使用 `max_input_value=15000`，保持 `Training Input Scale = fastai`，`Inference Input Scale = max`，`Max Input Value = 15000`。
- 若訓練 tensor 確定是 raw DN，才把 `Training Input Scale` 改為 `max`。ArcGIS `prepare_data()` 常見情況仍建議使用 `fastai`。
