# 07. 16-bit、4-band 與未來 hyperspectral 支援建議

## 1. DINOv3 是否支援 16-bit？

深度學習模型最終輸入通常是 `float tensor`，不是直接以 `uint16` 做卷積。所謂支援 16-bit，實務上是指：

- 保留 16-bit DN 的相對關係。
- 不使用顯示用 8-bit stretch 當模型輸入。
- 使用固定或一致的 radiometric normalization，例如 `value / 65535` 或 `value / 15000`。

## 2. 訓練與推論尺度一致

如果訓練資料透過 `arcgis.learn.prepare_data` 進入 fastai pipeline，通常建議：

```text
Training Input Scale: fastai
```

推論時依影像 DN 分布設定：

```text
Inference Input Scale: 0_65535
```

或：

```text
Inference Input Scale: max
Max Input Value: 15000
```

`max_input_value=15000` 適用於有效 DN 大致落在 0~15000 的 16-bit 影像。

## 3. 4-band SAT-493M

若權重為 4-channel patch embedding，建議：

```text
Extract Bands: 0,1,2,3
Input Adapter: auto
Input Channels: 空白
```

理想 log：

```text
selected_input_channels=4 | backbone_input_channels=4 | input_adapter=none
```

若只想用 RGB 三波段餵 4-channel backbone，則：

```text
Extract Bands: 0,1,2
Input Adapter: auto
```

工具會嘗試使用 learned adapter 將 3-channel input 投影至 backbone 需要的 channel 數。

## 4. 高光譜 / 多光譜擴充

v2.32.1 已有 `Input Adapter` 架構，但高光譜正式支援仍建議加入：

- band selection UI 或 band group preset
- per-band normalization statistics
- spectral adapter 更完整的保存與 PRF 推論支援
- 訓練/推論一致的 radiometry metadata
- 大量波段時的 tile memory 控制

## 5. 全部輸出 0 的常見原因

- class 0 / background 比例過高。
- 0 被當作真實類別學習，而不是 NoData mask。
- 推論影像 DN 與訓練 chip scale 不一致。
- band order 不一致。
- class mapping 未正確對應 EMD class value。
- ArcGIS 顯示把 0 當 NoData。

建議先用診斷模式 `raw_index`、`confidence`、`nonzero_class_value` 排查。
