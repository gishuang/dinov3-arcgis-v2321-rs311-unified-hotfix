# 09. GitHub 發布步驟

## 建議 repo 名稱

```text
dinov3-arcgis-v2321-rs311-unified-hotfix
```

## 不建議提交的內容

請勿提交：

- `.pth`、`.pt`、`.dlpk`、`.emd`、`.tif`、`.gdb`
- 原始影像
- training data chips / labels
- 權重與模型成果
- 任何含專案敏感資料的 log

`.gitignore` 已預先排除常見大型檔案與模型成果。

## 在 GitHub 建立空 repo 後推送

在本機此資料夾執行：

```bat
cd /d D:\python程式資料區\dinov3-arcgis-v2321-rs311-unified-hotfix

git init
git add .
git commit -m "Initial release: DINOv3 ArcGIS v2.32.1 rs311 unified hotfix"
git branch -M main
git remote add origin https://github.com/<OWNER>/dinov3-arcgis-v2321-rs311-unified-hotfix.git
git push -u origin main
```

## 建議設為 private repo

此工具涉及 ArcGIS Pro、Deep Learning Libraries、DINOv3 權重與專案資料流程，初期建議使用 private repository。公開前請檢查授權、資料敏感性與第三方依賴。

## GitHub Release 建議

建立 release：

```text
Tag: v2.32.1-rs311-unified-hotfix
Title: DINOv3 ArcGIS Toolkit v2.32.1-rs311-unified-hotfix
```

Release notes 可使用 `docs/10_release_notes.md`。
