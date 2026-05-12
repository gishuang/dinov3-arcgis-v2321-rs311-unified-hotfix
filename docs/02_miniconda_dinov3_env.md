# 02. Miniconda DINOv3 rs311 環境建置

此環境用於驗證 DINOv3 repo、權重與 PyTorch forward，不直接取代 ArcGIS Pro 的 Python。

## 1. 建立環境

```bat
conda create -n dinov3-rs311 python=3.11 -y
conda activate dinov3-rs311
python -m pip install --upgrade pip
```

## 2. 安裝 PyTorch GPU 版

以 CUDA 12.1 wheel 為例：

```bat
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

若僅做 CPU 測試，可改用 PyTorch CPU 版。

## 3. 安裝輔助套件

```bat
python -m pip install "transformers==4.56.2" "huggingface_hub>=0.34,<1.0"
python -m pip install pillow numpy opencv-python tqdm matplotlib timm safetensors scikit-image scikit-learn
```

Windows CMD 中含有 `<`、`>` 的版本條件必須加雙引號，否則會被解讀成重新導向。

## 4. Clone DINOv3 repo

建議放在：

```text
C:\src\dinov3
```

```bat
mkdir C:\src
cd /d C:\src
git clone https://github.com/facebookresearch/dinov3.git
cd /d C:\src\dinov3
python -m pip install -e .
```

若 `git` 無法辨識，請先安裝 Git for Windows，或使用 ZIP 下載法並確保 `C:\src\dinov3\hubconf.py` 存在。

## 5. 驗證

```bat
conda activate dinov3-rs311
python -c "import dinov3, os; print('dinov3 OK'); print(os.path.dirname(dinov3.__file__))"
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 6. 權重路徑

建議將 DINOv3 SAT-493M 權重放在：

```text
C:\weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth
```

權重不應提交至 GitHub repo。
