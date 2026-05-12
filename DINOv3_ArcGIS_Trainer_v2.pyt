# -*- coding: utf-8 -*-
"""
ArcGIS Pro Python toolbox for DINOv3 ArcGIS Pro Segmentation Toolkit v2.32.1-rs311-unified-hotfix.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import arcpy

CURRENT_DIR = Path(__file__).resolve().parent
TOOL_VERSION = "v2.32.1-rs311-unified-hotfix"


PARAMETER_HELP_TW = {
    "input_training_data": "選擇 ArcGIS Pro 的 Export Training Data For Deep Learning 輸出資料夾，可多選。每個資料夾應使用一致的 class schema、chip size 與 label 格式。若用 0 作 NoData/background，建議搭配 Ignore Index=0。",
    "output_root": "模型訓練成果輸出根目錄。工具會在此建立 training_logs 與 deployment 資料夾，包含 .pth、.emd、.dlpk、manifest、metrics 與 CSV。建議使用空的新資料夾，避免混到舊版成果。",
    "repo_dir": "本機 DINOv3 官方 repository 路徑，例如 C:\\src\\dinov3。資料夾內應有 dinov3、hubconf.py 等內容，工具會將必要 repo snapshot 打包進 DLPK。",
    "backbone_weights": "DINOv3 backbone 權重 .pth。若使用 SAT-493M 4-channel 權重，Extract Bands 通常設定 0,1,2,3。權重 channel 數會在 runtime 自動偵測。",
    "model_name": "輸出的模型名稱，會用於 training log folder、deployment folder、.emd 與 .dlpk 檔名。建議使用英數、底線，不要含空白或特殊符號。",
    "chip_size": "訓練 chip 大小。ViT-L/16 需可被 16 整除；常用 224、256、384。必須盡量與 Export Training Data 時的 chip size 一致。",
    "batch_size": "每次訓練送入 GPU 的 chip 數。ViT-L 建議先用 2、4 或 8；若 VRAM 充足再提高。Batch size 過大可能造成 GPU OOM 或推論 timeout。",
    "epochs": "訓練輪數。若忽略 0 類並使用 focal loss，建議至少 20 起跳；快速測試可用 3~5。Early Stopping 開啟時可能提早停止。",
    "learning_rate": "學習率。Freeze Backbone=True 時常用 1e-4；若解凍 backbone，建議降低到 1e-5 或更小。",
    "val_split_pct": "驗證資料比例，0.2 表示 20% 影像做 validation。多資料夾輸入時仍由 arcgis.learn 依資料來源進行分割。",
    "seed": "隨機種子，用於資料分割與訓練重現。相同資料與環境下，固定 seed 可提升結果可重現性。",
    "freeze_backbone": "是否凍結 DINOv3 backbone。建議先 True，只訓練 decoder 與 adapter；資料量大或 domain shift 明顯時才考慮 False。",
    "decoder_type": "語意分割 decoder 類型。fpn_lite 較適合一般分類；linear 較簡單、速度快但表現可能較弱。",
    "decoder_channels": "decoder feature width，不是類別數，也不是影像波段數。建議 256；低於 32 通常不合理。",
    "dropout": "decoder dropout 比例，用於降低 overfitting。常用 0.1；資料少或過擬合時可提高到 0.2。",
    "loss_mode": "損失函數。ce 為標準 cross entropy；focal 適合類別不平衡或背景比例很高的資料。",
    "class_weights": "各類別權重，以逗號分隔，數量需等於 class 數。例如 0,2,3,3,2,5,2 可忽略/降低 0 類並提高少數類別權重。空白表示不使用。",
    "focal_gamma": "Focal loss 的 gamma。常用 2.0；數值越高越強調困難樣本。只在 Loss Mode=focal 時有主要意義。",
    "ignore_index": "訓練時忽略的 label 值。若 0 代表 NoData/background 且不希望模型學它，設 0；若沒有忽略類別，使用 -100。",
    "use_early_stopping": "是否啟用 early stopping。若驗證 loss 長時間不改善，會提前停止訓練。正式訓練建議開啟。",
    "early_stopping_patience": "early stopping 等待輪數。例如 5 表示 validation 指標 5 個 epoch 未改善才停止。",
    "evaluate_after_training": "訓練後是否計算 validation metrics，包含 per-class IoU/F1/precision/recall。建議開啟，方便判斷是否背景比例過高。",
    "max_validation_batches": "限制驗證 batch 數。空白表示完整驗證；若資料量很大且只想快速測試，可填 5 或 10。",
    "extract_bands": "輸入波段索引，以 0 起算。例如 RGB=0,1,2；RGB+NIR=0,1,2,3。需與訓練資料與推論影像 band order 一致。",
    "input_scale": "推論時 raw pixel 轉成 0~1 的方式。auto 自動判斷；0_65535 適合 full-range UInt16；max 會搭配 Max Input Value，例如 15000。需盡量與訓練尺度一致。",
    "max_input_value": "當 Input Scale 或 Training Input Scale 使用 max 時的固定最大值。例如 DMCIII 16-bit 有效 DN 約 15000，就填 15000。空白表示不使用。",
    "model_padding": "推論時 tile 邊界 padding。通常 0 即可；若 tile 邊界有接縫或分類破碎，可測 16、32 或 64。",
    "confidence_threshold": "信心門檻。0 表示不過濾，直接輸出最高分 class。若只想保留高信心結果，可提高，例如 0.3。",
    "run_environment_check": "訓練前是否檢查 Python、torch、DINOv3 repo、權重、CUDA 與 forward。建議開啟；已確認環境穩定後可關閉以節省時間。",
    "aggressive_repo_trim": "打包 DLPK 時是否精簡 DINOv3 repo snapshot，排除 .git、notebooks、docs、pycache 等不必要內容。建議開啟。",
    "input_channels": "模型實際輸入 channel 數。空白時等於 Extract Bands 數。只有在自訂 adapter 或高光譜 subset 時才需手動指定。",
    "input_adapter": "輸入光譜 adapter。auto 會依 backbone channel 自動選 native 或 learned_1x1；none 僅適合 selected bands 與 backbone channels 完全一致。",
    "training_input_scale": "訓練 batch 的尺度處理。arcgis.learn/fastai 輸入通常用 fastai；若直接訓練 raw DN，可用 0_255、0_65535 或 max。需與推論尺度一致。",
    "execution_mode": "執行模式。external_subprocess 會在外部 Python 執行訓練，避免 ArcGIS Pro GUI 直接載入 torch/CUDA；建議使用。in_process 僅供除錯。",
    "arcgispro_dl_python": "外部訓練或驗證用 Python.exe。建議指定已可 import arcgis.learn 的 DL 環境，例如 arcgispro-py3-DL-2DTEST\\python.exe。空白時工具會自動搜尋。",
    "out_json": "環境檢查報告 JSON 輸出路徑。空白時會寫到 TEMP 目錄。建議保留，方便比對 Python、torch、CUDA、repo 與 forward test 結果。",
}


def _apply_parameter_help(params):
    for param in params:
        try:
            desc = PARAMETER_HELP_TW.get(param.name)
            if desc:
                param.description = desc
        except Exception:
            pass
    return params


def _add_tool_dir_to_syspath():
    tool_dir = str(CURRENT_DIR)
    if tool_dir not in sys.path:
        sys.path.insert(0, tool_dir)


def _load_core():
    """Lazy-load heavy DINOv3/torch/arcgis.learn modules only after tool execution starts.

    This avoids importing torch, torchvision, arcgis.learn, GDAL-adjacent DLLs, or a
    local DINOv3 repo while ArcGIS Pro is merely browsing/validating the .pyt file.
    """
    _add_tool_dir_to_syspath()
    from dinov3_arcgis_v2 import train_and_package_v2, verify_runtime_environment  # noqa: E402
    return train_and_package_v2, verify_runtime_environment


def _strip_outer_quotes(text):
    text = str(text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1].strip()
    return text


def _split_arcgis_multivalue_paths(value):
    text = str(value or "").strip()
    if not text:
        return []
    parts, buf, quote = [], [], None
    for ch in text:
        if ch in ("'", '"'):
            if quote is None:
                quote = ch
                continue
            if quote == ch:
                quote = None
                continue
        if ch == ";" and quote is None:
            part = _strip_outer_quotes("".join(buf))
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    part = _strip_outer_quotes("".join(buf))
    if part:
        parts.append(part)
    return parts


def _candidate_python_executables():
    candidates = []
    for value in (sys.executable, sys.prefix, getattr(sys, "exec_prefix", None), getattr(sys, "base_prefix", None), os.environ.get("PYTHONHOME"), os.environ.get("CONDA_PREFIX")):
        if value:
            pp = Path(value)
            candidates.append(pp if pp.name.lower() == "python.exe" else pp / "python.exe")
    localapp = os.environ.get("LOCALAPPDATA")
    if localapp:
        candidates.append(Path(localapp) / "ESRI" / "conda" / "envs" / "arcgispro-py3-DL" / "python.exe")
        candidates.append(Path(localapp) / "ESRI" / "conda" / "envs" / "arcgispro-py3-clone-1" / "python.exe")
    candidates.append(Path(r"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"))

    out, seen = [], set()
    for c in candidates:
        try:
            c = c.expanduser().resolve()
        except Exception:
            pass
        key = str(c).lower()
        if key not in seen:
            out.append(c)
            seen.add(key)
    return out


def _resolve_python_executable(explicit_path=None):
    if explicit_path:
        p = Path(str(explicit_path).strip().strip('"')).expanduser()
        if p.is_file():
            return str(p)
        raise RuntimeError(f"指定的 Python Executable 不存在：{p}")
    for p in _candidate_python_executables():
        if p.is_file() and p.name.lower() == "python.exe":
            return str(p)
    raise RuntimeError(
        "找不到可用的 ArcGIS Pro Python Executable。請在工具參數 'ArcGIS Pro DL Python Executable' 指定，例如："
        r"C:\Users\<user>\AppData\Local\ESRI\conda\envs\arcgispro-py3-DL\python.exe"
    )


def _augment_arcgis_subprocess_env(env, run_cmd, messages=None):
    """Prepend ArcGIS Pro and conda-env DLL folders for standalone arcpy/arcgis imports.

    Launching the cloned ArcGIS Pro DL environment python.exe directly from a .pyt
    may not inherit the same PATH that ArcGIS Pro Python Command Prompt prepares.
    arcgis.learn imports arcgis -> arcpy -> ArcGIS native geoprocessing DLLs; missing
    DLL search paths can end in Windows fatal exception 0xe0000001 instead of a
    Python ImportError.  v2.32 makes the subprocess path explicit.
    """
    candidates = []
    try:
        py = Path(str(run_cmd[0])).expanduser()
        if py.name.lower() == "python.exe":
            env_dir = py.parent
            candidates.extend([
                env_dir,
                env_dir / "Scripts",
                env_dir / "Library" / "bin",
                env_dir / "Library" / "usr" / "bin",
                env_dir / "Library" / "mingw-w64" / "bin",
            ])
            env["CONDA_PREFIX"] = str(env_dir)
            env["CONDA_DEFAULT_ENV"] = env_dir.name
    except Exception:
        pass

    pro_root = Path(os.environ.get("ARCGIS_PRO_ROOT", r"C:\Program Files\ArcGIS\Pro"))
    candidates.extend([
        pro_root / "bin",
        pro_root / "Resources" / "ArcPy",
        pro_root / "bin" / "Python" / "Scripts",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Library" / "bin",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Scripts",
    ])

    seen = set()
    prepend = []
    for c in candidates:
        try:
            c = Path(c)
            if c.exists():
                key = str(c).lower()
                if key not in seen:
                    prepend.append(str(c))
                    seen.add(key)
        except Exception:
            pass
    if prepend:
        env["PATH"] = os.pathsep.join(prepend + [env.get("PATH", "")])
        try:
            if messages is not None:
                messages.addMessage("ArcGIS subprocess PATH prepended with:")
                for item in prepend:
                    messages.addMessage("  " + item)
        except Exception:
            pass
    env.setdefault("ESRI_SOFTWARE_CLASS", "Professional")
    return env


def _run_subprocess(cmd, messages, cwd=None, log_path=None):
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    run_cmd = [str(x) for x in cmd]
    # v2.32: keep subprocess output unbuffered but do NOT enable faulthandler by default.
    # ArcPy can emit internally-handled Windows fatal-exception traces when faulthandler is enabled,
    # which makes successful ArcPy imports look like failures. Use manual preflight with -X faulthandler only when needed.
    try:
        exe_name = Path(run_cmd[0]).name.lower() if run_cmd else ""
        if exe_name == "python.exe" and len(run_cmd) > 1 and not str(run_cmd[1]).startswith("-"):
            run_cmd = [run_cmd[0], "-u", *run_cmd[1:]]
    except Exception:
        pass
    try:
        messages.addMessage("External subprocess command:")
        messages.addMessage(" ".join(run_cmd))
    except Exception:
        pass
    env = os.environ.copy()
    # Prevent accidental Miniconda/C:\src path leakage into ArcGIS Pro subprocess unless user explicitly configured it.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONFAULTHANDLER", None)
    env["PYTHONUNBUFFERED"] = "1"
    # Force UTF-8 child-process stdio so Traditional Chinese paths do not crash
    # with UnicodeEncodeError under legacy Windows code pages such as cp1252.
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8:replace"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env = _augment_arcgis_subprocess_env(env, run_cmd, messages)
    proc = subprocess.Popen(
        run_cmd,
        cwd=str(cwd or CURRENT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    with (open(log_path, "a", encoding="utf-8") if log_path else open(os.devnull, "w", encoding="utf-8")) as lf:
        if log_path:
            lf.write("COMMAND: " + " ".join([str(x) for x in cmd]) + "\n\n")
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if log_path:
                lf.write(line + "\n")
                lf.flush()
            try:
                messages.addMessage(line)
            except Exception:
                try:
                    arcpy.AddMessage(line)
                except Exception:
                    pass
    rc = proc.wait()
    if rc != 0:
        # v2.32-rs311 safety net: on some ArcGIS Pro DL environments, native
        # libraries can crash during interpreter finalization after all model
        # artifacts have already been written.  Treat the run as successful only
        # when the subprocess log contains explicit success markers.
        success_markers_found = False
        if log_path and os.path.exists(log_path):
            try:
                log_text = open(log_path, "r", encoding="utf-8", errors="replace").read()
                success_markers_found = (
                    "Training and packaging complete." in log_text
                    and "DLPK TO USE IN ARCGIS PRO:" in log_text
                    and "Deployment checkpoint:" in log_text
                    and "EMD:" in log_text
                )
            except Exception:
                success_markers_found = False
        if success_markers_found:
            warn = (
                f"External subprocess ended with exit code {rc}, but training/export success markers "
                f"were found in the log. Treating as success. Log: {log_path}"
            )
            try:
                messages.addWarningMessage(warn)
            except Exception:
                try:
                    arcpy.AddWarning(warn)
                except Exception:
                    pass
            return 0
        raise RuntimeError(f"External subprocess failed with exit code {rc}. Log: {log_path or '(not saved)'}")
    return rc


class Toolbox(object):
    def __init__(self):
        self.label = "DINOv3 ArcGIS Pro Segmentation Toolkit v2.32.1-rs311-unified-hotfix"
        self.alias = "dinov3segv2"
        self.tools = [TrainDINOv3PixelClassifierV2, VerifyDINOv3EnvironmentV2]


class TrainDINOv3PixelClassifierV2(object):
    def __init__(self):
        self.label = "Train DINOv3 Pixel Classifier v2.32.1-rs311-unified-hotfix"
        self.description = (
            "Train a DINOv3 SAT-493M semantic segmentation model and export a custom .dlpk "
            "for Classify Pixels Using Deep Learning."
        )
        self.canRunInBackground = False

    def getParameterInfo(self):
        params = []

        p0 = arcpy.Parameter(displayName="Input Training Data Folder(s)", name="input_training_data", datatype="DEFolder", parameterType="Required", direction="Input")
        # multiValue=True enables ArcGIS Pro-style selection of multiple Export Training Data folders.
        # valueAsText is passed downstream as a semicolon-delimited path string.
        p0.multiValue = True
        p0.value = r"C:\training_data"
        p1 = arcpy.Parameter(displayName="Output Root Folder", name="output_root", datatype="DEFolder", parameterType="Required", direction="Input")
        p1.value = r"C:\dinov3_output"
        p2 = arcpy.Parameter(displayName="Local DINOv3 Repository Folder", name="repo_dir", datatype="DEFolder", parameterType="Required", direction="Input")
        p2.value = r"C:\src\dinov3"
        p3 = arcpy.Parameter(displayName="DINOv3 Backbone Weights (.pth)", name="backbone_weights", datatype="DEFile", parameterType="Required", direction="Input")
        p3.filter.list = ["pth"]
        p3.value = r"C:\weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        p4 = arcpy.Parameter(displayName="Model Name", name="model_name", datatype="GPString", parameterType="Optional", direction="Input")
        p4.value = "dinov3_sat493m_seg_v2"

        p5 = arcpy.Parameter(displayName="Chip Size", name="chip_size", datatype="GPLong", parameterType="Optional", direction="Input")
        p5.value = 224
        p6 = arcpy.Parameter(displayName="Batch Size", name="batch_size", datatype="GPLong", parameterType="Optional", direction="Input")
        p6.value = 4
        p7 = arcpy.Parameter(displayName="Epochs", name="epochs", datatype="GPLong", parameterType="Optional", direction="Input")
        p7.value = 20
        p8 = arcpy.Parameter(displayName="Learning Rate", name="learning_rate", datatype="GPDouble", parameterType="Optional", direction="Input")
        p8.value = 0.0001
        p9 = arcpy.Parameter(displayName="Validation Split Percentage", name="val_split_pct", datatype="GPDouble", parameterType="Optional", direction="Input")
        p9.value = 0.2
        p10 = arcpy.Parameter(displayName="Random Seed", name="seed", datatype="GPLong", parameterType="Optional", direction="Input")
        p10.value = 42

        p11 = arcpy.Parameter(displayName="Freeze Backbone", name="freeze_backbone", datatype="GPBoolean", parameterType="Optional", direction="Input")
        p11.value = True
        p12 = arcpy.Parameter(displayName="Decoder Type", name="decoder_type", datatype="GPString", parameterType="Optional", direction="Input")
        p12.filter.type = "ValueList"
        p12.filter.list = ["fpn_lite", "linear"]
        p12.value = "fpn_lite"
        p13 = arcpy.Parameter(displayName="Decoder Channels", name="decoder_channels", datatype="GPLong", parameterType="Optional", direction="Input")
        p13.value = 256
        p14 = arcpy.Parameter(displayName="Decoder Dropout", name="dropout", datatype="GPDouble", parameterType="Optional", direction="Input")
        p14.value = 0.1

        p15 = arcpy.Parameter(displayName="Loss Mode", name="loss_mode", datatype="GPString", parameterType="Optional", direction="Input")
        p15.filter.type = "ValueList"
        p15.filter.list = ["ce", "focal"]
        p15.value = "ce"
        p16 = arcpy.Parameter(displayName="Class Weights", name="class_weights", datatype="GPString", parameterType="Optional", direction="Input")
        p16.value = ""
        p17 = arcpy.Parameter(displayName="Focal Gamma", name="focal_gamma", datatype="GPDouble", parameterType="Optional", direction="Input")
        p17.value = 2.0
        p18 = arcpy.Parameter(displayName="Ignore Index", name="ignore_index", datatype="GPLong", parameterType="Optional", direction="Input")
        p18.value = -100

        p19 = arcpy.Parameter(displayName="Use Early Stopping", name="use_early_stopping", datatype="GPBoolean", parameterType="Optional", direction="Input")
        p19.value = True
        p20 = arcpy.Parameter(displayName="Early Stopping Patience", name="early_stopping_patience", datatype="GPLong", parameterType="Optional", direction="Input")
        p20.value = 5
        p21 = arcpy.Parameter(displayName="Evaluate After Training", name="evaluate_after_training", datatype="GPBoolean", parameterType="Optional", direction="Input")
        p21.value = True
        p22 = arcpy.Parameter(displayName="Max Validation Batches", name="max_validation_batches", datatype="GPLong", parameterType="Optional", direction="Input")

        p23 = arcpy.Parameter(displayName="Extract Bands", name="extract_bands", datatype="GPString", parameterType="Optional", direction="Input")
        p23.value = "0,1,2,3"
        p24 = arcpy.Parameter(displayName="Inference Input Scale", name="input_scale", datatype="GPString", parameterType="Optional", direction="Input")
        p24.filter.type = "ValueList"
        p24.filter.list = ["auto", "0_1", "0_255", "0_65535", "max"]
        p24.value = "auto"
        p25 = arcpy.Parameter(displayName="Max Input Value", name="max_input_value", datatype="GPDouble", parameterType="Optional", direction="Input")
        p26 = arcpy.Parameter(displayName="Model Padding", name="model_padding", datatype="GPLong", parameterType="Optional", direction="Input")
        p26.value = 0
        p27 = arcpy.Parameter(displayName="Confidence Threshold", name="confidence_threshold", datatype="GPDouble", parameterType="Optional", direction="Input")
        p27.value = 0.0

        p28 = arcpy.Parameter(displayName="Run Environment Check", name="run_environment_check", datatype="GPBoolean", parameterType="Optional", direction="Input")
        p28.value = True
        p29 = arcpy.Parameter(displayName="Aggressive Repo Trim", name="aggressive_repo_trim", datatype="GPBoolean", parameterType="Optional", direction="Input")
        p29.value = True

        p30 = arcpy.Parameter(displayName="Input Channels (blank = number of Extract Bands)", name="input_channels", datatype="GPLong", parameterType="Optional", direction="Input")
        p31 = arcpy.Parameter(displayName="Input Adapter", name="input_adapter", datatype="GPString", parameterType="Optional", direction="Input")
        p31.filter.type = "ValueList"
        p31.filter.list = ["auto", "none", "learned_1x1", "learned_3x3"]
        p31.value = "auto"
        p32 = arcpy.Parameter(displayName="Training Input Scale", name="training_input_scale", datatype="GPString", parameterType="Optional", direction="Input")
        p32.filter.type = "ValueList"
        p32.filter.list = ["fastai", "0_1", "0_255", "0_65535", "max", "auto"]
        p32.value = "fastai"

        p33 = arcpy.Parameter(displayName="Execution Mode", name="execution_mode", datatype="GPString", parameterType="Optional", direction="Input")
        p33.filter.type = "ValueList"
        p33.filter.list = ["external_subprocess", "in_process"]
        p33.value = "external_subprocess"
        p34 = arcpy.Parameter(displayName="ArcGIS Pro DL Python Executable", name="arcgispro_dl_python", datatype="DEFile", parameterType="Optional", direction="Input")
        p34.filter.list = ["exe"]
        p34.value = ""

        params.extend([
            p0, p1, p2, p3, p4,
            p5, p6, p7, p8, p9, p10,
            p11, p12, p13, p14,
            p15, p16, p17, p18,
            p19, p20, p21, p22,
            p23, p24, p25, p26, p27,
            p28, p29, p30, p31, p32, p33, p34,
        ])
        _apply_parameter_help(params)
        return params

    def isLicensed(self):
        return True

    def updateParameters(self, parameters):
        # v2.16: keep the ArcGIS UI friendly for 4-band/multispectral input.
        # When users type 0,1,2,3 and leave Input Adapter at auto, the backend
        # resolves it to learned_1x1. We avoid showing a hard error before the
        # user scrolls down to the Input Adapter parameter.
        try:
            extract_bands = parameters[23].valueAsText or "0,1,2"
            bands = [int(x.strip()) for x in extract_bands.replace(";", ",").split(",") if x.strip()]
            if len(parameters) > 31:
                adapter_text = (parameters[31].valueAsText or "auto").strip().lower()
                # Leave explicitly chosen values untouched. Default auto is safe.
                if adapter_text in ("", None):
                    parameters[31].value = "auto"
                if len(bands) != 3 and adapter_text == "none" and not getattr(parameters[31], "altered", False):
                    parameters[31].value = "auto"
        except Exception:
            pass
        return

    def updateMessages(self, parameters):
        chip_size = parameters[5].value
        if chip_size and int(chip_size) % 16 != 0:
            parameters[5].setErrorMessage("Chip size must be divisible by 16 for ViT-L/16.")
        # Decoder Channels is decoder feature width, not class count or band count.
        try:
            dec_ch = int(parameters[13].value) if parameters[13].value not in (None, "") else 256
            if dec_ch < 32:
                parameters[13].setErrorMessage("Decoder Channels is the decoder feature width, not the number of classes/bands. Use 128 or 256; recommended default is 256.")
        except Exception:
            parameters[13].setErrorMessage("Decoder Channels must be an integer, recommended 256.")
        extract_bands = parameters[23].valueAsText or "0,1,2"
        input_adapter = ((parameters[31].valueAsText if len(parameters) > 31 else "auto") or "auto").strip().lower()
        input_channels_value = parameters[30].value if len(parameters) > 30 else None
        try:
            bands = [int(x.strip()) for x in extract_bands.replace(";", ",").split(",") if x.strip()]
            resolved_adapter = input_adapter
            if resolved_adapter in ("", "auto"):
                if len(parameters) > 31:
                    parameters[31].setWarningMessage(
                        "Input Adapter=auto will resolve after the DINOv3 backbone is loaded; it can use native channels or learned_1x1 as needed."
                    )
            if resolved_adapter in ("none", "identity", "rgb") and len(bands) != 3:
                parameters[23].setErrorMessage("Legacy RGB mode requires exactly 3 bands, e.g. 0,1,2. Set Input Adapter=auto, learned_1x1, or learned_3x3 for 4-band/multispectral/hyperspectral input.")
            if resolved_adapter not in ("", "auto", "none", "identity", "rgb") and input_channels_value not in (None, ""):
                if len(bands) != int(input_channels_value):
                    parameters[30].setErrorMessage("Input Channels must equal the number of Extract Bands when using a learned input adapter.")
        except Exception:
            parameters[23].setErrorMessage("Extract Bands must be comma-separated integers, e.g. 0,1,2 or 0,1,2,3.")
        class_weights = parameters[16].valueAsText or ""
        if class_weights.strip():
            try:
                [float(x.strip()) for x in class_weights.replace(";", ",").split(",") if x.strip()]
            except Exception:
                parameters[16].setErrorMessage("Class Weights must be comma-separated numbers, e.g. 0.5,2,1,3.")
        return

    def execute(self, parameters, messages):
        max_val_batches = parameters[22].value
        max_input_value = parameters[25].value
        kwargs = {
            "training_data_dir": parameters[0].valueAsText,
            "output_root": parameters[1].valueAsText,
            "repo_dir": parameters[2].valueAsText,
            "backbone_weights_path": parameters[3].valueAsText,
            "model_name": parameters[4].valueAsText or "dinov3_sat493m_seg_v2",
            "chip_size": int(parameters[5].value),
            "batch_size": int(parameters[6].value),
            "epochs": int(parameters[7].value),
            "learning_rate": float(parameters[8].value),
            "val_split_pct": float(parameters[9].value),
            "seed": int(parameters[10].value),
            "freeze_backbone": bool(parameters[11].value),
            "decoder_type": parameters[12].valueAsText or "fpn_lite",
            "decoder_channels": int(parameters[13].value),
            "dropout": float(parameters[14].value),
            "loss_mode": parameters[15].valueAsText or "ce",
            "class_weights": parameters[16].valueAsText or None,
            "focal_gamma": float(parameters[17].value),
            "ignore_index": int(parameters[18].value),
            "use_early_stopping": bool(parameters[19].value),
            "early_stopping_patience": int(parameters[20].value),
            "evaluate_after_training": bool(parameters[21].value),
            "max_validation_batches": None if max_val_batches in (None, "") else int(max_val_batches),
            "extract_bands": parameters[23].valueAsText or "0,1,2",
            "input_scale": parameters[24].valueAsText or "auto",
            "max_input_value": None if max_input_value in (None, "") else float(max_input_value),
            "model_padding": int(parameters[26].value),
            "confidence_threshold": float(parameters[27].value),
            "run_environment_check": bool(parameters[28].value),
            "aggressive_repo_trim": bool(parameters[29].value),
            "input_channels": None if len(parameters) <= 30 or parameters[30].value in (None, "") else int(parameters[30].value),
            "input_adapter": "none" if len(parameters) <= 31 else (parameters[31].valueAsText or "none"),
            "training_input_scale": "fastai" if len(parameters) <= 32 else (parameters[32].valueAsText or "fastai"),
            "dinov3_model_name": "dinov3_vitl16",
            "embed_dim": 1024,
        }

        def _tool_log(msg):
            try:
                messages.addMessage(str(msg))
            except Exception:
                try:
                    arcpy.AddMessage(str(msg))
                except Exception:
                    pass

        kwargs["log_callback"] = _tool_log

        try:
            execution_mode = (parameters[33].valueAsText if len(parameters) > 33 else "external_subprocess") or "external_subprocess"
            if execution_mode == "external_subprocess":
                python_exe = _resolve_python_executable(parameters[34].valueAsText if len(parameters) > 34 else None)
                training_items = _split_arcgis_multivalue_paths(parameters[0].valueAsText) or [parameters[0].valueAsText]
                log_path = str(Path(parameters[1].valueAsText) / f"{parameters[4].valueAsText or 'dinov3_sat493m_seg_v2'}_training_logs" / "arcgis_pyt_subprocess.log")
                cmd = [
                    python_exe, str(CURRENT_DIR / "train_dinov3_model_v2.py"),
                    "--training-data", *training_items,
                    "--output-root", parameters[1].valueAsText,
                    "--repo-dir", parameters[2].valueAsText,
                    "--backbone-weights", parameters[3].valueAsText,
                    "--model-name", parameters[4].valueAsText or "dinov3_sat493m_seg_v2",
                    "--chip-size", str(int(parameters[5].value)),
                    "--batch-size", str(int(parameters[6].value)),
                    "--epochs", str(int(parameters[7].value)),
                    "--learning-rate", str(float(parameters[8].value)),
                    "--val-split-pct", str(float(parameters[9].value)),
                    "--seed", str(int(parameters[10].value)),
                    "--decoder-type", parameters[12].valueAsText or "fpn_lite",
                    "--decoder-channels", str(int(parameters[13].value)),
                    "--dropout", str(float(parameters[14].value)),
                    "--loss-mode", parameters[15].valueAsText or "ce",
                    "--focal-gamma", str(float(parameters[17].value)),
                    "--ignore-index", str(int(parameters[18].value)),
                    "--early-stopping-patience", str(int(parameters[20].value)),
                    "--extract-bands", parameters[23].valueAsText or "0,1,2",
                    "--input-scale", parameters[24].valueAsText or "auto",
                    "--model-padding", str(int(parameters[26].value)),
                    "--confidence-threshold", str(float(parameters[27].value)),
                    "--input-adapter", (parameters[31].valueAsText if len(parameters) > 31 else "auto") or "auto",
                    "--training-input-scale", (parameters[32].valueAsText if len(parameters) > 32 else "fastai") or "fastai",
                ]
                if bool(parameters[11].value):
                    cmd.append("--freeze-backbone")
                else:
                    cmd.append("--unfreeze-backbone")
                if parameters[16].valueAsText:
                    cmd.extend(["--class-weights", parameters[16].valueAsText])
                if max_val_batches not in (None, ""):
                    cmd.extend(["--max-validation-batches", str(int(max_val_batches))])
                if max_input_value not in (None, ""):
                    cmd.extend(["--max-input-value", str(float(max_input_value))])
                if len(parameters) > 30 and parameters[30].value not in (None, ""):
                    cmd.extend(["--input-channels", str(int(parameters[30].value))])
                if not bool(parameters[19].value):
                    cmd.append("--no-early-stopping")
                if not bool(parameters[21].value):
                    cmd.append("--no-eval")
                if not bool(parameters[28].value):
                    cmd.append("--skip-environment-check")
                if not bool(parameters[29].value):
                    cmd.append("--no-aggressive-repo-trim")

                messages.addMessage("Starting DINOv3 training in external_subprocess mode. ArcGIS Pro will not import torch in the .pyt process.")
                _run_subprocess(cmd, messages, cwd=CURRENT_DIR, log_path=log_path)
                messages.addMessage(f"Subprocess log: {log_path}")
                return

            messages.addMessage("Starting DINOv3 training in in_process mode...")
            train_and_package_v2, _ = _load_core()
            outputs = train_and_package_v2(**kwargs)
            messages.addMessage(f"Deployment checkpoint: {outputs.deployment_checkpoint}")
            messages.addMessage(f"EMD written to: {outputs.emd_path}")
            messages.addMessage(f"DLPK TO USE IN ARCGIS PRO: {outputs.dlpk_path}")
            messages.addMessage(f"Manifest written to: {outputs.manifest_path}")
            if outputs.epoch_log_csv:
                messages.addMessage(f"Epoch progress CSV: {outputs.epoch_log_csv}")
            if outputs.metrics_json:
                messages.addMessage(f"Validation metrics: {outputs.metrics_json}")
            if outputs.metrics_csv:
                messages.addMessage(f"Per-class metrics: {outputs.metrics_csv}")
        except Exception as ex:
            tb = traceback.format_exc()
            arcpy.AddError(str(ex))
            arcpy.AddError(tb)
            raise


class VerifyDINOv3EnvironmentV2(object):
    def __init__(self):
        self.label = "Verify DINOv3 Environment v2.32.1-rs311-unified-hotfix"
        self.description = "Check local DINOv3 repo, weights, torch/CUDA, and a dummy forward pass."
        self.canRunInBackground = False

    def getParameterInfo(self):
        p0 = arcpy.Parameter(displayName="Local DINOv3 Repository Folder", name="repo_dir", datatype="DEFolder", parameterType="Required", direction="Input")
        p0.value = r"C:\src\dinov3"
        p1 = arcpy.Parameter(displayName="DINOv3 Backbone Weights (.pth)", name="backbone_weights", datatype="DEFile", parameterType="Required", direction="Input")
        p1.filter.list = ["pth"]
        p1.value = r"C:\weights\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
        p2 = arcpy.Parameter(displayName="Chip Size", name="chip_size", datatype="GPLong", parameterType="Optional", direction="Input")
        p2.value = 224
        p3 = arcpy.Parameter(displayName="Output Report JSON", name="out_json", datatype="DEFile", parameterType="Optional", direction="Output")
        p3.filter.list = ["json"]
        p4 = arcpy.Parameter(displayName="Execution Mode", name="execution_mode", datatype="GPString", parameterType="Optional", direction="Input")
        p4.filter.type = "ValueList"
        p4.filter.list = ["external_subprocess", "in_process"]
        p4.value = "external_subprocess"
        p5 = arcpy.Parameter(displayName="ArcGIS Pro DL Python Executable", name="arcgispro_dl_python", datatype="DEFile", parameterType="Optional", direction="Input")
        p5.filter.list = ["exe"]
        p5.value = ""
        params = [p0, p1, p2, p3, p4, p5]
        _apply_parameter_help(params)
        return params

    def isLicensed(self):
        return True

    def updateMessages(self, parameters):
        chip_size = parameters[2].value
        if chip_size and int(chip_size) % 16 != 0:
            parameters[2].setErrorMessage("Chip size must be divisible by 16 for ViT-L/16.")
        return

    def execute(self, parameters, messages):
        execution_mode = (parameters[4].valueAsText if len(parameters) > 4 else "external_subprocess") or "external_subprocess"
        if execution_mode == "external_subprocess":
            python_exe = _resolve_python_executable(parameters[5].valueAsText if len(parameters) > 5 else None)
            out_json = parameters[3].valueAsText
            if not out_json:
                out_json = str(Path(os.environ.get("TEMP", str(CURRENT_DIR))) / "dinov3_verify_report.json")
            cmd = [
                python_exe, str(CURRENT_DIR / "verify_dinov3_arcgis_env_v2.py"),
                "--repo-dir", parameters[0].valueAsText,
                "--backbone-weights", parameters[1].valueAsText,
                "--chip-size", str(int(parameters[2].value)),
                "--out-json", out_json,
            ]
            _run_subprocess(cmd, messages, cwd=CURRENT_DIR, log_path=str(Path(out_json).with_suffix(".log")))
            messages.addMessage(f"Report JSON: {out_json}")
            return

        _, verify_runtime_environment = _load_core()
        report = verify_runtime_environment(
            repo_dir=parameters[0].valueAsText,
            backbone_weights_path=parameters[1].valueAsText,
            chip_size=int(parameters[2].value),
        )
        for k, v in report.items():
            messages.addMessage(f"{k}: {v}")
        out_json = parameters[3].valueAsText
        if out_json:
            Path(out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not report.get("forward_test"):
            arcpy.AddError("DINOv3 environment verification failed. See messages/report JSON for details.")
            raise RuntimeError("DINOv3 environment verification failed.")
