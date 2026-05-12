from __future__ import annotations
import os, sys
from pathlib import Path


def add_paths():
    py = Path(sys.executable)
    env_dir = py.parent
    pro_root = Path(os.environ.get("ARCGIS_PRO_ROOT", r"C:\Program Files\ArcGIS\Pro"))
    candidates = [
        env_dir,
        env_dir / "Scripts",
        env_dir / "Library" / "bin",
        pro_root / "bin",
        pro_root / "Resources" / "ArcPy",
        pro_root / "bin" / "Python" / "Scripts",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Library" / "bin",
        pro_root / "bin" / "Python" / "envs" / "arcgispro-py3" / "Scripts",
    ]
    added=[]
    for c in candidates:
        try:
            if c.exists():
                os.add_dll_directory(str(c))
                added.append(str(c))
        except Exception:
            pass
    os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])
    return added

print("[PRE-FLIGHT QUIET] Python:", sys.executable, flush=True)
print("[PRE-FLIGHT QUIET] Version:", sys.version, flush=True)
print("[PRE-FLIGHT QUIET] cwd:", os.getcwd(), flush=True)
print("[PRE-FLIGHT QUIET] PATH entries prepared:", flush=True)
for item in add_paths():
    print("  " + item, flush=True)
print("[PRE-FLIGHT QUIET] Importing arcpy...", flush=True)
import arcpy
print("[PRE-FLIGHT QUIET] arcpy OK", arcpy.GetInstallInfo().get("Version"), flush=True)
print("[PRE-FLIGHT QUIET] Importing arcgis...", flush=True)
import arcgis
print("[PRE-FLIGHT QUIET] arcgis OK", getattr(arcgis, "__version__", "unknown"), flush=True)
print("[PRE-FLIGHT QUIET] Importing arcgis.learn...", flush=True)
from arcgis.learn import prepare_data, ModelExtension
print("[PRE-FLIGHT QUIET] arcgis.learn OK", flush=True)
