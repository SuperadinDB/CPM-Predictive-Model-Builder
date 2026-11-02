# PyInstaller onedir app, served locally by Streamlit.
from pathlib import Path
from PyInstaller.utils.hooks import collect_all
from importlib.util import find_spec

datas = [("app.py", ".")]
sample_root = Path(SPECPATH) / "samples"
if sample_root.is_dir():
    for sample_file in sample_root.rglob("*"):
        if sample_file.is_file():
            relative_folder = sample_file.parent.relative_to(sample_root)
            destination = Path("samples") / relative_folder
            datas.append((str(sample_file), str(destination)))
binaries = []
hiddenimports = [
    "streamlit", "streamlit.web.cli", "sklearn", "openpyxl",
    "cpm", "cpm.data", "cpm.i18n", "cpm.models", "cpm.ui",
    "cpm.forecasting", "cpm.notebook_export", "cpm.trace",
]
for package in ["streamlit", "sklearn", "openpyxl", "catboost", "xgboost", "lightgbm"]:
    if find_spec(package) is None:
        continue
    package_datas, package_bins, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_bins
    hiddenimports += package_hidden

a = Analysis(
    ["launch.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="CPM_Model_Builder", debug=False,
    bootloader_ignore_signals=False, strip=False, upx=True,
    console=False, disable_windowed_traceback=False,
)
COLLECT(exe, a.binaries, a.datas, strip=False, upx=True,
        upx_exclude=[], name="CPM_Model_Builder")
