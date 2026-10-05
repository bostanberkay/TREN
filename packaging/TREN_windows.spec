# PyInstaller spec for the portable Windows x64 build (a TREN folder with
# TREN.exe; one-folder, not one-file). Build with packaging/build_windows.ps1.
import os

from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))

a = Analysis(
    [os.path.join(ROOT, "cs_annotator_app.py")],
    # packaging/ holds tren_selftest.py, which `TREN.exe --self-test` imports.
    pathex=[ROOT, os.path.join(ROOT, "packaging")],
    datas=[(os.path.join(ROOT, "resources"), "resources")]
    + collect_data_files("stanza")
    + collect_data_files("emoji"),
    # scikit-learn is never imported by TREN's code; it is needed only to
    # unpickle resources/models/*.joblib, so analysis cannot find it.
    hiddenimports=[
        "sklearn.linear_model._logistic",
        "sklearn.feature_extraction.text",
        "sklearn.feature_extraction._dict_vectorizer",
        "tren_selftest",
    ],
    excludes=["pytest", "_pytest", "PyInstaller"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TREN",
    console=False,
    upx=False,
    # PyInstaller converts the PNG to .ico (needs Pillow at build time only).
    icon=os.path.join(ROOT, "assets", "tren_icon.png"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="TREN",
    upx=False,
)
