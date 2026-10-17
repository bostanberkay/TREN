# PyInstaller spec for the portable Windows x64 build (a TREN folder with
# TREN.exe; one-folder, not one-file). Build with packaging/build_windows.ps1.
import os
import sys

from PyInstaller.utils.hooks import collect_data_files

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(ROOT, "packaging"))
import collect_licenses

a = Analysis(
    [os.path.join(ROOT, "src", "tren", "__main__.py")],
    # packaging/ holds tren_selftest.py, which `TREN.exe --self-test` imports.
    pathex=[os.path.join(ROOT, "src"), os.path.join(ROOT, "packaging")],
    # Bundled at tren/resources, next to the package's modules, as in the source tree.
    datas=[(os.path.join(ROOT, "src", "tren", "resources"), os.path.join("tren", "resources"))]
    + collect_data_files("stanza")
    + collect_data_files("emoji"),
    # scikit-learn is never imported by TREN's code; it is needed only to
    # unpickle tren/resources/models/*.joblib, so analysis cannot find it.
    hiddenimports=[
        "sklearn.linear_model._logistic",
        "sklearn.feature_extraction.text",
        "sklearn.feature_extraction._dict_vectorizer",
        "tren_selftest",
    ],
    excludes=["pytest", "_pytest", "PyInstaller"],
    noarchive=False,
)
# Licenses of what this build actually bundles, found from the analysis, not from requirements.txt.
LICENSES_DIR = os.path.join(workpath, "third_party_licenses")
collect_licenses.collect_from_analysis(a, LICENSES_DIR)
a.datas += collect_licenses.data_entries(LICENSES_DIR, os.path.join("licenses", "third_party"))
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
