# PyInstaller spec for the macOS TREN.app. Build with packaging/build_macos.sh.
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
sys.path.insert(0, os.path.join(ROOT, "packaging"))
import collect_licenses
VERSION = os.environ.get("TREN_VERSION", "1.4.1")

a = Analysis(
    [os.path.join(ROOT, "src", "tren", "__main__.py")],
    pathex=[os.path.join(ROOT, "src")],
    # Bundled at tren/resources, next to the package's modules, as in the source tree.
    datas=[(os.path.join(ROOT, "src", "tren", "resources"), os.path.join("tren", "resources")),
           # License and third-party notices travel inside the app as well as on the DMG.
           (os.path.join(ROOT, "LICENSE"), "licenses"),
           (os.path.join(ROOT, "THIRD_PARTY_NOTICES.md"), "licenses")],
    # scikit-learn is never imported by TREN's code; it is needed only to
    # unpickle tren/resources/models/*.joblib, so analysis cannot find it.
    hiddenimports=[
        "sklearn.linear_model._logistic",
        "sklearn.feature_extraction.text",
        "sklearn.feature_extraction._dict_vectorizer",
    ],
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
    argv_emulation=False,
)
# BUNDLE straight from the EXE and TOCs, without a COLLECT folder, so the build
# holds one copy of the (torch-sized) app instead of two.
app = BUNDLE(
    exe,
    a.binaries,
    a.datas,
    name="TREN.app",
    icon=os.path.join(ROOT, "assets", "tren_icon.icns"),
    bundle_identifier="TREN",
    version=VERSION,
    info_plist={
        "CFBundleDisplayName": "TREN",
        "CFBundleVersion": VERSION,
        "NSHighResolutionCapable": True,
    },
)
