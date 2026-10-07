# PyInstaller spec for the macOS TREN.app. Build with packaging/build_macos.sh.
import os

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
VERSION = os.environ.get("TREN_VERSION", "1.4.1")

a = Analysis(
    [os.path.join(ROOT, "src", "tren", "__main__.py")],
    pathex=[os.path.join(ROOT, "src")],
    # Bundled at tren/resources, next to the package's modules, as in the source tree.
    datas=[(os.path.join(ROOT, "src", "tren", "resources"), os.path.join("tren", "resources"))],
    # scikit-learn is never imported by TREN's code; it is needed only to
    # unpickle tren/resources/models/*.joblib, so analysis cannot find it.
    hiddenimports=[
        "sklearn.linear_model._logistic",
        "sklearn.feature_extraction.text",
        "sklearn.feature_extraction._dict_vectorizer",
    ],
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
