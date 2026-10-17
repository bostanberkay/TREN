#!/usr/bin/env bash
# Build TREN.app and TREN_v<version>.dmg on macOS from the current source tree.
#
#   packaging/build_macos.sh [version] [output_dir]
#
# Runs PyInstaller from the repository's .venv (requirements.txt plus
# pyinstaller). The app bundles whatever the analysis finds there, so packages
# installed in .venv beyond those (for example pytest or transformers) end up in
# it and in licenses/third_party/; build from a clean .venv to avoid that.
# Intermediate files go to a temporary directory that is removed
# afterwards; only the DMG is left in output_dir (default: dist/). The app is
# ad-hoc signed by PyInstaller, not Developer ID signed or notarized, and is
# built for the architecture of the Python in .venv. The DMG also carries
# LICENSE.txt and THIRD_PARTY_NOTICES.md; TREN.spec bundles copies in the app's
# `licenses` folder, together with the license files of the packages the build
# bundles (licenses/third_party/, collected by packaging/collect_licenses.py).
set -euo pipefail

VERSION="${1:-1.4.1}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${2:-$ROOT/dist}"
PYTHON="$ROOT/.venv/bin/python"
DMG="$OUT_DIR/TREN_v$VERSION.dmg"

TMP="$(mktemp -d "${TMPDIR:-/tmp}/tren-build.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

TREN_VERSION="$VERSION" "$PYTHON" -m PyInstaller --noconfirm --clean \
    --workpath "$TMP/work" --distpath "$TMP/dist" "$ROOT/packaging/TREN.spec"
rm -rf "$TMP/work"

# Move (not copy) the app into the DMG staging folder to keep one copy on disk.
mkdir -p "$TMP/stage" "$OUT_DIR"
mv "$TMP/dist/TREN.app" "$TMP/stage/TREN.app"
ln -s /Applications "$TMP/stage/Applications"
cp "$ROOT/LICENSE" "$TMP/stage/LICENSE.txt"
cp "$ROOT/THIRD_PARTY_NOTICES.md" "$TMP/stage/THIRD_PARTY_NOTICES.md"
hdiutil create -volname "TREN" -srcfolder "$TMP/stage" -ov -format UDZO "$DMG"

echo "Built $DMG"
