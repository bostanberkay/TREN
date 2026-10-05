#!/usr/bin/env bash
# Build TREN.app and TREN_v<version>.dmg on macOS from the current source tree.
#
#   packaging/build_macos.sh [version] [output_dir]
#
# Runs PyInstaller from the repository's .venv (requirements.txt plus
# pyinstaller). Intermediate files go to a temporary directory that is removed
# afterwards; only the DMG is left in output_dir (default: dist/). The app is
# ad-hoc signed by PyInstaller, not Developer ID signed or notarized, and is
# built for the architecture of the Python in .venv.
set -euo pipefail

VERSION="${1:-1.4.0}"
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
hdiutil create -volname "TREN" -srcfolder "$TMP/stage" -ov -format UDZO "$DMG"

echo "Built $DMG"
