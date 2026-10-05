# Install TREN's Python dependencies on Windows. The Windows CI job and the
# Windows package workflow both use this script, so they install the same way.
#
#   powershell -ExecutionPolicy Bypass -File packaging\install_windows_deps.ps1 [-Dev] [-Build]
#
# fasttext 0.9.3 has no Windows wheel, and its PyPI source needs a two-line fix
# to compile with MSVC on Python 3.10+ (see packaging/build_fasttext_windows.py).
# That script builds a wheel from the hash-checked sdist; it is installed first,
# so requirements.txt then finds fasttext==0.9.3 already installed instead of
# building the unpatched source. Needs the Microsoft C++ Build Tools.
#
# -Dev also installs requirements-dev.txt (tests); -Build also installs
# packaging/requirements-build.txt (PyInstaller). Stops at the first failing
# command, and finishes by loading resources/lid.176.ftz and checking real
# predictions (packaging/check_fasttext.py).
param(
    [switch]$Dev,
    [switch]$Build
)
$ErrorActionPreference = "Stop"

function Invoke-Step([string]$Name, [scriptblock]$Command) {
    Write-Host "==> $Name"
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "$Name failed with exit code $LASTEXITCODE" }
}

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WheelDir = Join-Path ([System.IO.Path]::GetTempPath()) ("tren-fasttext-" + [guid]::NewGuid().ToString("N"))

try {
    Invoke-Step "Upgrade pip" { python -m pip install --upgrade pip }
    Invoke-Step "Build fasttext 0.9.3 wheel (patched for MSVC)" {
        python (Join-Path $Root "packaging\build_fasttext_windows.py") --wheel-dir $WheelDir
    }
    $Wheel = Get-ChildItem -Path $WheelDir -Filter "fasttext-0.9.3-*.whl" | Select-Object -First 1
    if (-not $Wheel) { throw "no fasttext-0.9.3 wheel in $WheelDir" }
    Invoke-Step "Install $($Wheel.Name)" { python -m pip install --no-deps $Wheel.FullName }
    Invoke-Step "Install requirements.txt" { python -m pip install -r (Join-Path $Root "requirements.txt") }
    if ($Dev) {
        Invoke-Step "Install requirements-dev.txt" { python -m pip install -r (Join-Path $Root "requirements-dev.txt") }
    }
    if ($Build) {
        Invoke-Step "Install packaging/requirements-build.txt" {
            python -m pip install -r (Join-Path $Root "packaging\requirements-build.txt")
        }
    }
    Invoke-Step "Check fasttext with resources/lid.176.ftz" { python (Join-Path $Root "packaging\check_fasttext.py") }
} finally {
    Remove-Item -Recurse -Force $WheelDir -ErrorAction SilentlyContinue
}
