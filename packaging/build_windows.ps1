# Build the portable Windows x64 TREN folder and TREN_v<version>_windows_x64.zip
# from the current source tree.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1 [-Version 1.4.0] [-OutDir dist]
#
# Uses the `python` on PATH, which must already have requirements.txt and
# packaging/requirements-build.txt installed (the GitHub Actions workflow
# .github/workflows/windows-package.yml does exactly this). Intermediate files
# go to a temporary directory that is removed afterwards. The result is not
# code-signed.
param(
    [string]$Version = "1.4.0",
    [string]$OutDir = ""
)
$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $OutDir) { $OutDir = Join-Path $Root "dist" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$OutDir = (Resolve-Path $OutDir).Path
$ZipName = "TREN_v${Version}_windows_x64.zip"
$Zip = Join-Path $OutDir $ZipName

$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("tren-build-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Tmp | Out-Null
try {
    python -m PyInstaller --noconfirm --clean `
        --workpath (Join-Path $Tmp "work") --distpath (Join-Path $Tmp "dist") `
        (Join-Path $Root "packaging\TREN_windows.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $AppDir = Join-Path $Tmp "dist\TREN"
    if (-not (Test-Path (Join-Path $AppDir "TREN.exe"))) { throw "TREN.exe was not produced" }
    Copy-Item (Join-Path $Root "LICENSE") (Join-Path $AppDir "LICENSE.txt")
    Set-Content -Path (Join-Path $AppDir "VERSION.txt") -Value "TREN $Version (Windows x64, unsigned portable build)" -Encoding utf8

    if (Test-Path $Zip) { Remove-Item $Zip }
    # .NET's ZIP API: Unicode-safe paths and no Compress-Archive size limit.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [System.IO.Compression.ZipFile]::CreateFromDirectory(
        $AppDir, $Zip, [System.IO.Compression.CompressionLevel]::Optimal, $true)
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}

Write-Host "Built $Zip"
