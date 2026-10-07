# Verify a built TREN Windows ZIP independently of the source tree.
#
#   pwsh -File packaging\verify_windows_package.ps1 -Zip <zip> -WorkRoot <dir> [-SourceReport <dir>]
#
# Extracts the ZIP into a directory whose path has spaces and Turkish
# characters, strips Python from PATH, clears PYTHONPATH/PYTHONHOME, points
# USERPROFILE and the Stanza model cache at Turkish-named directories, and then:
#   1. runs `TREN.exe --self-test` online with an empty Stanza cache (first NER
#      use downloads the models through the GUI's download notice);
#   2. runs it again with the network blocked by a dead proxy and
#      --expect-ner-cached (NER must start from the cache alone);
#   2b. runs `TREN.exe --self-test DIR --tdk-https`: real lookups against
#      sozluk.gov.tr with certificate verification on (needs the network);
#   3. compares the pipeline output and every export with -SourceReport (the
#      same self-test run from source), byte for byte;
#   4. launches TREN.exe normally, checks that its window appears and stays up,
#      that a second launch exits (single instance), and that the log has no
#      traceback.
# Hiding the runner's own Python installation is done by the workflow around
# this script. Exit code 0 means every step passed.
#
# This file is kept ASCII-only (Windows PowerShell 5.1 would misread UTF-8
# without a BOM); Turkish characters are built from code points.
param(
    [Parameter(Mandatory = $true)][string]$Zip,
    [Parameter(Mandatory = $true)][string]$WorkRoot,
    [string]$SourceReport = "",
    [int]$TimeoutSeconds = 1500
)
$ErrorActionPreference = "Stop"

function TR([int[]]$codes) { -join ($codes | ForEach-Object { [char]$_ }) }
$C_CED = TR 0x00C7   # C-cedilla
$G_BRV = TR 0x011E   # G-breve
$S_CED = TR 0x015E   # S-cedilla
$i_DLS = TR 0x0131   # dotless i
$o_UML = TR 0x00F6   # o-umlaut
$u_UML = TR 0x00FC   # u-umlaut
$s_CED = TR 0x015F   # s-cedilla
$g_BRV = TR 0x011F   # g-breve

$Failures = New-Object System.Collections.Generic.List[string]
function Fail([string]$msg) { Write-Host "FAIL: $msg"; $Failures.Add($msg) }
function Pass([string]$msg) { Write-Host "PASS: $msg" }

New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null
$WorkRoot = (Resolve-Path $WorkRoot).Path
$Zip = (Resolve-Path $Zip).Path

$InstallDir = Join-Path $WorkRoot "TREN Paket Testi $C_CED$G_BRV$S_CED"
$UserHome = Join-Path $WorkRoot "Kullan${i_DLS}c${i_DLS}lar\${S_CED}${u_UML}kr${u_UML} ${C_CED}a${g_BRV}lar"
$StanzaDir = Join-Path $WorkRoot "stanza ${o_UML}nbellek $S_CED"
$OtherCwd = Join-Path $WorkRoot "ba${s_CED}ka ${C_CED}al${i_DLS}${s_CED}ma dizini"
$Report1 = Join-Path $WorkRoot "rapor 1 $C_CED"
$Report2 = Join-Path $WorkRoot "rapor 2 $C_CED"
$ReportTdk = Join-Path $WorkRoot "rapor tdk $C_CED"
foreach ($d in @($InstallDir, $UserHome, $StanzaDir, $OtherCwd)) {
    New-Item -ItemType Directory -Force -Path $d | Out-Null
}

# .NET's ZIP API handles non-ASCII paths; tar.exe goes through the ANSI code page.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::ExtractToDirectory($Zip, $InstallDir)
$Exe = Join-Path $InstallDir "TREN\TREN.exe"
if (-not (Test-Path $Exe)) { throw "TREN.exe not found after extracting $Zip" }
$SizeMB = [math]::Round(((Get-ChildItem -Recurse -File (Join-Path $InstallDir "TREN") | Measure-Object Length -Sum).Sum) / 1MB)
Write-Host "Extracted to $InstallDir ($SizeMB MB unpacked)"

foreach ($f in @("tren\resources\lid.176.ftz", "tren\resources\frequent_tr_words.txt", "tren\resources\frequent_en_words.txt",
                 "tren\resources\models\model.joblib", "tren\resources\models\vectorizer.joblib", "tren\resources\models\metadata.json")) {
    if (Test-Path (Join-Path $InstallDir "TREN\_internal\$f")) { Pass "bundled $f" } else { Fail "missing bundled $f" }
}
$Leaked = Get-ChildItem -Recurse -Directory (Join-Path $InstallDir "TREN\_internal") |
    Where-Object { $_.Name -in @("pytest", "_pytest", "PyInstaller") }
if ($Leaked) { Fail "development packages bundled: $($Leaked.FullName -join ', ')" } else { Pass "no pytest/PyInstaller in the bundle" }

# Isolate the packaged app from any Python on this machine.
$env:PATH = "$env:SystemRoot\System32;$env:SystemRoot;$env:SystemRoot\System32\WindowsPowerShell\v1.0"
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
Remove-Item Env:PYTHONHOME -ErrorAction SilentlyContinue
$env:USERPROFILE = $UserHome
$env:STANZA_RESOURCES_DIR = $StanzaDir
if (Get-Command python -ErrorAction SilentlyContinue) { Fail "python is still reachable on PATH" } else { Pass "no python on PATH" }

function Invoke-Tren([string[]]$TrenArgs, [int]$Timeout) {
    $argLine = ($TrenArgs | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    $p = Start-Process -FilePath $Exe -ArgumentList $argLine -WorkingDirectory $OtherCwd -PassThru
    $null = $p.Handle  # keep the handle so ExitCode is readable after exit
    if (-not $p.WaitForExit($Timeout * 1000)) {
        Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
        return $null
    }
    return $p.ExitCode
}

function Read-Report([string]$dir) {
    $path = Join-Path $dir "selftest_report.json"
    if (-not (Test-Path $path)) { return $null }
    return Get-Content -Raw -Encoding utf8 $path | ConvertFrom-Json
}

function Show-Report([string]$dir) {
    $txt = Join-Path $dir "selftest_report.txt"
    if (Test-Path $txt) { Get-Content -Encoding utf8 $txt | ForEach-Object { Write-Host "    $_" } }
}

# 1. Online, empty Stanza cache.
Write-Host "== Self-test 1: online, empty NER model cache"
$code = Invoke-Tren @("--self-test", $Report1) $TimeoutSeconds
Show-Report $Report1
$r1 = Read-Report $Report1
if ($null -eq $code) { Fail "self-test 1 timed out" }
elseif ($code -ne 0 -or $null -eq $r1 -or -not $r1.ok) { Fail "self-test 1 failed (exit $code)" }
else { Pass "self-test 1" }
if ($r1) {
    if ($r1.info.ner_models_cached_before_run -eq $false -and $r1.info.ner_download_notice_shown -eq $true) {
        Pass "first NER use showed the download notice"
    } else {
        Fail "expected the NER download notice on an empty cache (cached_before=$($r1.info.ner_models_cached_before_run), shown=$($r1.info.ner_download_notice_shown))"
    }
}

# 2. Offline: dead proxy for every HTTP(S) request, models must come from the cache.
Write-Host "== Self-test 2: network blocked, NER from cache"
$env:HTTP_PROXY = "http://127.0.0.1:9"
$env:HTTPS_PROXY = "http://127.0.0.1:9"
$env:NO_PROXY = ""
$code = Invoke-Tren @("--self-test", $Report2, "--expect-ner-cached") $TimeoutSeconds
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:NO_PROXY -ErrorAction SilentlyContinue
Show-Report $Report2
$r2 = Read-Report $Report2
if ($null -eq $code) { Fail "self-test 2 (offline) timed out" }
elseif ($code -ne 0 -or $null -eq $r2 -or -not $r2.ok) { Fail "self-test 2 (offline) failed (exit $code)" }
else { Pass "self-test 2 (offline, cached NER models)" }

# 2b. Real TDK HTTPS lookups (network required; a failure here is never covered by the mock check).
Write-Host "== TDK HTTPS: real sozluk.gov.tr, verified TLS"
$code = Invoke-Tren @("--self-test", $ReportTdk, "--tdk-https") 300
$tdkTxt = Join-Path $ReportTdk "tdk_https_report.txt"
if (Test-Path $tdkTxt) { Get-Content -Encoding utf8 $tdkTxt | ForEach-Object { Write-Host "    $_" } }
$tdkJson = Join-Path $ReportTdk "tdk_https_report.json"
$rt = if (Test-Path $tdkJson) { Get-Content -Raw -Encoding utf8 $tdkJson | ConvertFrom-Json } else { $null }
if ($null -eq $code) { Fail "TDK HTTPS check timed out" }
elseif ($code -ne 0 -or $null -eq $rt -or -not $rt.ok) { Fail "TDK HTTPS check failed (exit $code)" }
else { Pass "TDK HTTPS check ($($rt.info.diagnosis))" }

# 3. Packaged output must equal the source run's output byte for byte.
if ($SourceReport) {
    $exportDirName = "d${i_DLS}${s_CED}a aktar${i_DLS}m klas${o_UML}r${u_UML}"
    $names = New-Object System.Collections.Generic.List[string]
    $names.Add("pipeline_output.txt")
    $srcExports = Join-Path $SourceReport $exportDirName
    if (Test-Path $srcExports) {
        Get-ChildItem -File $srcExports | ForEach-Object { $names.Add("$exportDirName\$($_.Name)") }
    } else {
        Fail "source report has no export directory $srcExports"
    }
    foreach ($name in $names) {
        $a = Join-Path $SourceReport $name
        $b = Join-Path $Report1 $name
        if (-not (Test-Path $a) -or -not (Test-Path $b)) { Fail "missing $name for comparison"; continue }
        if ((Get-FileHash $a).Hash -eq (Get-FileHash $b).Hash) { Pass "identical to source: $name" }
        else { Fail "differs from source run: $name" }
    }
}

# 4. Normal launch: window appears and stays, second launch exits, no traceback.
Write-Host "== Normal launch"
$Log = Join-Path $UserHome ".cs_annotator\tren.log"
$logStart = 0
if (Test-Path $Log) { $logStart = (Get-Item $Log).Length }
$p = Start-Process -FilePath $Exe -WorkingDirectory $OtherCwd -PassThru
$deadline = (Get-Date).AddSeconds(180)
$title = ""
while ((Get-Date) -lt $deadline -and -not $p.HasExited) {
    Start-Sleep -Seconds 2
    $p.Refresh()
    if ($p.MainWindowHandle -ne 0 -and $p.MainWindowTitle) { $title = $p.MainWindowTitle; break }
}
if ($title) { Pass "main window appeared: '$title'" } else { Fail "no main window within 180 s (exited=$($p.HasExited))" }
Start-Sleep -Seconds 10
$p.Refresh()
if ($p.HasExited) { Fail "TREN.exe exited on its own after starting (exit $($p.ExitCode))" } else { Pass "TREN.exe still running after 10 s" }

$second = Start-Process -FilePath $Exe -WorkingDirectory $OtherCwd -PassThru
$null = $second.Handle
if ($second.WaitForExit(120000)) { Pass "second launch exited (single instance), exit $($second.ExitCode)" }
else { Stop-Process -Id $second.Id -Force; Fail "second launch kept running alongside the first" }
Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

if (Test-Path $Log) {
    $bytes = [System.IO.File]::ReadAllBytes($Log)
    $logText = [System.Text.Encoding]::UTF8.GetString($bytes, [int]$logStart, $bytes.Length - [int]$logStart)
    if ($logText -match "Traceback") { Fail "traceback in tren.log during normal launch"; Write-Host $logText }
    else { Pass "no traceback in tren.log during normal launch" }
} else {
    Write-Host "note: no tren.log written (nothing was printed)"
}

Write-Host ""
if ($Failures.Count -gt 0) {
    Write-Host "$($Failures.Count) check(s) failed:"
    $Failures | ForEach-Object { Write-Host " - $_" }
    exit 1
}
Write-Host "All package checks passed."
exit 0
