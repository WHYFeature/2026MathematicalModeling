param(
    [string]$TargetRoot = "",
    [ValidateRange(1, 10000)]
    [int]$Epochs = 30
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv_problem1\Scripts\python.exe"
if (-not $TargetRoot) { $TargetRoot = Join-Path $ProjectRoot "problem2_outputs_bert_av_reproduce" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
& $Python (Join-Path $ProjectRoot "problem2_recover_history.py") --target-root $TargetRoot --epochs $Epochs
if ($LASTEXITCODE -ne 0) { throw "Replay verification or plotting did not complete. Original model results remain intact; see the preceding error and history_recovery logs." }
& (Join-Path $ProjectRoot "run_problem2_report.ps1") -InputRoot $TargetRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
