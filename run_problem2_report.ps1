param(
    [string]$InputRoot = "",
    [string]$OutputRoot = "",
    [int]$Dpi = 300
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv_problem1\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "problem2_report.py"
if (-not $InputRoot) { $InputRoot = Join-Path $ProjectRoot "problem2_outputs_bert_av_reproduce" }
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found: $Python" }
& $Python -c "import numpy, matplotlib" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "Missing report dependencies. Run: & '$Python' -m pip install 'numpy>=1.26' 'matplotlib>=3.8'"
}
$ReportArgs = @($Script, "--input-root", $InputRoot, "--dpi", "$Dpi")
if ($OutputRoot) { $ReportArgs += @("--output-root", $OutputRoot) }
& $Python @ReportArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
