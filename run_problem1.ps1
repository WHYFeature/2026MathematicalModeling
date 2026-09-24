param(
    [int]$Limit = 0,
    [switch]$SmokeTest,
    [switch]$Overwrite
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv_problem1\Scripts\python.exe"
$Pipeline = Join-Path $ProjectRoot "problem1_pipeline.py"
$DataRoot = Join-Path $ProjectRoot "DATA"
$OutputRoot = Join-Path $ProjectRoot "problem1_outputs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run .\setup_problem1.ps1 first."
}

$argsList = @($Pipeline, "--data-root", $DataRoot, "--output-root", $OutputRoot)
if ($Limit -gt 0) { $argsList += @("--limit", $Limit) }
if ($Overwrite) { $argsList += "--overwrite" }
if ($SmokeTest) {
    $argsList += @("--allow-fallback", "--dry-run")
} else {
    $ModelDir = Join-Path $ProjectRoot "models\bert-base-uncased"
    if (-not (Test-Path -LiteralPath $ModelDir)) {
        throw "Local BERT model not found: $ModelDir. Run the model-cache command in README_problem1.md first."
    }
    $argsList += @("--text-backend", "transformers", "--text-model", $ModelDir)
}

& $Python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

