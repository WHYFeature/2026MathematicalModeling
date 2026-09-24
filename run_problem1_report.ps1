param(
    [string]$ExampleId = "",
    [switch]$NoMedia
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv_problem1\Scripts\python.exe"
$Script = Join-Path $ProjectRoot "problem1_report.py"
$Visualizer = Join-Path $ProjectRoot "problem1_visualize.py"
$DataRoot = Join-Path $ProjectRoot "DATA"
$OutputRoot = Join-Path $ProjectRoot "problem1_outputs"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual environment not found. Run .\setup_problem1.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $OutputRoot "problem1_aligned_features.pkl"))) {
    throw "Feature file not found: $OutputRoot\problem1_aligned_features.pkl. Run .\run_problem1.ps1 -Overwrite first."
}

# The report figures use Matplotlib. Keep installation explicit and manual.
& $Python -c "import matplotlib" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "matplotlib is missing. Run: & $Python -m pip install 'matplotlib>=3.8'"
}

$argsList = @($Script, "--data-root", $DataRoot, "--output-root", $OutputRoot)
if ($ExampleId) { $argsList += @("--example-id", $ExampleId) }
if ($NoMedia) { $argsList += "--no-media" }
& $Python @argsList
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Refresh the four aggregate diagnostic figures from the same current PKL.
& $Python $Visualizer --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
