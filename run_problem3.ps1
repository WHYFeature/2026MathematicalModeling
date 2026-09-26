[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\E_math\DATA',
    [string]$TextModel = 'D:\E_math\models\bert-base-uncased',
    [string]$OutputRoot = 'D:\E_math\problem3_outputs',
    [int]$Epochs = 20,
    [int]$BatchSize = 64,
    [int]$Seed = 42,
    [switch]$SkipAttachment4,
    [switch]$SkipFigures,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment not found: $Py" }
$arguments = @(
    '-m', 'problem3.train',
    '--data-root', $DataRoot,
    '--text-model', $TextModel,
    '--output-root', $OutputRoot,
    '--epochs', [string]$Epochs,
    '--batch-size', [string]$BatchSize,
    '--seed', [string]$Seed
)
if ($SkipAttachment4) { $arguments += '--skip-attachment4' }
if ($SkipFigures) { $arguments += '--skip-figures' }
if ($Overwrite) { $arguments += '--overwrite' }
& $Py @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
