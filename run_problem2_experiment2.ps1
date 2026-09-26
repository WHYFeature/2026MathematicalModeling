[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\E_math\DATA',
    [string]$TextModel = 'D:\E_math\models\bert-base-uncased',
    [string]$OutputRoot = 'D:\E_math\problem2_experiment2',
    [string]$Models = 'baseline,robust_noaug,robust',
    [string]$Seeds = '42,43,44',
    [int]$Epochs = 8,
    [int]$BatchSize = 20,
    [switch]$SkipAttachment3,
    [switch]$Balanced,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment not found: $Py" }
$arguments = @(
    '-m', 'problem2.bert_missingness_experiment',
    '--data-root', $DataRoot,
    '--text-model', $TextModel,
    '--output-root', $OutputRoot,
    '--models', $Models,
    '--seeds', $Seeds,
    '--epochs', [string]$Epochs,
    '--batch-size', [string]$BatchSize
)
if ($SkipAttachment3) { $arguments += '--skip-attachment3' }
if ($Balanced) { $arguments += '--balanced' }
if ($Overwrite) { $arguments += '--overwrite' }
& $Py @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
