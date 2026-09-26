[CmdletBinding()]
param(
    [string]$DataRoot = 'D:\E_math\DATA',
    [string]$TextModel = 'D:\E_math\models\bert-base-uncased',
    [string]$OutputRoot = 'D:\E_math\problem2_missingness_experiment',
    [ValidateSet('aligned','unaligned')]
    [string]$FeatureVersion = 'aligned',
    [string]$Seeds = '42,43,44',
    [int]$Epochs = 20,
    [int]$BatchSize = 64,
    [switch]$SkipAttachment3,
    [switch]$Overwrite
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) {
    throw "Python environment not found: $Py"
}
$arguments = @(
    '-m', 'problem2.missingness_experiment',
    '--data-root', $DataRoot,
    '--text-model', $TextModel,
    '--output-root', $OutputRoot,
    '--feature-version', $FeatureVersion,
    '--seeds', $Seeds,
    '--epochs', [string]$Epochs,
    '--batch-size', [string]$BatchSize
)
if ($SkipAttachment3) { $arguments += '--skip-attachment3' }
if ($Overwrite) { $arguments += '--overwrite' }
& $Py @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
