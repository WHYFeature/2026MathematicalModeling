[CmdletBinding()]
param(
    [string]$OutputRoot = 'D:\E_math\problem3_outputs'
)

$ErrorActionPreference = 'Stop'
Set-Location 'D:\E_math'
$Py = 'D:\E_math\.venv_problem1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Py)) { throw "Python environment not found: $Py" }
& $Py -m problem3.analyze_existing --output-root $OutputRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
