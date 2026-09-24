param(
    [string]$Python = "python",
    [switch]$SkipMl
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectRoot ".venv_problem1"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$CoreRequirements = Join-Path $ProjectRoot "requirements_problem1.txt"

if (-not (Test-Path -LiteralPath $VenvPython)) {
    Write-Host "Creating virtual environment: $VenvDir"
    & $Python -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment with Python command: $Python" }
}

& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip" }
& $VenvPython -m pip install -r $CoreRequirements
if ($LASTEXITCODE -ne 0) { throw "Failed to install core requirements" }

if (-not $SkipMl) {
    # Match TorchAudio with PyTorch and use CUDA 12.6 wheels supported by the
    # user's NVIDIA driver. The extra index supplies ordinary PyPI dependencies.
    & $VenvPython -m pip install --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple "torch==2.11.0+cu126" "torchaudio==2.11.0+cu126"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install CUDA PyTorch/TorchAudio" }
    & $VenvPython -m pip install "transformers>=4.40"
    if ($LASTEXITCODE -ne 0) { throw "Failed to install Transformers" }
}

Write-Host "Environment ready. Python: $VenvPython"
Write-Host "For a real run, cache a local BERT model and pass --text-model with its directory."
