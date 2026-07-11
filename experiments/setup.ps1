# Installs this harness's incremental dependencies (anthropic SDK, pandas,
# tqdm, python-dotenv) into the EXISTING backend/.venv, which already has
# torch/transformers/peft/accelerate installed — avoids downloading a second
# multi-GB CUDA environment just for these experiments.
#
# Run backend/setup.ps1 first if backend/.venv doesn't exist yet.

$ErrorActionPreference = "Stop"

$venvPython = Join-Path $PSScriptRoot "..\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "backend/.venv not found. Run backend/setup.ps1 first (it installs torch/transformers/peft, which this harness also needs)."
    exit 1
}

& (Join-Path $PSScriptRoot "..\backend\.venv\Scripts\pip.exe") install -r (Join-Path $PSScriptRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Error "pip install failed (exit code $LASTEXITCODE)"
    exit 1
}

Write-Host "Setup complete."
Write-Host "Copy experiments/.env.example to experiments/.env and fill in ANTHROPIC_API_KEY, then:"
Write-Host "  cd experiments"
Write-Host "  ..\backend\.venv\Scripts\python.exe run_experiment1.py"
