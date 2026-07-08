# One-time setup: creates a venv and installs torch (CUDA 12.1 build) + deps.
# Re-run safely; pip will skip what's already installed.
#
# peft>=0.18 requires Python 3.10+, so this uses the `py -3.10` launcher
# rather than whatever `python` resolves to on PATH.

function Invoke-Step {
    param([string]$Description, [scriptblock]$Command)
    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed: $Description (exit code $LASTEXITCODE)"
        exit 1
    }
}

Invoke-Step "Creating venv with Python 3.10..." { py -3.10 -m venv .venv }
Invoke-Step "Upgrading pip..." { .\.venv\Scripts\python.exe -m pip install --upgrade pip }
Invoke-Step "Installing torch (CUDA 12.1)..." { .\.venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu121 }
Invoke-Step "Installing remaining requirements..." { .\.venv\Scripts\pip.exe install -r requirements.txt }

Write-Host "Setup complete. Run .\run_server.ps1 to start the model server."
