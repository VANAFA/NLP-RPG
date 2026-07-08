# Starts the narrator LLM API on http://localhost:8000
# First run: .\setup.ps1
#
# If activation fails with "running scripts is disabled on this system",
# run this once in an elevated PowerShell: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

. .\.venv\Scripts\Activate.ps1
uvicorn server:app --host 0.0.0.0 --port 8000
