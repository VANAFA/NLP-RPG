# Exposes the local model server (http://localhost:8000) to the public
# internet via a Cloudflare Quick Tunnel, so a deployed web app (Vercel,
# Netlify, etc.) can reach the model running on this PC.
#
# Run this in a SEPARATE terminal from run_server.ps1 (the API must already
# be running on port 8000).
#
# The command prints a random URL like:
#   https://random-words-1234.trycloudflare.com
# Put that URL (no trailing slash) into VITE_API_URL in the frontend's .env
# file. Quick Tunnels are ephemeral: the URL changes every time you restart
# this script, so update .env again after each restart.

if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "cloudflared not found. Installing via winget..."
    winget install --id Cloudflare.cloudflared -e
    Write-Host "Installed. You may need to restart this terminal for PATH changes to take effect."
}

cloudflared tunnel --url http://localhost:8000
