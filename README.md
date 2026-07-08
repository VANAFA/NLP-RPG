# NLP-RPG

A DOS-style chat RPG UI built with Vite + React.

## Features

- Green terminal aesthetic using `#33ff00`
- Chat-based narrator input at the bottom left
- Grid world map with player pointer
- Inventory slots
- HP and XP bars plus level display
- Click-to-level stat grid with available points
- NPC window in the bottom right
- Wireframe 3D-style NPC viewer with randomized model variants
- NPC fallback logo when no NPC is active

## Run

```powershell
npm install
npm run dev
```

## Build

```powershell
npm run build
```

## Backend (Narrator LLM)

The narrator is powered by a local FastAPI server (`backend/`) that loads
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
plus the `fireball-nlp/fireball-qwen3-4b-lora-10k` LoRA adapter, and exposes
an OpenAI-compatible `/v1/chat/completions` endpoint.

1. One-time setup (creates a venv, installs torch + deps — requires Python 3.10+ and an NVIDIA GPU):

   ```powershell
   cd backend
   .\setup.ps1
   ```

2. Start the server (downloads the model on first run, ~8GB, then serves at `http://localhost:8000`):

   ```powershell
   .\run_server.ps1
   ```

3. In the running game, click **⚙ LINK** and set the Base URL to `http://localhost:8000`, then Save. The game will use it for both narrator turns and character generation.

4. (Optional) To let a deployed frontend (e.g. GitHub Pages) reach the model running on this PC, expose it with a Cloudflare Quick Tunnel in a second terminal:

   ```powershell
   .\start_tunnel.ps1
   ```

   This prints a public `https://*.trycloudflare.com` URL — use that as the Base URL in ⚙ LINK instead. The URL changes each time you restart the tunnel.

## Notes

This implementation is set up so the narrator LLM can be wired to real tools later:

- `inventory.add`
- `inventory.drop`
- `player.move`
- `player.hp.add`
- `player.hp.remove`
- `player.xp.add`
- `npc.spawn`
- `npc.release`

I recommend this stack over `ogex.app` for this use case because it is easier to control the exact UI layout, integrate LLM/tool state, and swap the NPC renderer later if you want a higher-fidelity 3D engine like `three.js` or `@react-three/fiber`.
