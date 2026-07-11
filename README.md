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
[`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
optionally combined with a LoRA adapter and/or activation steering (see
below), and exposes an OpenAI-compatible `/v1/chat/completions` endpoint.

### 1. One-time setup

Requires Python 3.10+ and an NVIDIA GPU. Creates a venv and installs torch + deps:

```powershell
cd backend
.\setup.ps1
```

### 2. Start the server

Downloads the model on first run (~8GB), then serves at `http://localhost:8000`:

```powershell
.\run_server.ps1
```

### 3. Connect the game to it

In the running game, click **⚙ LINK** and set the Base URL to `http://localhost:8000`, then Save. The game will use it for both narrator turns and character generation.

### 4. (Optional) Expose it to a deployed frontend

To let a deployed frontend (e.g. GitHub Pages) reach the model running on this PC, expose it with a Cloudflare Quick Tunnel in a second terminal:

```powershell
.\start_tunnel.ps1
```

This prints a public `https://*.trycloudflare.com` URL — use that as the Base URL in ⚙ LINK instead. The URL changes each time you restart the tunnel.

### Choosing the model configuration

Which weights load, and whether activation steering is applied, is controlled
by environment variables read at server startup — set them before running
`.\run_server.ps1` (in PowerShell: `$env:NAME = "value"`, in the same
terminal, before starting the server). Restart the server for a change to
take effect.

| Variable | Default | Description |
|---|---|---|
| `NARRATOR_MODEL_CONFIG` | `base_steered` | Which of the 4 configs below to load |
| `NARRATOR_BASE_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | Base model repo id |
| `NARRATOR_LORA_ADAPTER` | `fireball-nlp/fireball-qwen3-4b-lora-10k` | LoRA adapter repo id (used when the config includes a LoRA) |
| `NARRATOR_STEERING_ALPHA` | `8.0` | Steering strength (used when the config includes steering) |
| `NARRATOR_STEERING_LAYER_FRACTION` | `0.5` | Which decoder layer to steer, as a fraction of model depth |
| `NARRATOR_COMPRESSION_LEVEL` | `medium` | History compression level: `none` / `light` / `medium` / `aggressive` |

`NARRATOR_MODEL_CONFIG` accepts:

| key | config |
|---|---|
| `base` | Base model, unmodified |
| `base_steered` | **Base model + activation steering (default)** |
| `finetuned` | Base model + FIREBALL LoRA adapter |
| `finetuned_steered` | Finetuned model + activation steering |

The default, **Base + Activation Steering**, is the configuration recommended
by this project's paper (`experiments/paper/noob_paper_en.tex`): across the
four configs it evaluated, steering the base model reached the highest
thematic adherence (staying in the fantasy setting instead of accepting
anachronisms) of any variant, at roughly half the latency cost of steering
the FIREBALL-finetuned model instead. Activation steering itself needs no
retraining — it nudges one decoder layer's activations at generation time
using contrastive pairs in `backend/data/contrastive_pairs.json` (method:
Contrastive Activation Addition, see `backend/steering.py`).

To run the FIREBALL-finetuned model instead (e.g. to compare it yourself, or
to prioritize game-mechanic tool-calling reliability over thematic
strictness):

```powershell
$env:NARRATOR_MODEL_CONFIG = "finetuned"
.\run_server.ps1
```

Any other Qwen-family causal LM can be used as the base by overriding
`NARRATOR_BASE_MODEL` (and `NARRATOR_LORA_ADAPTER` if it has a matching LoRA);
`GET /health` reports the active configuration so you can confirm what a
running server actually loaded.

### History compression

Long-running conversations are compressed with CAVEMAN-style compression
(`backend/caveman.py`: strips function words and near-duplicate sentences,
keeps named entities/numbers/hard facts) before being sent to the model —
every message is compressed except the system prompt and the player's
current turn. `NARRATOR_COMPRESSION_LEVEL` controls how aggressively:

| level | effect |
|---|---|
| `none` | No compression |
| `light` | Removes only near-exact repeated sentences |
| `medium` | **Default.** Also strips function words, keeping the dense fact structure |
| `aggressive` | Also drops whole sentences with no hard facts left in them |

`medium` is the default because the paper's Experiment 2 found it the best
available trade-off between context savings and factual memory: ~36% of
tokens reclaimed for ~0.52 retention accuracy (IRA), versus `aggressive`'s
~70% tokens reclaimed but only ~0.34 retention (`light` currently reclaims
close to 0 tokens — see the paper's Discussion for why it still needs
recalibrating before it's a real middle ground). See
`experiments/README.md`'s Experiment 2 section for the full methodology.

Note: the current game frontend (`src/llmService.ts`) sends only a system
prompt plus the player's current line each turn — no accumulated history —
so compression is a no-op against it today. It applies automatically to any
client that does send multi-turn history through this same
`/v1/chat/completions` endpoint.

## Notes

The narrator responds with a schema-constrained JSON object (see `src/llmService.ts`) that can call a few tools directly:

- `addItem`
- `dropItem`
- `setObjective`
- `remember`

XP, HP, and location changes aren't tool calls — they're inferred after the fact by separate analyzer calls that read the narrator's own story text, since the narrator doesn't reliably remember to call a tool every time something happens in the narration.
