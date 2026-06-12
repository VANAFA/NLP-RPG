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
