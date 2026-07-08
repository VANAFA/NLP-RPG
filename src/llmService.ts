export interface InventoryItemInput {
  name: string;
  quantity: number;
  kind: string;
  desc?: string;
}

export interface LLMContext {
  hp: number;
  hpMax: number;
  xp: number;
  xpMax: number;
  level: number;
  stats: Record<string, number>;
  location: { x: number; y: number; label: string };
  inventory: InventoryItemInput[];
  objective: string;
  worldNotes: string[];
  worldMapSummary: string;
}

// Claves de localStorage donde la app guarda la URL/API key del servidor LLM.
// Exportadas para que App.tsx las reuse al leer/escribir el panel de ajustes.
export const LS_BASE_URL_KEY = 'nlprpg.llm.baseUrl';
export const LS_API_KEY_KEY = 'nlprpg.llm.apiKey';

// Debe coincidir con el alias usado en `--lora-modules <alias>=...` / el backend local.
const MODEL_NAME = 'fireball-qwen3-4b-lora-10k';

// 'move' and 'spawnNpc' are intentionally not narrator tools — the narrator
// often narrates movement/NPCs without remembering to call the matching
// tool, so those are decided by dedicated analyzer calls instead (see
// analyzeNarratorTurn) which can't skip a required schema field the way an
// optional tool call can be skipped.
const TOOL_NAMES = ['addItem', 'dropItem', 'setObjective', 'remember'];

// A fully generic `args: { type: 'object' }` (no declared properties) makes
// lm-format-enforcer crash as soon as the model writes a real key into it
// (AttributeError: 'bool' object has no attribute 'anyOf' — its parser falls
// back to `additionalProperties`, which Pydantic/plain-JS emits as a bare
// boolean rather than a schema it knows how to walk). Giving every tool's
// args an explicit shape avoids that fallback path entirely.
const TOOL_ARGS_SCHEMAS = {
  addItem: {
    type: 'object',
    properties: {
      name: { type: 'string', minLength: 1, maxLength: 40 },
      quantity: { type: 'integer', minimum: 1 },
      kind: { type: 'string', minLength: 1, maxLength: 20 },
      desc: { type: 'string', minLength: 1, maxLength: 90 },
    },
    required: ['name', 'quantity', 'kind', 'desc'],
    additionalProperties: false,
  },
  dropItem: {
    type: 'object',
    properties: { index: { type: 'integer', minimum: 0 } },
    required: ['index'],
    additionalProperties: false,
  },
  setObjective: {
    type: 'object',
    properties: { objective: { type: 'string', minLength: 1, maxLength: 150 } },
    required: ['objective'],
    additionalProperties: false,
  },
  remember: {
    type: 'object',
    properties: { note: { type: 'string', minLength: 1, maxLength: 150 } },
    required: ['note'],
    additionalProperties: false,
  },
};

// Schema-constrained decoding on this model/vocab is slow per token (see
// backend/server.py), so string lengths are capped here to bound worst-case
// latency deterministically instead of gambling on max_tokens alone.
const TURN_RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    thinking: { type: 'string', maxLength: 120 },
    story: { type: 'string', minLength: 10, maxLength: 320 },
    toolCalls: {
      type: 'array',
      maxItems: 3,
      items: {
        type: 'object',
        properties: {
          name: { type: 'string', enum: TOOL_NAMES },
          args: { anyOf: Object.values(TOOL_ARGS_SCHEMAS) },
        },
        required: ['name', 'args'],
      },
    },
  },
  required: ['thinking', 'story', 'toolCalls'],
};

async function callLlmOnce(systemPrompt: string, userText: string, schema: object, temperature: number, maxTokens: number) {
  const baseUrl = localStorage.getItem(LS_BASE_URL_KEY);
  if (!baseUrl) {
    throw new Error('LLM_NOT_CONFIGURED');
  }

  const apiKey = localStorage.getItem(LS_API_KEY_KEY);

  const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      // Avoids the HTML warning page ngrok's free tier shows to browser navigations.
      'ngrok-skip-browser-warning': 'true',
      ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    },
    body: JSON.stringify({
      model: MODEL_NAME,
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userText },
      ],
      temperature,
      max_tokens: maxTokens,
      response_format: { type: 'json_schema', json_schema: { name: 'response', schema } },
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Error HTTP ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  let rawContent = data.choices[0].message.content.trim();

  if (rawContent.startsWith('```')) {
    rawContent = rawContent.replace(/^```json/, '').replace(/^```/, '').replace(/```$/, '').trim();
  }

  return JSON.parse(rawContent);
}

// Occasionally the model still runs out of its token budget mid-string
// despite the maxLength caps in the schema, leaving invalid JSON. One retry
// is a cheap way to paper over that intermittent case before giving up.
async function callLlm(systemPrompt: string, userText: string, schema: object, temperature: number, maxTokens: number) {
  try {
    return await callLlmOnce(systemPrompt, userText, schema, temperature, maxTokens);
  } catch (error) {
    if (error instanceof Error && error.message === 'LLM_NOT_CONFIGURED') throw error;
    return await callLlmOnce(systemPrompt, userText, schema, temperature, maxTokens);
  }
}

function buildTurnSystemPrompt(context: LLMContext): string {
  const inventoryDesc = context.inventory.length
    ? context.inventory.map((item) => `${item.name} x${item.quantity}`).join(', ')
    : 'empty';
  const notesDesc = context.worldNotes.length ? context.worldNotes.join(' | ') : 'none';

  return `IMPORTANT: You must write ONLY in English. Never use Spanish or any other language. All text you generate (thinking, story, item names, notes) must be in English, no exceptions.

IMPORTANT: This game's world is STRICTLY FANTASY — swords, magic, castles, forests, medieval-era settings and creatures. The "retro-terminal" look is only the on-screen display style, NOT part of the game world. NEVER write sci-fi or technology elements: no guns, no computers, no terminals, no futuristic or modern-day devices of any kind, inside the story itself.

You are the Game Master of a dark fantasy RPG, presented through a retro-terminal display. You narrate the world, react to the player's actions, and use tools (toolCalls) to modify the game state when appropriate.
Respond ONLY with a JSON object matching the requested schema, written entirely in English.

-- Current Game State --
- HP: ${context.hp}/${context.hpMax} | Level: ${context.level} | XP: ${context.xp}/${context.xpMax}
- Location: ${context.location.label} [X:${context.location.x}, Y:${context.location.y}]
- Stats: ${JSON.stringify(context.stats)}
- Inventory: ${inventoryDesc}
- Current objective: ${context.objective}
- World notes (facts you already established, do not contradict them): ${notesDesc}
- Map: ${context.worldMapSummary}

Available tools:
1. {"name": "addItem", "args": {"name": string, "quantity": number, "kind": string, "desc": string}}
2. {"name": "dropItem", "args": {"index": number}}
3. {"name": "setObjective", "args": {"objective": string}} — use it when the player's objective changes or is completed
4. {"name": "remember", "args": {"note": string}} — use it to note world facts you invent (NPC names, places, plot) so you don't forget them in future turns

CRITICAL: Return ONLY the JSON.`;
}

export async function fetchLlmResponse(userText: string, context: LLMContext) {
  try {
    const parsed = await callLlm(buildTurnSystemPrompt(context), userText, TURN_RESPONSE_SCHEMA, 0.7, 350);

    return {
      thinking: parsed.thinking || '[No thinking provided]',
      story: parsed.story || '[No story narrative provided]',
      toolCalls: parsed.toolCalls || [],
    };
  } catch (error) {
    if (error instanceof Error && error.message === 'LLM_NOT_CONFIGURED') {
      return {
        thinking: '[CONFIG] The server URL has not been configured.',
        story: 'The link to the Game Master is not configured. Open settings (⚙ LINK) and enter your server URL.',
        toolCalls: [],
      };
    }

    console.error('LLM Connection Error:', error);
    return {
      thinking: `[CRITICAL ERROR]\nDetails: ${error}`,
      story: 'The connection to the Game Master has been lost. Check that the server is running and that the URL in ⚙ LINK is correct.',
      toolCalls: [],
    };
  }
}

// --- Post-narration analyzer agents ---------------------------------------
//
// The narrator is asked to narrate AND remember to call the right tool when
// its own narration implies a state change (movement, an NPC appearing).
// That's unreliable — it can narrate "you arrive at the village" without
// ever calling `move`. These analyzers read the narrator's own story text
// after the fact and decide state changes independently; because each field
// is required by its schema, the decision can't be silently skipped the way
// an optional tool call can be.

const NPC_TYPES = ['OGRE', 'MERCHANT', 'SOLDIER', 'CULTIST', 'NONE'];

const XP_SCHEMA = {
  type: 'object',
  properties: {
    xpAwarded: { type: 'integer', minimum: 0, maximum: 30 },
  },
  required: ['xpAwarded'],
};

const HP_SCHEMA = {
  type: 'object',
  properties: {
    hpDelta: { type: 'integer', minimum: -50, maximum: 50 },
  },
  required: ['hpDelta'],
};

const MAP_SCHEMA = {
  type: 'object',
  properties: {
    locationChanged: { type: 'boolean' },
    x: { type: 'integer', minimum: 0 },
    y: { type: 'integer', minimum: 0 },
    label: { type: 'string', minLength: 1, maxLength: 50 },
  },
  required: ['locationChanged', 'x', 'y', 'label'],
};

const NPC_SCHEMA = {
  type: 'object',
  properties: {
    npcPresent: { type: 'boolean' },
    npcType: { type: 'string', enum: NPC_TYPES },
  },
  required: ['npcPresent', 'npcType'],
};

export interface TurnAnalysis {
  xpAwarded: number;
  hpDelta: number;
  locationChanged: boolean;
  location: { x: number; y: number; label: string };
  npcPresent: boolean;
  npcType: string;
}

const NO_OP_ANALYSIS: Omit<TurnAnalysis, 'location'> = {
  xpAwarded: 0,
  hpDelta: 0,
  locationChanged: false,
  npcPresent: false,
  npcType: 'NONE',
};

async function getXpAwarded(playerInstruction: string, narratorStory: string): Promise<number> {
  const systemPrompt = `You are the XP referee for a fantasy RPG. Always respond in English.
Read the player's action and the narrator's response, then decide how much XP (0 to 30) the player earns this turn. Award more for combat, discovery, clever problem-solving, or meaningful roleplay. Award 0 for trivial actions like looking around or small talk.
Respond ONLY with the JSON object matching the requested schema.`;
  const userText = `Player action: ${playerInstruction}\nNarrator response: ${narratorStory}`;
  const parsed = await callLlm(systemPrompt, userText, XP_SCHEMA, 0.3, 100);
  return typeof parsed.xpAwarded === 'number' ? parsed.xpAwarded : 0;
}

async function getHpDelta(narratorStory: string): Promise<number> {
  const systemPrompt = `You are the HP referee for a fantasy RPG. Always respond in English.
Read the narrator's response and decide if the player took damage or was healed during this turn. Return a negative number for damage, a positive number for healing, or 0 if neither happened.
Respond ONLY with the JSON object matching the requested schema.`;
  const parsed = await callLlm(systemPrompt, `Narrator response: ${narratorStory}`, HP_SCHEMA, 0.3, 100);
  return typeof parsed.hpDelta === 'number' ? parsed.hpDelta : 0;
}

async function getLocationChange(
  narratorStory: string,
  context: LLMContext,
): Promise<{ locationChanged: boolean; location: { x: number; y: number; label: string } }> {
  const systemPrompt = `You are the map tracker for a fantasy RPG. Always respond in English.
Read the narrator's response, the player's current location, and the world map. Decide if the player's location changed during this turn (including arriving at a notable place mentioned in the map). If it changed, return the new coordinates and a short label for the place. If it did not change, return locationChanged: false and repeat the current coordinates/label.

Current location: ${context.location.label} [X:${context.location.x}, Y:${context.location.y}]
Map: ${context.worldMapSummary}

Respond ONLY with the JSON object matching the requested schema.`;
  const parsed = await callLlm(systemPrompt, `Narrator response: ${narratorStory}`, MAP_SCHEMA, 0.3, 120);

  return {
    locationChanged: Boolean(parsed.locationChanged),
    location: {
      x: typeof parsed.x === 'number' ? parsed.x : context.location.x,
      y: typeof parsed.y === 'number' ? parsed.y : context.location.y,
      label: typeof parsed.label === 'string' && parsed.label ? parsed.label : context.location.label,
    },
  };
}

async function getNpcState(narratorStory: string): Promise<{ npcPresent: boolean; npcType: string }> {
  const systemPrompt = `You are the NPC tracker for a fantasy RPG. Always respond in English.
Read the narrator's response and decide if an NPC is now present and interacting with the player. If so, pick whichever npcType best fits: OGRE, MERCHANT, SOLDIER, or CULTIST. If no NPC is present or speaking, set npcPresent to false and npcType to NONE.
Respond ONLY with the JSON object matching the requested schema.`;
  const parsed = await callLlm(systemPrompt, `Narrator response: ${narratorStory}`, NPC_SCHEMA, 0.3, 100);

  return {
    npcPresent: Boolean(parsed.npcPresent),
    npcType: typeof parsed.npcType === 'string' ? parsed.npcType : 'NONE',
  };
}

export async function analyzeNarratorTurn(
  playerInstruction: string,
  narratorStory: string,
  context: LLMContext,
): Promise<TurnAnalysis> {
  const [xpAwarded, hpDelta, locationResult, npcResult] = await Promise.all([
    getXpAwarded(playerInstruction, narratorStory).catch(() => NO_OP_ANALYSIS.xpAwarded),
    getHpDelta(narratorStory).catch(() => NO_OP_ANALYSIS.hpDelta),
    getLocationChange(narratorStory, context).catch(() => ({
      locationChanged: NO_OP_ANALYSIS.locationChanged,
      location: context.location,
    })),
    getNpcState(narratorStory).catch(() => ({
      npcPresent: NO_OP_ANALYSIS.npcPresent,
      npcType: NO_OP_ANALYSIS.npcType,
    })),
  ]);

  return {
    xpAwarded,
    hpDelta,
    locationChanged: locationResult.locationChanged,
    location: locationResult.location,
    npcPresent: npcResult.npcPresent,
    npcType: npcResult.npcType,
  };
}
