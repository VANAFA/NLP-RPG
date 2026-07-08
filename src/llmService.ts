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
//
// `npc` is a required sub-object rather than an optional tool call: the
// narrator must always state whether an NPC is speaking this turn, and if
// so, write their dialogue into `npc.dialogue` — never into `story`. That
// keeps NPC speech reliably separated into its own UI channel (below the
// NPC's head) instead of relying on parsing it back out of prose.
const TURN_RESPONSE_SCHEMA = {
  type: 'object',
  properties: {
    thinking: { type: 'string', maxLength: 100 },
    story: { type: 'string', minLength: 5, maxLength: 260 },
    npc: {
      type: 'object',
      properties: {
        present: { type: 'boolean' },
        name: { type: 'string', maxLength: 30 },
        description: { type: 'string', maxLength: 120 },
        dialogue: { type: 'string', maxLength: 220 },
      },
      required: ['present', 'name', 'description', 'dialogue'],
    },
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
  required: ['thinking', 'story', 'npc', 'toolCalls'],
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

IMPORTANT — NPC dialogue goes in a separate field, never in "story": "story" is ONLY for scene, environment, and action narration. If an NPC is present and speaking this turn (including if the player is directly addressing one), set npc.present to true, give them a name and a short profile description (appearance/role/personality — invent one the first time they appear, then stay consistent with it), and write what they say aloud in npc.dialogue. NEVER put an NPC's spoken words inside "story". If no NPC is present or speaking, set npc.present to false and leave name/description/dialogue as empty strings.

IMPORTANT — "thinking" is YOUR private Game Master scratchpad, not narration and not the player's inner voice. Use it to briefly track continuity or reasoning as the GM: what you're keeping consistent, what you're planning to introduce next, why something just happened. Write it in third person, from the GM's point of view, about the world and the player character — NEVER in first person as if you were the player (no "I will...", no "I wonder...", no "I don't know what I'll do..."). It is never shown to the player.

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

export interface NpcTurnState {
  present: boolean;
  name: string;
  description: string;
  dialogue: string;
}

// Worst case (npc profile + 3 maxed-out tool calls all present at once) adds
// up to roughly 500-600 tokens of schema-constrained content; 900 leaves
// enough headroom that the model finishes its closing brace before hitting
// the cap instead of getting cut off mid-string (the "text cutting off"
// bug — max_tokens ending the response before valid JSON is complete).
const TURN_MAX_TOKENS = 900;

export async function fetchLlmResponse(userText: string, context: LLMContext) {
  try {
    const parsed = await callLlm(buildTurnSystemPrompt(context), userText, TURN_RESPONSE_SCHEMA, 0.7, TURN_MAX_TOKENS);

    const npc: NpcTurnState = {
      present: Boolean(parsed.npc?.present),
      name: typeof parsed.npc?.name === 'string' ? parsed.npc.name : '',
      description: typeof parsed.npc?.description === 'string' ? parsed.npc.description : '',
      dialogue: typeof parsed.npc?.dialogue === 'string' ? parsed.npc.dialogue : '',
    };

    return {
      thinking: parsed.thinking || '[No thinking provided]',
      story: parsed.story || '[No story narrative provided]',
      toolCalls: parsed.toolCalls || [],
      npc,
    };
  } catch (error) {
    const noNpc: NpcTurnState = { present: false, name: '', description: '', dialogue: '' };

    if (error instanceof Error && error.message === 'LLM_NOT_CONFIGURED') {
      return {
        thinking: '[CONFIG] The server URL has not been configured.',
        story: 'The link to the Game Master is not configured. Open settings (⚙ LINK) and enter your server URL.',
        toolCalls: [],
        npc: noNpc,
      };
    }

    console.error('LLM Connection Error:', error);
    return {
      thinking: `[CRITICAL ERROR]\nDetails: ${error}`,
      story: 'The connection to the Game Master has been lost. Check that the server is running and that the URL in ⚙ LINK is correct.',
      toolCalls: [],
      npc: noNpc,
    };
  }
}

// --- Post-narration analyzer agents ---------------------------------------
//
// The narrator is asked to narrate AND remember to call the right tool when
// its own narration implies a state change (movement). That's unreliable —
// it can narrate "you arrive at the village" without ever calling `move`.
// These analyzers read the narrator's own story text after the fact and
// decide state changes independently; because each field is required by its
// schema, the decision can't be silently skipped the way an optional tool
// call can be. (NPC presence/dialogue is no longer analyzed after the fact —
// it's part of the main narrator response itself, see `npc` in
// TURN_RESPONSE_SCHEMA — since that's more reliable than inferring it back
// out of already-written prose.)

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

// The map is always generated as 10 wide x 8 tall (see makeMap(10, 8) in
// App.tsx) — x/y bounds below must match. Leaving them unbounded let the
// model return coordinates far outside the real map, which then got clamped
// to a map edge/corner regardless of what actually happened in the story —
// that's the "map updates weirdly" bug.
const MAP_SCHEMA = {
  type: 'object',
  properties: {
    locationChanged: { type: 'boolean' },
    x: { type: 'integer', minimum: 0, maximum: 9 },
    y: { type: 'integer', minimum: 0, maximum: 7 },
    label: { type: 'string', minLength: 1, maxLength: 50 },
  },
  required: ['locationChanged', 'x', 'y', 'label'],
};

export interface TurnAnalysis {
  xpAwarded: number;
  hpDelta: number;
  locationChanged: boolean;
  location: { x: number; y: number; label: string };
}

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
Read the narrator's response, the player's current location, and the world map, then decide where the player is now. You have full freedom to place the player wherever best fits the story — including indoor or freeform locations (a room, a cellar, a courtyard, a cave) that aren't one of the map's named outdoor landmarks; don't limit yourself to only the landmarks listed. When the player clearly arrives at one of the map's named landmarks, use that landmark's exact coordinates; otherwise pick whatever nearby coordinates best represent where the scene is now taking place, moving them as far as the story implies. Movement that is only described as heading toward / walking in the direction of a place, without confirming arrival, does NOT count as a change yet — keep locationChanged false and repeat the current coordinates in that case. Coordinates must stay within the map's bounds (X: 0-9, Y: 0-7).

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

export async function analyzeNarratorTurn(
  playerInstruction: string,
  narratorStory: string,
  context: LLMContext,
): Promise<TurnAnalysis> {
  const [xpAwarded, hpDelta, locationResult] = await Promise.all([
    getXpAwarded(playerInstruction, narratorStory).catch(() => 0),
    getHpDelta(narratorStory).catch(() => 0),
    getLocationChange(narratorStory, context).catch(() => ({
      locationChanged: false,
      location: context.location,
    })),
  ]);

  return {
    xpAwarded,
    hpDelta,
    locationChanged: locationResult.locationChanged,
    location: locationResult.location,
  };
}
