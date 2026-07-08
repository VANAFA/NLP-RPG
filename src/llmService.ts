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

export interface NewCharacter {
  stats: Record<string, number>;
  inventory: InventoryItemInput[];
  location: { x: number; y: number; label: string };
  objective: string;
  story: string;
}

// Claves de localStorage donde la app guarda la URL/API key del servidor LLM.
// Exportadas para que App.tsx las reuse al leer/escribir el panel de ajustes.
export const LS_BASE_URL_KEY = 'nlprpg.llm.baseUrl';
export const LS_API_KEY_KEY = 'nlprpg.llm.apiKey';

// Debe coincidir con el alias usado en `--lora-modules <alias>=...` / el backend local.
const MODEL_NAME = 'fireball-qwen3-4b-lora-10k';

const TOOL_NAMES = ['move', 'addItem', 'dropItem', 'spawnNpc', 'setObjective', 'remember'];

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
          args: { type: 'object' },
        },
        required: ['name', 'args'],
      },
    },
  },
  required: ['thinking', 'story', 'toolCalls'],
};

const CHARACTER_CREATION_SCHEMA = {
  type: 'object',
  properties: {
    stats: {
      type: 'object',
      properties: {
        Strength: { type: 'integer', minimum: 1, maximum: 8 },
        Perception: { type: 'integer', minimum: 1, maximum: 8 },
        Endurance: { type: 'integer', minimum: 1, maximum: 8 },
        Charisma: { type: 'integer', minimum: 1, maximum: 8 },
        Intelligence: { type: 'integer', minimum: 1, maximum: 8 },
        Agility: { type: 'integer', minimum: 1, maximum: 8 },
        Luck: { type: 'integer', minimum: 1, maximum: 8 },
      },
      required: ['Strength', 'Perception', 'Endurance', 'Charisma', 'Intelligence', 'Agility', 'Luck'],
    },
    inventory: {
      type: 'array',
      minItems: 2,
      maxItems: 4,
      items: {
        type: 'object',
        properties: {
          name: { type: 'string', minLength: 2, maxLength: 40 },
          quantity: { type: 'integer', minimum: 1 },
          kind: { type: 'string', minLength: 2, maxLength: 20 },
          desc: { type: 'string', minLength: 5, maxLength: 90 },
        },
        required: ['name', 'quantity', 'kind', 'desc'],
      },
    },
    location: {
      type: 'object',
      properties: {
        x: { type: 'integer', minimum: 0 },
        y: { type: 'integer', minimum: 0 },
        label: { type: 'string', minLength: 3, maxLength: 50 },
      },
      required: ['x', 'y', 'label'],
    },
    objective: { type: 'string', minLength: 10, maxLength: 150 },
    story: { type: 'string', minLength: 40, maxLength: 350 },
  },
  required: ['stats', 'inventory', 'location', 'objective', 'story'],
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

You are the Game Master of a dark, retro-terminal RPG. You narrate the world, react to the player's actions, and use tools (toolCalls) to modify the game state when appropriate.
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
1. {"name": "move", "args": {"dx": number, "dy": number}}
2. {"name": "addItem", "args": {"name": string, "quantity": number, "kind": string, "desc": string}}
3. {"name": "dropItem", "args": {"index": number}}
4. {"name": "spawnNpc", "args": {}}
5. {"name": "setObjective", "args": {"objective": string}} — use it when the player's objective changes or is completed
6. {"name": "remember", "args": {"note": string}} — use it to note world facts you invent (NPC names, places, plot) so you don't forget them in future turns

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

export async function generateNewCharacter(worldMapSummary: string): Promise<NewCharacter> {
  const systemPrompt = `IMPORTANT: You must write ONLY in English. Never use Spanish or any other language. All text you generate (item names, descriptions, objective, story) must be in English, no exceptions.

You are the Game Master creating a new character to start a dark, retro-terminal RPG session.
Invent a coherent character: who they are, their current situation, and their first objective.

Rules:
- Stats (Strength, Perception, Endurance, Charisma, Intelligence, Agility, Luck): a NOVICE character, each stat between 1 and 6, totaling roughly 22-26 points across all 7. Not a powerful hero — an ordinary person starting their story.
- Inventory: 2 to 4 logical items for that character and their situation (with name, quantity, kind, and a brief one-sentence description) — all in English.
- Location: pick a map cell (coordinates within the available range) consistent with the terrain type, with a descriptive label for the place, in English.
- Objective: a clear, concrete sentence describing the character's initial mission or goal, in English.
- Story: 2 to 4 narrative sentences in English introducing the player to who their character is, where they are, and what their objective is. This is the first message the player will read, so it needs to hook them.

Available map: ${worldMapSummary}

Respond ONLY with the JSON object matching the requested schema, written entirely in English.`;

  const parsed = await callLlm(
    systemPrompt,
    'Generate a new character to start the game.',
    CHARACTER_CREATION_SCHEMA,
    0.9,
    800,
  );

  return {
    stats: parsed.stats,
    inventory: parsed.inventory,
    location: parsed.location,
    objective: parsed.objective,
    story: parsed.story,
  };
}
