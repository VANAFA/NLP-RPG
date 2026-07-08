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

async function callLlm(systemPrompt: string, userText: string, schema: object, temperature: number, maxTokens: number) {
  const baseUrl = localStorage.getItem(LS_BASE_URL_KEY);
  if (!baseUrl) {
    throw new Error('LLM_NOT_CONFIGURED');
  }

  const apiKey = localStorage.getItem(LS_API_KEY_KEY);

  const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/v1/chat/completions`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      // Evita la página de advertencia HTML que ngrok free tier muestra a navegaciones de browser.
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

function buildTurnSystemPrompt(context: LLMContext): string {
  const inventoryDesc = context.inventory.length
    ? context.inventory.map((item) => `${item.name} x${item.quantity}`).join(', ')
    : 'vacío';
  const notesDesc = context.worldNotes.length ? context.worldNotes.join(' | ') : 'ninguna';

  return `Sos el Game Master de un RPG de terminal retro oscuro. Narrás el mundo, reaccionás a las acciones del jugador, y usás herramientas (toolCalls) para modificar el estado del juego cuando corresponda.
Respondé ÚNICAMENTE con un objeto JSON matching el schema pedido.

-- Estado Actual del Juego --
- HP: ${context.hp}/${context.hpMax} | Nivel: ${context.level} | XP: ${context.xp}/${context.xpMax}
- Ubicación: ${context.location.label} [X:${context.location.x}, Y:${context.location.y}]
- Stats: ${JSON.stringify(context.stats)}
- Inventario: ${inventoryDesc}
- Objetivo actual: ${context.objective}
- Notas del mundo (hechos que ya estableciste, no los contradigas): ${notesDesc}
- Mapa: ${context.worldMapSummary}

Herramientas disponibles:
1. {"name": "move", "args": {"dx": number, "dy": number}}
2. {"name": "addItem", "args": {"name": string, "quantity": number, "kind": string, "desc": string}}
3. {"name": "dropItem", "args": {"index": number}}
4. {"name": "spawnNpc", "args": {}}
5. {"name": "setObjective", "args": {"objective": string}} — usalo cuando el objetivo del jugador cambie o se complete
6. {"name": "remember", "args": {"note": string}} — usalo para anotar hechos del mundo que inventes (nombres de NPCs, lugares, trama) para no olvidarlos en turnos futuros

CRÍTICO: Devolvé SOLO el JSON.`;
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
        thinking: '[CONFIG] No se ha configurado la URL del servidor.',
        story: 'El enlace con el Game Master no está configurado. Abrí los ajustes (⚙ LINK) e ingresá la URL de tu servidor.',
        toolCalls: [],
      };
    }

    console.error('LLM Connection Error:', error);
    return {
      thinking: `[CRITICAL ERROR]\nDetails: ${error}`,
      story: 'La conexión con el Game Master se ha perdido. Verificá que el servidor esté encendido y que la URL en ⚙ LINK sea la correcta.',
      toolCalls: [],
    };
  }
}

export async function generateNewCharacter(worldMapSummary: string): Promise<NewCharacter> {
  const systemPrompt = `Sos el Game Master creando un nuevo personaje para arrancar una partida de un RPG de terminal retro oscuro.
Inventá un personaje coherente: quién es, cuál es su situación actual, y su primer objetivo.

Reglas:
- Stats (Strength, Perception, Endurance, Charisma, Intelligence, Agility, Luck): personaje NOVATO, cada stat entre 1 y 6, sin superar un total aproximado de 22-26 puntos entre las 7. No es un héroe poderoso, es alguien común arrancando su historia.
- Inventory: 2 a 4 objetos lógicos para ese personaje y su situación (con nombre, cantidad, tipo, y una descripción breve de una frase).
- Location: elegí una celda del mapa (coordenadas dentro del rango disponible) coherente con el tipo de terreno, con una etiqueta descriptiva del lugar.
- Objective: una frase clara y concreta de la misión u objetivo inicial del personaje.
- Story: 2 a 4 frases narrativas en castellano presentando al jugador quién es su personaje, dónde está, y cuál es su objetivo. Este es el primer mensaje que el jugador va a leer, así que tiene que enganchar.

Mapa disponible: ${worldMapSummary}

Respondé ÚNICAMENTE con el objeto JSON matching el schema pedido.`;

  const parsed = await callLlm(
    systemPrompt,
    'Generá un nuevo personaje para iniciar la partida.',
    CHARACTER_CREATION_SCHEMA,
    0.9,
    550,
  );

  return {
    stats: parsed.stats,
    inventory: parsed.inventory,
    location: parsed.location,
    objective: parsed.objective,
    story: parsed.story,
  };
}
