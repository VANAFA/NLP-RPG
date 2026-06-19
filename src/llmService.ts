export interface LLMContext {
  hp: number;
  hpMax: number;
  stats: Record<string, number>;
  location: { x: number; y: number; label: string };
  inventory: any[];
}

// Claves de localStorage donde la app guarda la URL/API key del servidor de Kaggle.
// Exportadas para que App.tsx las reuse al leer/escribir el panel de ajustes.
export const LS_BASE_URL_KEY = 'nlprpg.llm.baseUrl';
export const LS_API_KEY_KEY = 'nlprpg.llm.apiKey';

// Debe coincidir con el alias usado en `--lora-modules <alias>=...` del notebook de Kaggle.
const MODEL_NAME = 'fireball-qwen3-4b-lora-10k';

export async function fetchLlmResponse(userText: string, context: LLMContext) {
  const baseUrl = localStorage.getItem(LS_BASE_URL_KEY);

  if (!baseUrl) {
    return {
      thinking: '[CONFIG] No se ha configurado la URL del servidor de Kaggle.',
      story: 'El enlace con el Game Master no está configurado. Abrí los ajustes (⚙ LINK) e ingresá la URL de tu sesión de Kaggle.',
      toolCalls: []
    };
  }

  const apiKey = localStorage.getItem(LS_API_KEY_KEY);

  const systemPrompt = `You are the Game Master of a dark, retro-terminal RPG.
You must respond ONLY with a valid JSON object matching the following TypeScript interface:
{
  "thinking": "string",
  "story": "string",
  "toolCalls": Array<{ name: 'move' | 'addItem' | 'dropItem' | 'spawnNpc', args: any }>
}

-- Current Game Context --
- Player HP: ${context.hp}/${context.hpMax}
- Location: ${context.location.label} [X:${context.location.x}, Y:${context.location.y}]
- Stats: ${JSON.stringify(context.stats)}
- Inventory: ${JSON.stringify(context.inventory.map(i => i.name))}

Available Tools:
1. {"name": "move", "args": {"dx": number, "dy": number}}
2. {"name": "addItem", "args": {"name": "string", "quantity": number, "kind": "string", "desc": "string"}}
3. {"name": "dropItem", "args": {"index": number}}
4. {"name": "spawnNpc", "args": {}}

CRITICAL: Return ONLY raw JSON.`;

  try {
    const response = await fetch(`${baseUrl.replace(/\/+$/, '')}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Evita la página de advertencia HTML que ngrok free tier muestra a navegaciones de browser.
        'ngrok-skip-browser-warning': 'true',
        ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userText }
        ],
        temperature: 0.2,
        max_tokens: 1024
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Error HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    let rawContent = data.choices[0].message.content.trim();

    if (rawContent.startsWith("```")) {
      rawContent = rawContent.replace(/^```json/, "").replace(/^```/, "").replace(/```$/, "").trim();
    }

    const parsedResponse = JSON.parse(rawContent);

    return {
      thinking: parsedResponse.thinking || '[No thinking provided]',
      story: parsedResponse.story || '[No story narrative provided]',
      toolCalls: parsedResponse.toolCalls || []
    };

  } catch (error) {
    console.error("LLM Connection Error:", error);
    return {
      thinking: `[CRITICAL ERROR]\nDetails: ${error}`,
      story: 'La conexión con el servidor de Kaggle se ha perdido. Verificá que la sesión esté encendida y que la URL en ⚙ LINK sea la correcta.',
      toolCalls: []
    };
  }
}
