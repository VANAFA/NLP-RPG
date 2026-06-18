export interface LLMContext {
  hp: number;
  hpMax: number;
  stats: Record<string, number>;
  location: { x: number; y: number; label: string };
  inventory: any[];
}

// 1. Envolvemos la URL de Hugging Face dentro del proxy público para engañar al CORS
const HF_API_URL = 'https://api-inference.huggingface.co/models/pengu1n7/fireball-qwen3-4b-merged/v1/chat/completions';

// 2. IMPORTANTE: Borra este token y genera uno nuevo cuando termines el proyecto, 
// ya que al pegarlo en el chat de IA acaba de quedar expuesto.
const HF_TOKEN = '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'; 

export async function fetchLlmResponse(userText: string, context: LLMContext) {
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
    const response = await fetch(HF_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${HF_TOKEN}` 
      },
      body: JSON.stringify({
        model: "pengu1n7/fireball-qwen3-4b-merged", 
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
      // Si recibes un error 503, significa que el modelo está cargando. Solo espera 20 segs y reintenta.
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
      story: "La conexión con el servidor remoto se ha perdido.",
      toolCalls: []
    };
  }
}