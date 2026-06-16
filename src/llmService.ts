export interface LLMContext {
  hp: number;
  hpMax: number;
  stats: Record<string, number>;
  location: { x: number; y: number; label: string };
  inventory: any[];
}

export async function fetchLlmResponse(userText: string, context: LLMContext) {
  // Simulamos el delay del modelo
  await new Promise(resolve => setTimeout(resolve, 1500));

  const thinking = `[EVALUANDO INPUT DEL OPERADOR]
Mensaje recibido: "${userText}"
-- Analizando Estado Actual --
Vitalidad: ${context.hp}/${context.hpMax}
Coordenadas: ${context.location.label} [${context.location.x}, ${context.location.y}]
Objetos en inventario: ${context.inventory.length}
Stats detectados: STR(${context.stats.Strength}) PER(${context.stats.Perception})
[DECISIÓN DEL MODELO]
Ejecutando protocolo de estrés: Forzando llamadas aleatorias a todas las herramientas disponibles.`;

  const story = "There will be consecuences to your actions... When the LLM is actually implemented. Allan please add details";

  const toolCalls: { name: string; args: any }[] = [];

  // 1. Mover al jugador
  toolCalls.push({ 
    name: 'move', 
    args: { dx: Math.random() > 0.5 ? 1 : -1, dy: Math.random() > 0.5 ? 1 : -1 } 
  });

  // 2. Spawn NPC
  toolCalls.push({ name: 'spawnNpc', args: {} });

  // 3. Añadir ítem
  const junkNames = ['Cable Pelado', 'Batería de Litio', 'Engranaje Roto', 'Chip Quemado'];
  toolCalls.push({
    name: 'addItem',
    args: {
      name: junkNames[Math.floor(Math.random() * junkNames.length)],
      quantity: 1,
      kind: 'junk',
      desc: '> Objeto de prueba inyectado directamente por el LLM simulado.'
    }
  });

  // 4. Tirar ítem
  if (context.inventory.length > 0) {
    const randomInvIndex = Math.floor(Math.random() * context.inventory.length);
    toolCalls.push({ name: 'dropItem', args: { index: randomInvIndex } });
  }

  // 5. Modificar HP
  const hpChange = Math.random() > 0.5 ? -10 : +10;
  toolCalls.push({ name: 'modifyHp', args: { amount: hpChange } });

  // 6. Incrementar xp random
  const xpGain = Math.floor(Math.random() * 20) + 5;
  toolCalls.push({ name: 'gainXp', args: { amount: xpGain } });

  // Devolvemos exactamente lo que tu App.tsx está esperando
  return { thinking, story, toolCalls };
}