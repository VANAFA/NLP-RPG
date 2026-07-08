import { useEffect, useRef, useState } from 'react';
import { fetchLlmResponse, analyzeNarratorTurn, LS_BASE_URL_KEY, LS_API_KEY_KEY, type LLMContext, type TurnAnalysis, type NpcTurnState } from './llmService';

type StatName = 'Strength' | 'Perception' | 'Endurance' | 'Charisma' | 'Intelligence' | 'Agility' | 'Luck';
type InventoryItem = { name: string; quantity: number; kind: string; desc?: string };
type TerrainType = 'plain' | 'mountain' | 'forest' | 'village' | 'castle' | 'swamp';

type MapCell = { 
  x: number; 
  y: number; 
  terrain: TerrainType; 
  blocked: boolean;
};

// Actualizado para separar pensamiento e historia
type ChatEntry = 
  | { role: 'SYSTEM' | 'PLAYER'; text: string }
  | { role: 'Narrator'; story: string; thinking: string };

type ObjGeometry = {
  vertices: [number, number, number][];
  faces: number[][];
  edges: [number, number][];
};

// Everything the NPC "is" and "says" lives here — rendered below its head in
// the NPC panel, never in the narrator log. `dialogue` accumulates each line
// the NPC speaks while it stays the same NPC (reset when a different NPC
// takes over, see submitPrompt).
type ActiveNpc = {
  name: string;
  description: string;
  seed: number;
  dialogue: string[];
};
type GameState = {
  hp: number;
  hpMax: number;
  xp: number;
  xpMax: number;
  level: number;
  statPoints: number;
  stats: Record<StatName, number>;
  location: { x: number; y: number; label: string };
  inventory: InventoryItem[];
  activeNpc: null | ActiveNpc;
  objective: string;
  worldNotes: string[];
};

const SAVE_KEY = 'nlprpg.save';

const getTerrainVisual = (terrain: TerrainType) => {
  switch (terrain) {
    case 'mountain': return { char: '▲', opacity: 0.8, weight: 'normal', size: '2.8rem' };
    case 'forest':   return { char: '♣', opacity: 0.5, weight: 'normal', size: '1.2rem' };
    case 'village':  return { char: '⌂', opacity: 1,   weight: 'bold',   size: '2.8rem' };
    case 'castle':   return { char: '♜', opacity: 1,   weight: 'bold',   size: '2.8rem' };
    case 'swamp':    return { char: '≈', opacity: 0.7, weight: 'normal', size: '2rem' };
    case 'plain':
    default:         return { char: '·', opacity: 0.2, weight: 'normal', size: '1.2rem' };
  }
};

const statNames: StatName[] = [
  'Strength',
  'Perception',
  'Endurance',
  'Charisma',
  'Intelligence',
  'Agility',
  'Luck',
];

const npcProfiles = ['OGRE', 'MERCHANT', 'SOLDIER', 'CULTIST'];
const npcModelUrl = './basic-head-mesh.obj';

// Fixed for every game: character/room intro, starting items, and starting
// position. Only stats are randomized fresh each time (see randomizeLowStats).
// Strictly fantasy setting — no sci-fi/tech elements.
const OPENING_STORY =
  "You don't remember your name — only the cold, and the taste of ash on your tongue. You're a survivor, nothing more: no title, no banner, just the will to keep breathing.\n\n" +
  "You come to at the foot of a crumbling stone gate, half-swallowed by creeping moss and drifting fog. Torches gutter in rusted iron sconces nearby, and a faded inscription is carved into the archway above you, worn smooth by centuries. Whatever happened here, it happened long ago — and you're the only living soul in sight.";

const initialState: GameState = {
  hp: 100,
  hpMax: 100,
  xp: 0,
  xpMax: 100,
  level: 1,
  statPoints: 0,
  stats: {
    Strength: 3,
    Perception: 3,
    Endurance: 3,
    Charisma: 3,
    Intelligence: 3,
    Agility: 3,
    Luck: 3,
  },
  location: { x: 7, y: 2, label: 'The Broken Gate' },
  inventory: [
    { name: 'Rusty Shortsword', quantity: 1, kind: 'weapon', desc: '> A worn blade, notched but still sharp enough to bite.' },
    { name: 'Rusty Key', quantity: 1, kind: 'quest', desc: '> Heavy iron key stamped with the numeral IV.' },
    { name: 'Healing Poultice', quantity: 2, kind: 'consumable', desc: '> A bundle of herbs and bandages. Restores 50 HP.' },
  ],
  activeNpc: null,
  objective: 'Uncover what happened to this place, and find a way out alive.',
  worldNotes: [],
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

// Novice-level character: each stat rolled 1-6, so every new game starts weak.
function randomizeLowStats(): GameState['stats'] {
  const stats = {} as GameState['stats'];
  statNames.forEach((name) => {
    stats[name] = 1 + Math.floor(Math.random() * 6);
  });
  return stats;
}

function makeMap(width: number, height: number): MapCell[][] {
  // 1. Crear matriz vacía
  let map: MapCell[][] = Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) => ({ x, y, terrain: 'plain' as TerrainType, blocked: false }))
  );

  // 2. Sembrar cúmulos (puntos de inicio de biomas)
  const seeds = [
    { type: 'mountain', x: 2, y: 2 },
    { type: 'forest', x: 10, y: 5 },
    { type: 'swamp', x: 12, y: 1 }
  ];

  seeds.forEach(seed => {
    // Generar radio de cúmulo
    for (let i = -2; i <= 2; i++) {
      for (let j = -2; j <= 2; j++) {
        const nx = seed.x + i, ny = seed.y + j;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height && Math.random() > 0.3) {
          map[ny][nx].terrain = seed.type as TerrainType;
          if (seed.type === 'mountain') map[ny][nx].blocked = true;
        }
      }
    }
  });

  // 3. Añadir puntos de interés individuales (Castillos/Villas)
  map[5][7].terrain = 'village';
  map[1][1].terrain = 'castle';
  
  return map;
}

function summarizeMap(map: MapCell[][], location: { x: number; y: number }): string {
  const height = map.length;
  const width = map[0]?.length ?? 0;
  const notable: string[] = [];

  map.forEach((row) => row.forEach((cell) => {
    if (cell.terrain !== 'plain') notable.push(`(${cell.x},${cell.y}):${cell.terrain}`);
  }));

  return `Grid ${width}x${height}. Player at (${location.x},${location.y}). Notable locations: ${notable.join(', ') || 'none'}.`;
}

function parseObjModel(text: string): ObjGeometry {
  const vertices: [number, number, number][] = [];
  const faces: number[][] = [];

  text.split(/\r?\n/).forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('#')) return;

    if (trimmed.startsWith('v ')) {
      const [, x, y, z] = trimmed.split(/\s+/);
      const vertex = [Number(x), Number(y), Number(z)] as [number, number, number];
      if (vertex.every((value) => Number.isFinite(value))) {
        vertices.push(vertex);
      }
      return;
    }

    if (trimmed.startsWith('f ')) {
      const indices = trimmed
        .split(/\s+/)
        .slice(1)
        .map((token) => Number(token.split('/')[0]) - 1)
        .filter((index) => Number.isInteger(index) && index >= 0);

      if (indices.length >= 3) {
        faces.push(indices);
      }
    }
  });

  const edgeSet = new Set<string>();
  const edges: [number, number][] = [];

  faces.forEach((face) => {
    face.forEach((from, index) => {
      const to = face[(index + 1) % face.length];
      const key = from < to ? `${from}:${to}` : `${to}:${from}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        edges.push([from, to]);
      }
    });
  });

  return { vertices, faces, edges };
}

export default function App() {
  const [game, setGame] = useState<GameState>(initialState);
  
  // Recuerda el índice del ítem seleccionado en la grilla
  const [selectedItemIndex, setSelectedItemIndex] = useState<number | null>(null);

  // Estados nuevos para el LLM y el modo Debug
  const [isDebugMode, setIsDebugMode] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  // Ajustes de conexión al servidor de Kaggle (URL + API key, persistidos en localStorage)
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [baseUrlInput, setBaseUrlInput] = useState('');
  const [apiKeyInput, setApiKeyInput] = useState('');

  // Función para tirar el ítem y limpiar la selección
  const dropItem = (index: number) => {
    setGame((prev) => {
      const newInventory = [...prev.inventory];
      const droppedItem = newInventory[index];
      newInventory.splice(index, 1); // Lo elimina del array
      
      // Opcional: Mandar un aviso a la consola del Narrador
      setChat((c) => [...c, { role: 'SYSTEM', text: `[SYSTEM] Dropped: ${droppedItem.name}` }]);
      
      return { ...prev, inventory: newInventory };
    });
    setSelectedItemIndex(null); // Oculta el menú inferior
  };

  const [input, setInput] = useState('');
  const [chat, setChat] = useState<ChatEntry[]>([]);
  const [map, setMap] = useState<MapCell[][]>(() => makeMap(10, 8));
  const [isInitializing, setIsInitializing] = useState(true);
  const logRef = useRef<HTMLDivElement>(null);
  // React 18 StrictMode double-invokes mount effects in dev; without this
  // guard that would fire two concurrent character-generation requests.
  const hasInitializedRef = useRef(false);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [chat, isThinking]);

  // Hidrata los campos de ajustes desde localStorage una sola vez al montar
  useEffect(() => {
    setBaseUrlInput(localStorage.getItem(LS_BASE_URL_KEY) ?? '');
    setApiKeyInput(localStorage.getItem(LS_API_KEY_KEY) ?? '');
  }, []);

  // Starts a new game: character/room intro, items, and position are fixed;
  // only stats are rolled fresh. No LLM call needed to begin playing.
  const startNewGame = () => {
    const freshMap = makeMap(10, 8);

    setGame({ ...initialState, stats: randomizeLowStats() });
    setMap(freshMap);
    setChat([
      { role: 'SYSTEM', text: 'C:\\> SYSTEM INITIALIZED. AI LINK ESTABLISHED.' },
      { role: 'Narrator', story: OPENING_STORY, thinking: '[INTERNAL SYSTEM]\nNew character initialized.' },
    ]);
    setIsInitializing(false);
  };

  // Al montar: intenta cargar una partida guardada; si no hay, genera una nueva.
  useEffect(() => {
    if (hasInitializedRef.current) return;
    hasInitializedRef.current = true;

    const saved = localStorage.getItem(SAVE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setGame(parsed.game);
        setChat(parsed.chat);
        setMap(parsed.map);
        setIsInitializing(false);
        return;
      } catch {
        // Guardado corrupto: seguimos a generar una partida nueva.
      }
    }
    startNewGame();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Persiste la partida en cada cambio (una vez que ya terminó de inicializar).
  useEffect(() => {
    if (isInitializing) return;
    localStorage.setItem(SAVE_KEY, JSON.stringify({ game, chat, map }));
  }, [game, chat, map, isInitializing]);

  const newGame = () => {
    localStorage.removeItem(SAVE_KEY);
    startNewGame();
  };

  const saveSettings = () => {
    localStorage.setItem(LS_BASE_URL_KEY, baseUrlInput.trim());
    localStorage.setItem(LS_API_KEY_KEY, apiKeyInput.trim());
    setIsSettingsOpen(false);
    setChat((prev) => [...prev, { role: 'SYSTEM', text: '[SYSTEM] Connection settings saved.' }]);
  };

  // Applies the background analyzer agents' decisions (XP, HP, location)
  // once they resolve — runs a moment after the narrator's story is already
  // on screen, per the chosen "don't block on this" UX. NPC state is NOT
  // handled here — it comes straight from the narrator's own response (see
  // submitPrompt), which is more reliable than inferring it back out of
  // already-written prose.
  const applyTurnAnalysis = (analysis: TurnAnalysis, width: number, height: number) => {
    setGame((prev) => {
      let xp = prev.xp;
      let level = prev.level;
      let xpMax = prev.xpMax;
      let statPoints = prev.statPoints;

      if (analysis.xpAwarded > 0) {
        xp += analysis.xpAwarded;
        while (xp >= xpMax) {
          xp -= xpMax;
          level += 1;
          statPoints += 1;
          xpMax = Math.round(xpMax * 1.2);
        }
      }

      const hp = analysis.hpDelta !== 0 ? clamp(prev.hp + analysis.hpDelta, 0, prev.hpMax) : prev.hp;

      const location = analysis.locationChanged
        ? {
            x: clamp(analysis.location.x, 0, width - 1),
            y: clamp(analysis.location.y, 0, height - 1),
            label: analysis.location.label,
          }
        : prev.location;

      return { ...prev, xp, level, xpMax, statPoints, hp, location };
    });
  };

  // Merges the narrator's per-turn `npc` field into game state. Runs every
  // turn: when npc.present is true we either start a fresh profile+dialogue
  // (new NPC) or append to the current one (same NPC continuing to talk);
  // when false, the NPC panel goes back to empty.
  const applyNpcUpdate = (npc: NpcTurnState) => {
    setGame((prev) => {
      if (!npc.present) {
        return prev.activeNpc ? { ...prev, activeNpc: null } : prev;
      }

      const isSameNpc = prev.activeNpc && prev.activeNpc.name === npc.name;
      const priorDialogue = isSameNpc ? prev.activeNpc!.dialogue : [];
      const dialogue = npc.dialogue ? [...priorDialogue, npc.dialogue].slice(-20) : priorDialogue;

      const activeNpc: ActiveNpc = {
        name: npc.name || prev.activeNpc?.name || 'Stranger',
        description: npc.description || (isSameNpc ? prev.activeNpc!.description : ''),
        seed: isSameNpc ? prev.activeNpc!.seed : Math.random(),
        dialogue,
      };

      return { ...prev, activeNpc };
    });
  };

  // LLM Submit Prompt integrado
  const submitPrompt = async (text: string) => {
    if (!text.trim() || isThinking) return;

    setChat((prev) => [...prev, { role: 'PLAYER', text }]);
    setInput('');
    setIsThinking(true);

    const context: LLMContext = {
      hp: game.hp,
      hpMax: game.hpMax,
      xp: game.xp,
      xpMax: game.xpMax,
      level: game.level,
      stats: game.stats,
      location: game.location,
      inventory: game.inventory,
      objective: game.objective,
      worldNotes: game.worldNotes,
      worldMapSummary: summarizeMap(map, game.location),
    };
    const width = map[0]?.length ?? 10;
    const height = map.length;

    try {
      const response = await fetchLlmResponse(text, context);

      // Narrator-driven tools: discrete, explicit actions the model states directly.
      response.toolCalls.forEach((call: any) => {
        if (call.name === 'addItem') {
           setGame(prev => ({ ...prev, inventory: [...prev.inventory, call.args] }));
        }
        if (call.name === 'dropItem') {
           setGame(prev => {
             const newInv = [...prev.inventory];
             if (newInv.length > 0 && call.args.index < newInv.length) {
                newInv.splice(call.args.index, 1);
             }
             return { ...prev, inventory: newInv };
           });
           setSelectedItemIndex(null);
        }
        if (call.name === 'setObjective') {
           setGame(prev => ({ ...prev, objective: call.args.objective ?? prev.objective }));
        }
        if (call.name === 'remember') {
           setGame(prev => ({
             ...prev,
             worldNotes: call.args.note ? [...prev.worldNotes, call.args.note].slice(-20) : prev.worldNotes,
           }));
        }
      });

      // NPC profile/dialogue is a first-class part of the narrator's response
      // (not narration text) — route it straight to the NPC panel, whether
      // the player is addressing the NPC directly or just witnessing it talk.
      applyNpcUpdate(response.npc);

      // Show the story right away — don't make the player wait through the
      // analyzer calls below just to read the response they're waiting on.
      setChat((prev) => [
        ...prev,
        { role: 'Narrator', story: response.story, thinking: response.thinking }
      ]);
      setIsThinking(false);

      // XP/HP/location/NPC-presence are decided by dedicated analyzer calls
      // reading the narrator's own story text, in the background.
      analyzeNarratorTurn(text, response.story, context)
        .then((analysis) => applyTurnAnalysis(analysis, width, height))
        .catch((error) => console.error('Turn analysis failed:', error));
    } catch (e) {
      setChat((prev) => [...prev, { role: 'SYSTEM', text: 'ERR: LLM connection failed.' }]);
      setIsThinking(false);
    }
  };

  const movePlayer = (dx: number, dy: number) =>
    setGame((prev) => ({
      ...prev,
      location: {
        x: clamp(prev.location.x + dx, 0, 9),
        y: clamp(prev.location.y + dy, 0, 9),
        label: `Sector ${clamp(prev.location.x + dx, 0, 9)}-${clamp(prev.location.y + dy, 0, 9)}`,
      },
    }));

  const toolAction = (name: string) => {
    setChat((prev) => [...prev, { role: 'SYSTEM', text: `[TOOL] ${name}` }]);
  };

  const levelStat = (stat: StatName) => {
    setGame((prev) => {
      if (prev.statPoints <= 0) return prev;
      return {
        ...prev,
        statPoints: prev.statPoints - 1,
        stats: { ...prev.stats, [stat]: prev.stats[stat] + 1 },
      };
    });
  };

  const spawnNpc = () => {
    const seed = Math.random();
    const npcName = npcProfiles[Math.floor(seed * npcProfiles.length)] ?? 'DESCONOCIDO';
    setGame((prev) => ({
      ...prev,
      activeNpc: { name: npcName, description: '(Debug) Manually spawned test NPC.', seed, dialogue: ['...'] },
    }));
    setChat((prev) => [...prev, { role: 'SYSTEM', text: `[TOOL] NPC spawn -> ${npcName}` }]);
  };

  const releaseNpc = () => {
    setGame((prev) => ({ ...prev, activeNpc: null }));
    setChat((prev) => [...prev, { role: 'SYSTEM', text: '[TOOL] NPC control returned to narrator LLM' }]);
  };

  if (isInitializing) {
    return (
      <div className="app-shell">
        <div className="scanlines" />
        <div className="logo-screen" style={{ height: '100vh', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
          <div className="logo">NLP-RPG</div>
          <div className="logo-sub">Generating character...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="app-shell">
      <div className="scanlines" />
      <header className="top-bar">
        <div className="top-meter top-meter-left">
          <span className="top-meter-label">XP</span>
          <div className="top-meter-track xp">
            <div style={{ width: `${(game.xp / game.xpMax) * 100}%` }} />
          </div>
          <span className="top-meter-value">Level {game.level}</span>
        </div>
        
        {/* BOTÓN TOGGLE DEBUG AL CENTRO DE LA BARRA */}
        <button 
          onClick={() => setIsDebugMode(!isDebugMode)}
          style={{ 
            backgroundColor: isDebugMode ? '#ffff00' : 'transparent', 
            color: isDebugMode ? '#000' : '#ffff00',
            border: '1px solid #ffff00',
            padding: '0.2rem 1rem',
            fontWeight: 'bold',
            cursor: 'pointer',
            fontFamily: 'inherit'
          }}
        >
          {isDebugMode ? 'DEBUG [ON]' : 'DEBUG [OFF]'}
        </button>

        {/* BOTÓN DE AJUSTES DE CONEXIÓN AL SERVIDOR DE KAGGLE */}
        <button
          onClick={() => setIsSettingsOpen(true)}
          style={{
            backgroundColor: 'transparent',
            color: '#33ff00',
            border: '1px solid #33ff00',
            padding: '0.2rem 1rem',
            fontWeight: 'bold',
            cursor: 'pointer',
            fontFamily: 'inherit'
          }}
        >
          ⚙ LINK
        </button>

        {/* BOTÓN PARA REINICIAR LA PARTIDA CON UN PERSONAJE NUEVO */}
        <button
          onClick={() => {
            if (window.confirm('Start a new game? Current progress will be lost.')) newGame();
          }}
          style={{
            backgroundColor: 'transparent',
            color: '#ff5555',
            border: '1px solid #ff5555',
            padding: '0.2rem 1rem',
            fontWeight: 'bold',
            cursor: 'pointer',
            fontFamily: 'inherit'
          }}
        >
          ⟳ NEW GAME
        </button>

        <div className="top-meter top-meter-right">
          <span className="top-meter-label">HP</span>
          <div className="top-meter-track">
            <div style={{ width: `${(game.hp / game.hpMax) * 100}%` }} />
          </div>
          <span className="top-meter-value">{game.hp}/{game.hpMax}</span>
        </div>
      </header>

      {isSettingsOpen && (
        <div className="settings-overlay" onClick={() => setIsSettingsOpen(false)}>
          <div className="settings-panel" onClick={(e) => e.stopPropagation()}>
            <div className="panel-title">LLM SERVER SETTINGS</div>
            <label>
              Base URL
              <input
                value={baseUrlInput}
                onChange={(e) => setBaseUrlInput(e.target.value)}
                placeholder="https://xxxx.ngrok-free.app"
              />
            </label>
            <label>
              API Key
              <input
                type="password"
                value={apiKeyInput}
                onChange={(e) => setApiKeyInput(e.target.value)}
                placeholder="(optional)"
              />
            </label>
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.5rem' }}>
              <button type="button" onClick={() => setIsSettingsOpen(false)}>Cancel</button>
              <button type="button" onClick={saveSettings}>Save</button>
            </div>
          </div>
        </div>
      )}

      <main className="layout">
        <section className="left-rail panel">
          <div className="panel-title">NARRATOR LOG</div>
          <div style={{ padding: '0.3rem 0.5rem', borderBottom: '1px dashed rgba(51,255,0,0.3)', fontSize: '0.85rem', opacity: 0.85 }}>
            <strong>OBJECTIVE:</strong> {game.objective || 'No objective set.'}
          </div>
          <div className="chat-log" ref={logRef}>
            {chat.map((entry, index) => {
              // RENDERIZADO ESPECIAL PARA EL LLM
              if (entry.role === 'Narrator') {
                return (
                  <div key={index} className="chat-line llm" style={{ marginBottom: '1rem', marginTop: '0.5rem' }}>
                    {isDebugMode && (
                      <div style={{ 
                        color: '#ffff00', 
                        whiteSpace: 'pre-wrap', 
                        marginBottom: '0.5rem', 
                        padding: '0.5rem', 
                        borderLeft: '2px solid #ffff00',
                        backgroundColor: 'rgba(255, 255, 0, 0.05)',
                        fontSize: '0.9rem'
                      }}>
                        {entry.thinking}
                      </div>
                    )}
                    <div>
                      <span className="chat-role">Narrator:</span> {entry.story}
                    </div>
                  </div>
                );
              }

              return (
                <div key={index} className={`chat-line ${entry.role.toLowerCase()}`}>
                  <span className="chat-role">{entry.role}:</span> {entry.text}
                </div>
              );
            })}
            
            {/* TEXTO DE PROCESAMIENTO VISUAL */}
            {isThinking && (
               <div style={{ opacity: 0.5, fontStyle: 'italic', marginTop: '0.5rem' }}>[ Narrator is thinking... ]</div>
            )}
          </div>
          <form
            className="input-row"
            onSubmit={(event) => {
              event.preventDefault();
              if (input.trim()) submitPrompt(input.trim());
            }}
          >
            <span className="prompt">&gt;</span>
            <input
              autoFocus
              value={input}
              onChange={(event) => setInput(event.target.value)}
              disabled={isThinking}
              placeholder="What do you do?"
            />
          </form>
        </section>

        <section className="center-column">
          <div className="panel inventory-panel">
            <div className="panel-title">INVENTORY</div>
            <div className="inventory-grid">
              {Array.from({ length: 12 }).map((_, index) => {
                const item = game.inventory[index];
                const isSelected = selectedItemIndex === index;
                
                return (
                  <button 
                    key={index} 
                    type="button"
                    className="inventory-slot"
                    onClick={() => item ? setSelectedItemIndex(index) : setSelectedItemIndex(null)}
                    style={{ 
                      cursor: item ? 'pointer' : 'default',
                      /* Si está seleccionado: Borde verde brillante
                         Si tiene ítem: Borde verde oscuro semi-transparente
                         Si está vacío: Borde punteado muy sutil */
                      border: isSelected 
                        ? '1px solid #33ff00' 
                        : item 
                          ? '1px solid rgba(51, 255, 0, 0.3)' 
                          : '1px dashed rgba(51, 255, 0, 0.1)',
                      /* El hueco vacío tendrá un fondo un pelín más negro para que resalte la caja */
                      backgroundColor: isSelected 
                        ? 'rgba(51, 255, 0, 0.1)' 
                        : item 
                          ? 'transparent' 
                          : 'rgba(0, 0, 0, 0.3)',
                      // minHeight: '40px', /* Asegura que la caja mantenga su forma aunque no haya texto */
                      minHeight: '70px',
                      display: 'flex',
                      flexDirection: 'column',
                      justifyContent: 'center',
                      padding: '0.5rem',
                      /* ... tus otros estilos ... */
                      height: '70px', /* <-- LA BALA DE PLATA: Altura estricta para llenos y vacíos */
                      maxHeight: '70px',
                      overflow: 'hidden', /* Evita que si un nombre es muy largo, deforme el botón */
                      /* ... el resto ... */
                    }}
                  >
                    {item ? (
                      <>
                        <span style={{ display: 'block', color: isSelected ? '#fff' : '#33ff00' }}>{item.name}</span>
                        <small style={{ opacity: 0.7, color: '#33ff00' }}>{item.quantity} · {item.kind}</small>
                      </>
                    ) : (
                      /* Truco fantasma: Replicamos el DOM exacto para igualar la altura */
                      <>
                        <span style={{ display: 'block', color: '#33ff00', opacity: 0.3 }}>[ Empty ]</span>
                        {/* Este small ocupa espacio físico pero no se ve */}
                        <small style={{ opacity: 0 }}>-</small> 
                      </>
                    )}
                  </button>
                );
              })}
            </div>
            
            {/* Panel inferior de Inspección / Acción */}
            <div className="inventory-details" style={{ marginTop: '1rem', borderTop: '1px dashed #33ff00', paddingTop: '0.8rem', minHeight: '60px' }}>
              {selectedItemIndex !== null && game.inventory[selectedItemIndex] ? (
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <div style={{ flex: 1, paddingRight: '1rem' }}>
                    <strong style={{ color: '#fff' }}>{game.inventory[selectedItemIndex].name}</strong>
                    <p style={{ margin: '0.4rem 0 0 0', fontSize: '0.9rem', opacity: 0.8, whiteSpace: 'pre-wrap' }}>
                      {game.inventory[selectedItemIndex].desc || `> Generic item, type: ${game.inventory[selectedItemIndex].kind}.`}
                    </p>
                  </div>
                  <button 
                    type="button" 
                    onClick={() => dropItem(selectedItemIndex)}
                    style={{ 
                      color: '#000', 
                      backgroundColor: '#33ff00', 
                      border: 'none', 
                      padding: '0.4rem 0.8rem', 
                      cursor: 'pointer', 
                      fontWeight: 'bold',
                      fontFamily: 'inherit'
                    }}
                  >
                    &gt; DROP
                  </button>
                </div>
              ) : (
                <span style={{ opacity: 0.4 }}>Click an item to inspect it...</span>
              )}
            </div>
          </div>

          <div className="panel stats-panel">
            <div className="panel-title">STATS</div>
            {/* Usamos whiteSpace: 'pre' y una fuente monoespaciada para garantizar la cuadrícula ASCII */}
            <div className="ascii-stats-container" style={{whiteSpace: 'pre', fontSize: '2rem', color: '#33ff00' }}>
              {/* Filas de Atributos */}
              {statNames.map((stat) => {
                const namePadded = stat.padEnd(12, ' ');
                const valPadded = game.stats[stat].toString().padStart(2, '0');
                
                return (
                  <div key={stat} className="ascii-stat-row" style={{ display: 'block' }}>
                    {` ${namePadded} : ${valPadded} `}
                    {game.statPoints > 0 ? (
                      <span 
                        className="ascii-btn" 
                        onClick={() => levelStat(stat)}
                        style={{ 
                          cursor: 'pointer', 
                          color: '#ffff00', 
                          fontWeight: 'bold',
                          textShadow: '0 0 5px #ffff00'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.color = '#fff';
                          e.currentTarget.style.backgroundColor = '#33ff00';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.color = '#ffff00';
                          e.currentTarget.style.backgroundColor = 'transparent';
                        }}
                      >
                        [+]
                      </span>
                    ) : (
                      '   '
                    )}
                    {``}
                  </div>
                );
              })}
              
              {/* {`+-----------------------+\n`} */}
            </div>
          </div>
        </section>

        <section className="right-column">
          <div className="panel map-panel">
            <div className="panel-title">WORLD MAP</div>
<div className="map-grid">
              {map.map((row) =>
                row.map((cell) => {
                  const isPlayerHere = game.location.x === cell.x && game.location.y === cell.y;
                  const visual = getTerrainVisual(cell.terrain);

                  return (
                    <div 
                      key={`${cell.x}-${cell.y}`} 
                      className="map-cell"
                      style={{ 
                        position: 'relative',
                        // Optional: subtle background for mountains to make them look dense
                        backgroundColor: cell.terrain === 'mountain' ? 'rgba(51, 255, 0, 0.05)' : 'transparent'
                      }}
                    >
                      {/* Terrain Graphic */}
                      <span style={{ 
                        opacity: visual.opacity, 
                        fontWeight: visual.weight as any, // "as any" evita quejas tontas de TypeScript aquí
                        fontSize: visual.size,            // <-- Toma el tamaño de tu diccionario
                        lineHeight: 1                     // <-- Evita que el ícono grande empuje la grilla hacia abajo
                      }}>
                        {visual.char}
                      </span>

                      {/* Player Overlay */}
                      {isPlayerHere && (
                        <span style={{ 
                          position: 'absolute', 
                          color: '#fff', // White core for the player
                          textShadow: '0 0 8px #33ff00', // Heavy green glow
                          fontWeight: 'bold',
                          zIndex: 10
                        }}>
                          ◉
                        </span>
                      )}
                    </div>
                  );
                }),
              )}
            </div>
            {/* if debug then show buttons */}
              {isDebugMode && (
              <div className="map-controls" style={{ marginTop: '1rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <button type="button" onClick={() => movePlayer(0, -1)} style={{ marginBottom: '0.5rem' }}>↑</button>
                <div>
                  <button type="button" onClick={() => movePlayer(-1, 0)} style={{ marginRight: '0.5rem' }}>←</button>
                  <button type="button" onClick={() => movePlayer(1, 0)}>→</button>
                </div>
                <button type="button" onClick={() => movePlayer(0, 1)} style={{ marginTop: '0.5rem' }}>↓</button>
              </div>
              )
            }
          </div>

          <div className="panel npc-panel">
            <div className="panel-title">NPC</div>
            <div className="npc-frame">
              {game.activeNpc ? (
                <WireframeModel src={npcModelUrl} seed={game.activeNpc.seed} />
              ) : (
                <div className="logo-screen">
                  <div className="logo">NLP-RPG</div>
                  <div className="logo-sub">No NPC active</div>
                </div>
              )}
            </div>

            {/* NPC profile + everything it says lives here, below its head —
                never in the narrator log, whether the narrator is describing
                the NPC or the player is talking to it directly. */}
            {game.activeNpc && (
              <div className="npc-dialogue-panel">
                <div className="npc-profile">
                  <strong className="npc-name">{game.activeNpc.name}</strong>
                  {game.activeNpc.description && (
                    <p className="npc-description">{game.activeNpc.description}</p>
                  )}
                </div>
                <div className="npc-dialogue-log">
                  {game.activeNpc.dialogue.length > 0 ? (
                    game.activeNpc.dialogue.map((line, index) => (
                      <p key={index} className="npc-dialogue-line">&ldquo;{line}&rdquo;</p>
                    ))
                  ) : (
                    <p className="npc-dialogue-line npc-dialogue-empty">...</p>
                  )}
                </div>
              </div>
            )}

            <div className="npc-controls">
              {isDebugMode && (
                <>
                  <button type="button" onClick={spawnNpc} style={{ backgroundColor: '#33ff00', color: '#000', border: 'none', padding: '0.4rem 0.8rem', cursor: 'pointer', fontWeight: 'bold', fontFamily: 'inherit' }}>Spawn NPC</button>
                  <button type="button" onClick={releaseNpc} style={{ backgroundColor: '#ff3333', color: '#fff', border: 'none', padding: '0.4rem 0.8rem', cursor: 'pointer', fontWeight: 'bold', fontFamily: 'inherit' }}>Return Control</button>
                </>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

function WireframeModel({ src, seed }: { src: string; seed: number }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [model, setModel] = useState<ObjGeometry | null>(null);

  useEffect(() => {
    let active = true;

    fetch(src)
      .then((response) => response.text())
      .then((text) => {
        if (active) {
          setModel(parseObjModel(text));
        }
      })
      .catch(() => {
        if (active) {
          setModel({ vertices: [], faces: [], edges: [] });
        }
      });

    return () => {
      active = false;
    };
  }, [src]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !model) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    window.addEventListener('resize', resize);

    let raf = 0;

    const project = (point: [number, number, number], time: number) => {
      // Rotate 180° around Y axis to face the viewer, plus a tiny wobble
      const wobble = Math.sin(time * 0.0015) * 0.03 + 0.05;
      const yaw = Math.PI + wobble;
      // Slight tilt for personality
      const pitch = -0.15;
      const roll = 0;

      const [z, y, x] = point;
      const cy = Math.cos(yaw);
      const sy = Math.sin(yaw);
      const cx = Math.cos(pitch);
      const sx = Math.sin(pitch);
      const cz = Math.cos(roll);
      const sz = Math.sin(roll);

      const rollX = x * cz - y * sz;
      const rollY = x * sz + y * cz;
      const pitchY = rollY * cx - z * sx;
      const pitchZ = rollY * sx + z * cx;
      const yawX = rollX * cy - pitchZ * sy;
      const depth = -pitchZ * cy + rollX * sy + 20;
      // Float bob: slow sine wave on screen Y
      const bob = Math.sin(time * 0.0018 + seed * 100) * 5;
      return {
        x: yawX * 1150 / depth - 50,
        y: -(pitchY * 1150 / depth) + bob + 157,
        z: depth,
      };
    };

    const draw = (time: number) => {
      const width = canvas.getBoundingClientRect().width;
      const height = canvas.getBoundingClientRect().height;
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(0,0,0,0.88)';
      ctx.fillRect(0, 0, width, height);

      const points = model.vertices.map((vertex) => project(vertex, time));
      const cx = width / 2;
      const cy = height / 2;

      const visibleVertices = new Set<number>();
      const visibleEdges = new Set<string>();
      const visibleFaces: { indices: number[]; depth: number }[] = [];

      model.faces.forEach((faceIndices) => {
        if (faceIndices.length < 3) return;

        const p0 = points[faceIndices[0]];
        const p1 = points[faceIndices[1]];
        const p2 = points[faceIndices[2]];

        const crossProduct = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);

        if (crossProduct < 0) return;

        let avgZ = 0;
        faceIndices.forEach((index: number, i: number) => {
          avgZ += points[index].z;
          visibleVertices.add(index); 
          
          const nextIndex = faceIndices[(i + 1) % faceIndices.length];
          const edgeKey = index < nextIndex ? `${index}:${nextIndex}` : `${nextIndex}:${index}`;
          visibleEdges.add(edgeKey);
        });
        avgZ /= faceIndices.length;

        visibleFaces.push({ indices: faceIndices, depth: avgZ });
      });

      visibleFaces.sort((a, b) => b.depth - a.depth);

      visibleFaces.forEach((face) => {
        ctx.beginPath();
        face.indices.forEach((index: number, faceIndex: number) => {
          const point = points[index];
          const px = cx + point.x;
          const py = cy + point.y;
          if (faceIndex === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        
        ctx.fillStyle = '#114400'; 
        ctx.fill();
        
        ctx.strokeStyle = '#33ff00'; 
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.fillStyle = '#33ff00';
        face.indices.forEach((index: number) => {
          const point = points[index];
          ctx.beginPath();
          ctx.arc(cx + point.x, cy + point.y, 1.5, 0, Math.PI * 2);
          ctx.fill();
        });
      });
      
      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', resize);
    };
  }, [model, seed]);

  return <canvas ref={canvasRef} className="wireframe-canvas" />;
}