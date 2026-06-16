import { useEffect, useMemo, useRef, useState } from 'react';
import { fetchLlmResponse } from './llmService';

type StatName = 'Strength' | 'Perception' | 'Endurance' | 'Charisma' | 'Intelligence' | 'Agility' | 'Luck';
type InventoryItem = { name: string; quantity: number; kind: string; desc?: string };

// Actualizado para separar pensamiento e historia
type ChatEntry = 
  | { role: 'SYSTEM' | 'PLAYER'; text: string }
  | { role: 'Narrator'; story: string; thinking: string };

type ObjGeometry = {
  vertices: [number, number, number][];
  faces: number[][];
  edges: [number, number][];
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
  activeNpc: null | { name: string; title: string; seed: number };
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

const npcProfiles = ['OGRO', 'MERCADER', 'SOLDADO', 'CULTISTA'];
const npcModelUrl = './basic-head-mesh.glb';

const initialState: GameState = {
  hp: 74,
  hpMax: 100,
  xp: 40,
  xpMax: 100,
  level: 3,
  statPoints: 2,
  stats: {
    Strength: 5,
    Perception: 6,
    Endurance: 4,
    Charisma: 7,
    Intelligence: 8,
    Agility: 5,
    Luck: 3,
  },
  location: { x: 7, y: 2, label: 'Outpost Gate' },
  inventory: [
    { name: '9mm Pistol', quantity: 1, kind: 'weapon', desc: '> Arma estándar. Condición: Operativa.' },
    { name: 'Rusty Key', quantity: 1, kind: 'quest', desc: '> Llave pesada de hierro con el número 4.' },
    { name: 'Medkit', quantity: 2, kind: 'consumable', desc: '> Restaura 50 HP.' },
  ],
  activeNpc: null,
};

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function makeMap(width: number, height: number) {
  return Array.from({ length: height }, (_, y) =>
    Array.from({ length: width }, (_, x) => ({ x, y, blocked: (x + y) % 7 === 0 })),
  );
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
  const [chat, setChat] = useState<ChatEntry[]>([
    { role: 'SYSTEM', text: 'C:\\> SISTEMA INICIADO. IA ENLACE ESTABLECIDO.' },
    { role: 'Narrator', story: 'Te despiertas en una habitación oscura. ¿Qué haces?', thinking: '[SISTEMA INTERNO]\nIniciando secuencia de despertar.' },
  ]);
  const map = useMemo(() => makeMap(10, 8), []);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [chat, isThinking]);

  const inventorySummary = game.inventory.map((item) => `${item.name} x${item.quantity}`).join(', ');

  const contextSnapshot = () => ({
    instruction: input.trim(),
    location: game.location,
    hp: `${game.hp}/${game.hpMax}`,
    level: game.level,
    xp: `${game.xp}/${game.xpMax}`,
    stats: game.stats,
    inventory: game.inventory,
    activeNpc: game.activeNpc,
  });

  // LLM Submit Prompt integrado
  const submitPrompt = async (text: string) => {
    if (!text.trim() || isThinking) return;
    
    setChat((prev) => [...prev, { role: 'PLAYER', text }]);
    setInput('');
    setIsThinking(true);

    try {
      // 1. Llamada al servicio LLM (mock)
      const response = await fetchLlmResponse(text, game);

      // 2. Procesamos las herramientas forzadas
      response.toolCalls.forEach((call: any) => {
        if (call.name === 'move') {
           setGame(prev => ({
             ...prev,
             location: {
               x: clamp(prev.location.x + call.args.dx, 0, 9),
               y: clamp(prev.location.y + call.args.dy, 0, 9),
               label: `Sector LLM-${Math.floor(Math.random()*100)}`
             }
           }));
        }
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
        if (call.name === 'spawnNpc') {
           spawnNpc();
        }
      });

      // 3. Añadimos el resultado narrativo y el pensamiento a la consola
      setChat((prev) => [
        ...prev,
        { role: 'Narrator', story: response.story, thinking: response.thinking }
      ]);
    } catch (e) {
      setChat((prev) => [...prev, { role: 'SYSTEM', text: 'ERR: Fallo de conexión con el LLM.' }]);
    } finally {
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
    setGame((prev) => ({ ...prev, activeNpc: { name: npcName, title: 'Sub-LLM activo', seed } }));
    setChat((prev) => [...prev, { role: 'SYSTEM', text: `[TOOL] NPC spawn -> ${npcName}` }]);
  };

  const releaseNpc = () => {
    setGame((prev) => ({ ...prev, activeNpc: null }));
    setChat((prev) => [...prev, { role: 'SYSTEM', text: '[TOOL] NPC control returned to narrator LLM' }]);
  };

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

        <div className="top-meter top-meter-right">
          <span className="top-meter-label">HP</span>
          <div className="top-meter-track">
            <div style={{ width: `${(game.hp / game.hpMax) * 100}%` }} />
          </div>
          <span className="top-meter-value">{game.hp}/{game.hpMax}</span>
        </div>
      </header>

      <main className="layout">
        <section className="left-rail panel">
          <div className="panel-title">NARRATOR LOG</div>
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
                      {game.inventory[selectedItemIndex].desc || `> Objeto genérico tipo: ${game.inventory[selectedItemIndex].kind}.`}
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
                row.map((cell) => (
                  <div key={`${cell.x}-${cell.y}`} className={`map-cell ${cell.blocked ? 'blocked' : ''}`}>
                    {game.location.x === cell.x && game.location.y === cell.y ? '◉' : ''}
                  </div>
                )),
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