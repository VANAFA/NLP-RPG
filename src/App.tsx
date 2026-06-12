import { useEffect, useMemo, useRef, useState } from 'react';

type StatName = 'Strength' | 'Perception' | 'Endurance' | 'Charisma' | 'Intelligence' | 'Agility' | 'Luck';
type InventoryItem = { name: string; quantity: number; kind: string; desc?: string };
type ChatEntry = { role: 'LLM' | 'PLAYER' | 'SYSTEM'; text: string };
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
const npcModelUrl = '/basic-head-mesh.obj';

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

function App() {
  const [game, setGame] = useState<GameState>(initialState);
  // Recuerda el índice del ítem seleccionado en la grilla
  const [selectedItemIndex, setSelectedItemIndex] = useState<number | null>(null);

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
    { role: 'LLM', text: 'Te despiertas en una habitación oscura. ¿Qué haces?' },
  ]);
  const map = useMemo(() => makeMap(10, 8), []);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' });
  }, [chat]);

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

  const submitPrompt = (text: string) => {
    const snapshot = contextSnapshot();
    setChat((prev) => [
      ...prev,
      { role: 'PLAYER', text },
      {
        role: 'SYSTEM',
        text: `CTX => loc:${snapshot.location.label} (${snapshot.location.x},${snapshot.location.y}) | hp:${snapshot.hp} | lvl:${snapshot.level} | xp:${snapshot.xp} | inv:${snapshot.inventory.length} items`,
      },
      { role: 'LLM', text: 'Narrador: respuesta simulada. Conecta aquí tu API LLM y las herramientas.' },
    ]);
    setInput('');
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
            {chat.map((entry, index) => (
              <div key={index} className={`chat-line ${entry.role.toLowerCase()}`}>
                <span className="chat-role">{entry.role}:</span> {entry.text}
              </div>
            ))}
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
              placeholder="Type your command for the narrator LLM..."
            />
          </form>
          <div className="tool-row">
            <button type="button" onClick={() => submitPrompt(input || 'look around')}>Send</button>
            <button type="button" onClick={() => toolAction('inventory.add( item )')}>Add Item Tool</button>
            <button type="button" onClick={() => toolAction('inventory.drop( item )')}>Drop Item Tool</button>
            <button type="button" onClick={() => toolAction('npc.spawn()')}>Spawn NPC</button>
          </div>
          <div className="help-text">
            LLM payload: instruction, location, stats, inventory, hp/xp/level, and active NPC state.
          </div>
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
                <span style={{ opacity: 0.4 }}>Haz clic en un ítem para inspeccionarlo...</span>
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
            <div className="map-footer">Location: {game.location.label}</div>
            <div className="map-controls">
              <button type="button" onClick={() => movePlayer(0, -1)}>↑</button>
              <div>
                <button type="button" onClick={() => movePlayer(-1, 0)}>←</button>
                <button type="button" onClick={() => movePlayer(1, 0)}>→</button>
              </div>
              <button type="button" onClick={() => movePlayer(0, 1)}>↓</button>
            </div>
          </div>

          <div className="panel npc-panel">
            <div className="panel-title">NPC WINDOW</div>
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
            <div className="npc-footer">
              {game.activeNpc ? `${game.activeNpc.name} · ${game.activeNpc.title}` : 'Spawn an NPC to show the 3D wireframe.'}
            </div>
            <div className="npc-controls">
              <button type="button" onClick={spawnNpc}>Spawn NPC</button>
              <button type="button" onClick={releaseNpc}>Return Control</button>
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

    // 1. Proyectamos todos los vértices como antes
      const points = model.vertices.map((vertex) => project(vertex, time));
      const cx = width / 2;
      const cy = height / 2;

      // Sets para rastrear qué vértices y aristas son visibles
      const visibleVertices = new Set<number>();
      const visibleEdges = new Set<string>();
      const visibleFaces: { indices: number[]; depth: number }[] = [];

      // 2. Backface Culling y cálculo de profundidad
      model.faces.forEach((faceIndices) => {
        if (faceIndices.length < 3) return;

        const p0 = points[faceIndices[0]];
        const p1 = points[faceIndices[1]];
        const p2 = points[faceIndices[2]];

        // Producto cruzado 2D de los primeros tres vértices proyectados
        const crossProduct = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);

        // Si crossProduct es mayor a 0, la cara apunta hacia atrás.
        // (Nota: Si el modelo se renderiza al revés -se ve la nuca y no la cara-, 
        // cambiá este > 0 por un < 0. Depende de cómo exportaste el .obj)
        if (crossProduct < 0) return;

        // Si pasó el culling, calculamos su profundidad para el Painter's Algorithm
        let avgZ = 0;
        faceIndices.forEach((index, i) => {
          avgZ += points[index].z;
          visibleVertices.add(index); // Marcamos el vértice como visible
          
          // Registramos la arista como visible
          const nextIndex = faceIndices[(i + 1) % faceIndices.length];
          const edgeKey = index < nextIndex ? `${index}:${nextIndex}` : `${nextIndex}:${index}`;
          visibleEdges.add(edgeKey);
        });
        avgZ /= faceIndices.length;

        visibleFaces.push({ indices: faceIndices, depth: avgZ });
      });

      // 3. Z-Sorting de las caras que sobrevivieron
      visibleFaces.sort((a, b) => b.depth - a.depth);

// 4. Dibujamos caras, aristas y vértices TODO JUNTO en orden de profundidad
      visibleFaces.forEach((face) => {
        // A. Dibujar la cara sólida (Polígono)
        ctx.beginPath();
        face.indices.forEach((index, faceIndex) => {
          const point = points[index];
          const px = cx + point.x;
          const py = cy + point.y;
          if (faceIndex === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        
        ctx.fillStyle = '#114400'; // Color sólido de relleno
        ctx.fill();
        
        ctx.strokeStyle = '#33ff00'; // Color del wireframe
        ctx.lineWidth = 1;
        ctx.stroke();

        // B. Dibujar los vértices (puntitos) correspondientes SOLO a esta cara
        ctx.fillStyle = '#33ff00';
        face.indices.forEach((index) => {
          const point = points[index];
          ctx.beginPath();
          ctx.arc(cx + point.x, cy + point.y, 1.5, 0, Math.PI * 2);
          ctx.fill();
        });
      });
      
      // ELIMINAR EL PASO 5 ANTERIOR (ya no necesitamos iterar visibleVertices)

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

export default App;
