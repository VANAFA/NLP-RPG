import { useEffect, useMemo, useRef, useState } from 'react';

type StatName = 'Strength' | 'Perception' | 'Endurance' | 'Charisma' | 'Intelligence' | 'Agility' | 'Luck';
type InventoryItem = { name: string; quantity: number; kind: string };
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
    { name: '9mm Pistol', quantity: 1, kind: 'weapon' },
    { name: 'Rusty Key', quantity: 1, kind: 'quest' },
    { name: 'Medkit', quantity: 2, kind: 'consumable' },
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
  const [input, setInput] = useState('');
  const [chat, setChat] = useState<ChatEntry[]>([
    { role: 'SYSTEM', text: 'C:\\> SISTEMA INICIADO. IA ENLACE ESTABLECIDO.' },
    { role: 'LLM', text: 'Te despiertas en una habitación oscura. ¿Qué haces?' },
  ]);
  const map = useMemo(() => makeMap(10, 10), []);
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
          <span className="top-meter-label">HP</span>
          <div className="top-meter-track">
            <div style={{ width: `${(game.hp / game.hpMax) * 100}%` }} />
          </div>
          <span className="top-meter-value">{game.hp}/{game.hpMax}</span>
        </div>
        <div className="top-meter top-meter-right">
          <span className="top-meter-label">XP</span>
          <div className="top-meter-track xp">
            <div style={{ width: `${(game.xp / game.xpMax) * 100}%` }} />
          </div>
          <span className="top-meter-value">Level {game.level}</span>
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
                return (
                  <div key={index} className="inventory-slot">
                    {item ? (
                      <>
                        <span>{item.name}</span>
                        <small>{item.quantity} · {item.kind}</small>
                      </>
                    ) : null}
                  </div>
                );
              })}
            </div>
            <div className="inventory-summary">{inventorySummary || 'Empty'}</div>
          </div>

          <div className="panel stats-panel">
            <div className="panel-title">STATS</div>
            <div className="stats-grid">
              {statNames.map((stat) => (
                <button key={stat} type="button" className="stat-card" onClick={() => levelStat(stat)}>
                  <span>{stat}</span>
                  <strong>{game.stats[stat]}</strong>
                </button>
              ))}
            </div>
            <div className="stat-points">Points available: {game.statPoints} · Click a stat to level it up</div>
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
      const wobble = Math.sin(time * 0.001) * 0.06;
      const yaw = Math.PI + wobble;
      // Slight tilt for personality
      const pitch = -0.15;
      const roll = 0;

      const [x, y, z] = point;
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
      const depth = pitchZ * cy + rollX * sy + 3.2;
      // Float bob: slow sine wave on screen Y
      const bob = Math.sin(time * 0.0018 + seed * 100) * 5;
      return {
        x: yawX * 110 / depth,
        y: pitchY * 110 / depth + bob,
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

      ctx.strokeStyle = '#33ff00';
      ctx.fillStyle = 'rgba(51,255,0,0.07)';
      ctx.lineWidth = 1;

      model.faces.forEach((face) => {
        ctx.beginPath();
        face.forEach((index, faceIndex) => {
          const point = points[index];
          const px = cx + point.x;
          const py = cy + point.y;
          if (faceIndex === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.closePath();
        ctx.fill();
      });

      model.edges.forEach(([from, to]) => {
        const a = points[from];
        const b = points[to];
        ctx.beginPath();
        ctx.moveTo(cx + a.x, cy + a.y);
        ctx.lineTo(cx + b.x, cy + b.y);
        ctx.stroke();
      });

      points.forEach((point) => {
        ctx.beginPath();
        ctx.arc(cx + point.x, cy + point.y, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = '#33ff00';
        ctx.fill();
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

export default App;
