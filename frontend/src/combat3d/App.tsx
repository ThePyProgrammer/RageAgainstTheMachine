import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { createOpponentInput } from "./ai/difficulty";
import { CombatView } from "./render/CombatView";
import { createDefaultControlPipeline } from "./bci/controlPipeline";
import {
  FRAME_MS,
  createCombatConfig,
  createStateFromSeed,
  stepSimulation,
} from "./engine";
import { InputQueue } from "./engine/inputQueue";
import { SeededRNG } from "./engine/seededRng";
import { DebugPanel } from "./debug/DebugPanel";
import { TauntBubble } from "./ui/TauntBubble";
import type { BCIStreamPacket } from "@ragemachine/bci-shared";
import { createCombat3DSocket } from "./net/socket";
import { type JoinResponse, type TauntPacket, type TelemetryPacket } from "./net/contracts";
import type { CombatState, InputSample } from "./engine/types";
import type { BarrierBreakEvent } from "./engine/barriers";
import { barriersFromObstacles, type Barrier } from "./engine/barriers";
import { generateObstacles } from "./render/Combat3DScene";

const WS_URL = import.meta.env.VITE_COMBAT3D_WS_URL || "ws://localhost:4001";

const mapTelemetry = (packet: TelemetryPacket): BCIStreamPacket => {
  const base = {
    confidence: packet.confidence,
    timestamp: packet.timeMs,
    sessionId: packet.sessionId,
  };

  if (packet.kind === "features") {
    const ordered = Object.keys(packet.payload)
      .filter((key) => key.startsWith("f"))
      .sort()
      .map((key) => packet.payload[key] ?? 0);
    return {
      ...base,
      kind: "features",
      features: ordered,
    };
  }

  return {
    ...base,
    kind: "classifier",
    label: "idle",
    probabilities: {
      left: packet.payload.left ?? 0,
      right: packet.payload.right ?? 0,
      throttle_up: packet.payload.throttle_up ?? 0,
      throttle_down: packet.payload.throttle_down ?? 0,
      fire: packet.payload.fire ?? 0,
      idle: packet.payload.idle ?? 0,
    },
  };
};

/* ── Keyboard held-key state ────────────────────────────────────────── */

const TRACKED_KEYS = new Set([
  "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight",
  "w", "W", "a", "A", "s", "S", "d", "D", " ",
]);

interface KeyState {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  fire: boolean;
}

const createKeyState = (): KeyState => ({
  up: false, down: false, left: false, right: false, fire: false,
});

/** Sample held-key state into a deterministic InputSample at the given sim time. */
const sampleKeyboard = (keys: KeyState, simTimeMs: number): InputSample => {
  let throttle = 0;
  let turn = 0;
  if (keys.up) throttle += 1;
  if (keys.down) throttle -= 1;
  if (keys.left) turn -= 0.8;
  if (keys.right) turn += 0.8;
  return { timestamp: simTimeMs, throttle, turn, fire: keys.fire };
};

/** Returns true if any movement/fire key is currently held. */
const hasActiveInput = (keys: KeyState): boolean =>
  keys.up || keys.down || keys.left || keys.right || keys.fire;

export interface DebugState {
  queueLen: number;
  rttMs: number;
  lastInput: { throttle: number; turn: number; fire: boolean; source: string };
  playerPos: { x: number; y: number; yaw: number };
  projectileCount: number;
  lastEvent: string;
  tick: number;
}

const INITIAL_DEBUG: DebugState = {
  queueLen: 0,
  rttMs: 0,
  lastInput: { throttle: 0, turn: 0, fire: false, source: "none" },
  playerPos: { x: 0, y: 0, yaw: 0 },
  projectileCount: 0,
  lastEvent: "—",
  tick: 0,
};

export const App = () => {
  const [tauntText, setTauntText] = useState("");
  const [playing, setPlaying] = useState(false);
  const [debug, setDebug] = useState<DebugState>(INITIAL_DEBUG);
  const [barrierBreaks, setBarrierBreaks] = useState<BarrierBreakEvent[]>([]);

  const stateRef = useRef<CombatState>(createStateFromSeed(0x2f2f));
  const config = useMemo(() => createCombatConfig(0x2f2f), []);
  const playerQueueRef = useRef(new InputQueue());
  const enemyQueueRef = useRef(new InputQueue());
  const rngRef = useRef(new SeededRNG(config.seed));
  const pipelineRef = useRef(createDefaultControlPipeline());
  const socketRef = useRef<ReturnType<typeof createCombat3DSocket> | null>(null);
  const keyStateRef = useRef<KeyState>(createKeyState());
  const prevScoreRef = useRef({ player: 0, enemy: 0 });
  const lastEventRef = useRef("—");

  // Barrier collision state — initialized from the same deterministic obstacle generation
  const barriersRef = useRef<Barrier[]>(
    barriersFromObstacles(generateObstacles(0x2f2f, 20)),
  );

  const lastFrameMsRef = useRef(performance.now());
  const accumulatorMsRef = useRef(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const socket = createCombat3DSocket({
      url: WS_URL,
      onTelemetry: (packet: TelemetryPacket) => {
        const sharedPacket = mapTelemetry(packet);
        const controls = pipelineRef.current.onPacket(sharedPacket);

        const input: InputSample = {
          timestamp: sharedPacket.timestamp,
          throttle: controls.throttle,
          turn: controls.turn,
          fire: controls.fire,
        };
        playerQueueRef.current.push(input);

        setDebug((current) => ({ ...current, queueLen: playerQueueRef.current.length + enemyQueueRef.current.length }));
      },
      onSession: (_payload: JoinResponse) => {
        // Socket connected — could trigger UI state in future
      },
      onTaunt: (payload: TauntPacket) => {
        setTauntText(payload.tone);
      },
    });

    socketRef.current = socket;

    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  useEffect(() => {
    let alive = true;

    const loop = (nowMs: number) => {
      if (!alive) {
        return;
      }

      const frameDelta = Math.min(250, nowMs - lastFrameMsRef.current);
      lastFrameMsRef.current = nowMs;
      accumulatorMsRef.current += frameDelta;

      while (accumulatorMsRef.current >= FRAME_MS) {
        const currentState = stateRef.current;
        const simTimeMs = currentState.simTimeMs + FRAME_MS;

        // ── Keyboard: sample held keys at sim time ──
        const kbInput = sampleKeyboard(keyStateRef.current, simTimeMs);
        if (hasActiveInput(keyStateRef.current)) {
          playerQueueRef.current.push(kbInput);
        }

        const deterministicEnemy = createOpponentInput(
          { x: currentState.player.x, y: currentState.player.y },
          currentState.enemy,
          currentState.difficulty,
          rngRef.current,
          simTimeMs,
        );
        enemyQueueRef.current.push(deterministicEnemy);

        const playerInput = playerQueueRef.current.drainUpTo(simTimeMs);
        const enemyInput = enemyQueueRef.current.drainUpTo(simTimeMs);
        const result = stepSimulation(
          currentState,
          {
            config,
            playerInput,
            enemyInput,
            dtMs: FRAME_MS,
            barriers: barriersRef.current,
          },
          rngRef.current,
        );

        stateRef.current = result.state;

        // Emit barrier break events to React state for rendering
        if (result.newBreakEvents.length > 0) {
          setBarrierBreaks((prev) => [...prev, ...result.newBreakEvents]);
        }

        // Track score change events
        const s = result.state.score;
        if (s.player !== prevScoreRef.current.player) {
          lastEventRef.current = `PlayerHit @${result.state.tick}`;
        } else if (s.enemy !== prevScoreRef.current.enemy) {
          lastEventRef.current = `EnemyHit @${result.state.tick}`;
        }
        prevScoreRef.current = s;

        if (stateRef.current.tick % 6 === 0) {
          setDebug({
            queueLen: playerQueueRef.current.length + enemyQueueRef.current.length,
            rttMs: 0,
            lastInput: {
              throttle: playerInput.throttle,
              turn: playerInput.turn,
              fire: playerInput.fire,
              source: hasActiveInput(keyStateRef.current) ? "keyboard" : "bci",
            },
            playerPos: {
              x: stateRef.current.player.x,
              y: stateRef.current.player.y,
              yaw: stateRef.current.player.yaw,
            },
            projectileCount: stateRef.current.projectiles.length,
            lastEvent: lastEventRef.current,
            tick: stateRef.current.tick,
          });
        }

        accumulatorMsRef.current -= FRAME_MS;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    const applyKey = (event: KeyboardEvent, value: boolean) => {
      if (!TRACKED_KEYS.has(event.key)) return;
      event.preventDefault();
      const keys = keyStateRef.current;
      const k = event.key.toLowerCase();
      if (k === "arrowup" || k === "w") keys.up = value;
      else if (k === "arrowdown" || k === "s") keys.down = value;
      else if (k === "arrowleft" || k === "a") keys.left = value;
      else if (k === "arrowright" || k === "d") keys.right = value;
      else if (k === " ") keys.fire = value;
    };

    const onKeyDown = (event: KeyboardEvent) => applyKey(event, true);
    const onKeyUp = (event: KeyboardEvent) => applyKey(event, false);

    rafRef.current = requestAnimationFrame(loop);
    window.addEventListener("keydown", onKeyDown, { passive: false });
    window.addEventListener("keyup", onKeyUp);

    return () => {
      alive = false;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [config]);

  const handlePlay = useCallback(() => {
    setPlaying(true);
  }, []);

  return (
    <div className="relative h-screen overflow-hidden">
      <CombatView
        stateRef={stateRef as RefObject<CombatState>}
        barrierBreaks={barrierBreaks}
      />

      <DebugPanel debug={debug} />

      {/* Speech-bubble taunt — only shown when text exists */}
      <TauntBubble text={tauntText} onDismiss={() => setTauntText("")} />

      {/* Play button — shown until first interaction */}
      {!playing && (
        <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <button
            type="button"
            onClick={handlePlay}
            className="rounded-xl border border-emerald-400/60 bg-emerald-600 px-8 py-3 text-lg font-bold uppercase tracking-wider text-white shadow-lg shadow-emerald-900/40 transition hover:bg-emerald-500 hover:shadow-emerald-700/60 active:scale-95"
          >
            ▶ Play
          </button>
        </div>
      )}
    </div>
  );
};

export default App;
