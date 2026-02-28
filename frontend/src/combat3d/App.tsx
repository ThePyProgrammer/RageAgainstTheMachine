import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { createOpponentInput } from "./ai/difficulty";
import { CombatView } from "./render/CombatView";
import { ControlPipeline, createDefaultControlPipeline } from "./bci/controlPipeline";
import {
  FRAME_MS,
  createCombatConfig,
  createStateFromSeed,
  stepSimulation,
} from "./engine";
import { InputQueue } from "./engine/inputQueue";
import { SeededRNG } from "./engine/seededRng";
import { HUD } from "./ui/HUD";
import { SessionControls } from "./ui/SessionControls";
import { DebugPanel } from "./debug/DebugPanel";
import type { BCIStreamPacket, BCIMode } from "@ragemachine/bci-shared";
import { createCombat3DSocket } from "./net/socket";
import { type JoinResponse, type TauntPacket, type TelemetryPacket } from "./net/contracts";
import type { CombatState, InputSample } from "./engine/types";

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

const createFallbackInput = (nowMs: number, event: KeyboardEvent): InputSample => {
  if (event.key === "ArrowUp") {
    return { timestamp: nowMs, throttle: 1, turn: 0, fire: false };
  }

  if (event.key === "ArrowDown") {
    return { timestamp: nowMs, throttle: -1, turn: 0, fire: false };
  }

  if (event.key === "ArrowLeft") {
    return { timestamp: nowMs, throttle: 0, turn: -0.8, fire: false };
  }

  if (event.key === "ArrowRight") {
    return { timestamp: nowMs, throttle: 0, turn: 0.8, fire: false };
  }

  if (event.key === " ") {
    return { timestamp: nowMs, throttle: 0, turn: 0, fire: true };
  }

  return { timestamp: nowMs, throttle: 0, turn: 0, fire: false };
};

export const App = () => {
  const [status, setStatus] = useState("disconnected");
  const [connected, setConnected] = useState(false);
  const [tauntText, setTauntText] = useState("server taunts pending");
  const [hud, setHud] = useState<CombatState | null>(null);
  const [debug, setDebug] = useState({ queueLen: 0, rttMs: 0 });

  const sessionId = useMemo(() => crypto.randomUUID(), []);
  const stateRef = useRef<CombatState>(createStateFromSeed(0x2f2f));
  const config = useMemo(() => createCombatConfig(0x2f2f), []);
  const playerQueueRef = useRef(new InputQueue());
  const enemyQueueRef = useRef(new InputQueue());
  const rngRef = useRef(new SeededRNG(config.seed));
  const pipelineRef = useRef(createDefaultControlPipeline());
  const socketRef = useRef<ReturnType<typeof createCombat3DSocket> | null>(null);

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

        setDebug((current) => ({ queueLen: playerQueueRef.current.length + enemyQueueRef.current.length, rttMs: current.rttMs }));
      },
      onSession: (_payload: JoinResponse) => {
        setConnected(true);
        setStatus("connected");
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
          },
          rngRef.current,
        );

        stateRef.current = result.state;

        if (stateRef.current.tick % 6 === 0) {
          setHud(stateRef.current);
          setDebug((current) => ({
            queueLen: playerQueueRef.current.length + enemyQueueRef.current.length,
            rttMs: current.rttMs,
          }));
        }

        accumulatorMsRef.current -= FRAME_MS;
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const now = performance.now();
      const input = createFallbackInput(now, event);
      playerQueueRef.current.push(input);
    };

    rafRef.current = requestAnimationFrame(loop);
    window.addEventListener("keydown", onKeyDown);

    return () => {
      alive = false;
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
      }
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [config]);

  const handleStart = (nextMode: BCIMode): void => {
    pipelineRef.current.updateMode(nextMode);
    setStatus("connecting");
    socketRef.current?.connect(sessionId, nextMode);
  };

  return (
    <div className="relative h-screen overflow-hidden">
      <CombatView stateRef={stateRef as RefObject<CombatState>} />
      <HUD state={hud} status={status} taunt={tauntText} />
      <DebugPanel debug={debug} />
      <SessionControls onStart={handleStart} disabled={connected} />
      <div className="absolute right-3 bottom-3 z-10 text-xs text-cyan-200">
        Arrow keys are manual fallback.
      </div>
    </div>
  );
};

export default App;
