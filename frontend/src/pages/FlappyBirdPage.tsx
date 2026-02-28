import { useCallback, useEffect, useRef, useState } from "react";
import { useBCIStream } from "@/hooks/useBCIStream";
import { EEGStreamModal } from "@/combat3d/ui/EEGStreamModal";

type GamePhase = "menu" | "running" | "paused" | "gameover";
type ControlMode = "manual" | "eeg";
type EegHemisphere = "left" | "right";

interface Pipe {
  id: number;
  x: number;
  gapCenterY: number;
  scored: boolean;
}

interface RuntimeState {
  birdY: number;
  birdVelocity: number;
  pipes: Pipe[];
  spawnAccumulatorMs: number;
  distance: number;
  score: number;
  nextPipeId: number;
}

interface CloudSeed {
  x: number;
  y: number;
  scale: number;
  speed: number;
  opacity: number;
}

interface EegPaneState {
  waveSamples: number[];
  leftPower: number;
  rightPower: number;
  confidence: number;
  packetRateHz: number;
  activeHemisphere: EegHemisphere;
}

const CANVAS_WIDTH = 960;
const CANVAS_HEIGHT = 540;
const GROUND_HEIGHT = 96;
const PLAYFIELD_HEIGHT = CANVAS_HEIGHT - GROUND_HEIGHT;
const BIRD_X = 220;
const BIRD_RADIUS = 20;
const GRAVITY = 1000;
const FLAP_VELOCITY = -465;
const MAX_FALL_SPEED = 820;
const PIPE_WIDTH = 108;
const BASE_PIPE_GAP = 184;
const MIN_PIPE_GAP = 132;
const PIPE_GAP_SHRINK_PER_SCORE = 2.35;
const BASE_PIPE_SPEED = 220;
const PIPE_SPEED_GAIN_PER_SCORE = 7.5;
const PIPE_SPEED_GAIN_CAP = 130;
const PIPE_SPAWN_INTERVAL_MS = 1420;
const PIPE_MARGIN_TOP = 70;
const PIPE_MARGIN_BOTTOM = 90;
const MAX_FRAME_DELTA_MS = 34;
const BEST_SCORE_STORAGE_KEY = "ratm.flappy.best-score";
const EEG_WAVE_POINTS = 42;
const EEG_WAVE_TICK_MS = 120;
const EEG_BLINK_DECAY_MS = 650;

const CLOUD_SEEDS: readonly CloudSeed[] = [
  { x: 120, y: 84, scale: 1.15, speed: 0.42, opacity: 0.72 },
  { x: 336, y: 140, scale: 0.84, speed: 0.56, opacity: 0.58 },
  { x: 612, y: 104, scale: 1.02, speed: 0.36, opacity: 0.66 },
  { x: 820, y: 170, scale: 0.92, speed: 0.48, opacity: 0.54 },
  { x: 1040, y: 110, scale: 1.25, speed: 0.33, opacity: 0.7 },
] as const;

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));
const clamp01 = (value: number): number => clamp(value, 0, 1);
const clampSignal = (value: number): number => clamp(value, -1, 1);

const wrap = (value: number, range: number): number => {
  const wrapped = value % range;
  return wrapped < 0 ? wrapped + range : wrapped;
};

const createDefaultWave = (phase = 0): number[] =>
  Array.from(
    { length: EEG_WAVE_POINTS },
    (_, index) => Math.sin(phase + (index / EEG_WAVE_POINTS) * Math.PI * 3.5) * 0.15,
  );

const createInitialEegPaneState = (): EegPaneState => ({
  waveSamples: createDefaultWave(),
  leftPower: 0.3,
  rightPower: 0.3,
  confidence: 0,
  packetRateHz: 0,
  activeHemisphere: "left",
});

const getPipeGap = (score: number): number =>
  Math.max(MIN_PIPE_GAP, BASE_PIPE_GAP - score * PIPE_GAP_SHRINK_PER_SCORE);

const getPipeSpeed = (score: number): number =>
  BASE_PIPE_SPEED + Math.min(score * PIPE_SPEED_GAIN_PER_SCORE, PIPE_SPEED_GAIN_CAP);

const loadBestScore = (): number => {
  if (typeof window === "undefined") {
    return 0;
  }
  try {
    const raw = window.localStorage.getItem(BEST_SCORE_STORAGE_KEY);
    if (!raw) {
      return 0;
    }
    const parsed = Number(raw);
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 0;
  } catch {
    return 0;
  }
};

const storeBestScore = (score: number): void => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    window.localStorage.setItem(BEST_SCORE_STORAGE_KEY, String(score));
  } catch {
    // Ignore storage failures in private modes.
  }
};

const createRuntimeState = (): RuntimeState => ({
  birdY: PLAYFIELD_HEIGHT * 0.46,
  birdVelocity: 0,
  pipes: [],
  spawnAccumulatorMs: 0,
  distance: 0,
  score: 0,
  nextPipeId: 1,
});

const spawnPipe = (runtime: RuntimeState, gapSize: number): void => {
  const minCenter = PIPE_MARGIN_TOP + gapSize / 2;
  const maxCenter = PLAYFIELD_HEIGHT - PIPE_MARGIN_BOTTOM - gapSize / 2;
  const gapCenterY =
    minCenter + Math.random() * Math.max(1, maxCenter - minCenter);
  runtime.pipes.push({
    id: runtime.nextPipeId,
    x: CANVAS_WIDTH + PIPE_WIDTH + 34,
    gapCenterY,
    scored: false,
  });
  runtime.nextPipeId += 1;
};

const drawCloud = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  scale: number,
  opacity: number,
) => {
  ctx.save();
  ctx.globalAlpha = opacity;
  ctx.fillStyle = "#ffffff";

  ctx.beginPath();
  ctx.arc(x, y, 28 * scale, 0, Math.PI * 2);
  ctx.arc(x + 30 * scale, y - 12 * scale, 24 * scale, 0, Math.PI * 2);
  ctx.arc(x + 58 * scale, y, 26 * scale, 0, Math.PI * 2);
  ctx.arc(x + 30 * scale, y + 8 * scale, 32 * scale, 0, Math.PI * 2);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
};

const drawBackground = (
  ctx: CanvasRenderingContext2D,
  distance: number,
  timeMs: number,
): void => {
  const sky = ctx.createLinearGradient(0, 0, 0, PLAYFIELD_HEIGHT);
  sky.addColorStop(0, "#7ad7ff");
  sky.addColorStop(0.5, "#9de9ff");
  sky.addColorStop(1, "#d4f7ff");
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, CANVAS_WIDTH, PLAYFIELD_HEIGHT);

  const sunX = CANVAS_WIDTH - 130;
  const sunY = 100;
  const sunGlow = ctx.createRadialGradient(sunX, sunY, 24, sunX, sunY, 118);
  sunGlow.addColorStop(0, "rgba(255, 247, 189, 0.95)");
  sunGlow.addColorStop(1, "rgba(255, 247, 189, 0)");
  ctx.fillStyle = sunGlow;
  ctx.beginPath();
  ctx.arc(sunX, sunY, 118, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ffe59d";
  ctx.beginPath();
  ctx.arc(sunX, sunY, 34, 0, Math.PI * 2);
  ctx.fill();

  for (const cloud of CLOUD_SEEDS) {
    const cloudX = wrap(
      cloud.x - distance * cloud.speed * 0.18,
      CANVAS_WIDTH + 260,
    ) - 120;
    drawCloud(ctx, cloudX, cloud.y, cloud.scale, cloud.opacity);
  }

  ctx.fillStyle = "#8ecf8b";
  ctx.beginPath();
  ctx.moveTo(0, PLAYFIELD_HEIGHT);
  for (let x = 0; x <= CANVAS_WIDTH; x += 14) {
    const y =
      PLAYFIELD_HEIGHT - 104 -
      Math.sin((x + distance * 0.18) / 100) * 24 -
      Math.sin((x + 48 + distance * 0.08) / 48) * 12;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(CANVAS_WIDTH, PLAYFIELD_HEIGHT);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#64b36f";
  ctx.beginPath();
  ctx.moveTo(0, PLAYFIELD_HEIGHT);
  for (let x = 0; x <= CANVAS_WIDTH; x += 12) {
    const y =
      PLAYFIELD_HEIGHT - 76 -
      Math.sin((x + distance * 0.32) / 72) * 18 -
      Math.sin((x + 38 + distance * 0.14) / 35) * 8;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(CANVAS_WIDTH, PLAYFIELD_HEIGHT);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "rgba(255, 255, 255, 0.2)";
  ctx.fillRect(0, PLAYFIELD_HEIGHT - 2, CANVAS_WIDTH, 2);

  const sparkOffset = (timeMs * 0.03) % 160;
  ctx.fillStyle = "rgba(255, 255, 255, 0.25)";
  for (let i = -1; i < 8; i += 1) {
    const x = i * 160 + sparkOffset;
    ctx.fillRect(x, 28 + ((i * 17) % 21), 2, 2);
    ctx.fillRect(x + 36, 54 + ((i * 13) % 16), 2, 2);
  }
};

const drawPipeSegment = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  invertCap: boolean,
) => {
  if (height <= 0) {
    return;
  }

  const body = ctx.createLinearGradient(x, y, x + width, y);
  body.addColorStop(0, "#2e9d57");
  body.addColorStop(0.45, "#54c770");
  body.addColorStop(1, "#2a8747");
  ctx.fillStyle = body;
  ctx.fillRect(x, y, width, height);

  ctx.fillStyle = "rgba(255, 255, 255, 0.22)";
  ctx.fillRect(x + 12, y + 10, 10, Math.max(0, height - 20));
  ctx.fillStyle = "rgba(19, 58, 28, 0.25)";
  ctx.fillRect(x + width - 12, y + 8, 8, Math.max(0, height - 16));

  const capHeight = 22;
  const capWidth = width + 16;
  const capX = x - 8;
  const capY = invertCap ? y + height - capHeight : y;
  const cap = ctx.createLinearGradient(capX, capY, capX + capWidth, capY);
  cap.addColorStop(0, "#4bc96f");
  cap.addColorStop(1, "#2c8f4d");
  ctx.fillStyle = cap;
  ctx.fillRect(capX, capY, capWidth, capHeight);

  ctx.fillStyle = "rgba(255, 255, 255, 0.28)";
  ctx.fillRect(capX + 12, capY + 3, capWidth - 28, 3);

  ctx.strokeStyle = "rgba(11, 33, 17, 0.35)";
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, width, height);
};

const drawPipes = (
  ctx: CanvasRenderingContext2D,
  pipes: readonly Pipe[],
  gapSize: number,
) => {
  for (const pipe of pipes) {
    const halfGap = gapSize / 2;
    const topHeight = pipe.gapCenterY - halfGap;
    const bottomStart = pipe.gapCenterY + halfGap;
    const bottomHeight = PLAYFIELD_HEIGHT - bottomStart;

    drawPipeSegment(ctx, pipe.x, 0, PIPE_WIDTH, topHeight, true);
    drawPipeSegment(ctx, pipe.x, bottomStart, PIPE_WIDTH, bottomHeight, false);
  }
};

const drawGround = (ctx: CanvasRenderingContext2D, distance: number): void => {
  const topStrip = ctx.createLinearGradient(
    0,
    PLAYFIELD_HEIGHT,
    0,
    PLAYFIELD_HEIGHT + 16,
  );
  topStrip.addColorStop(0, "#b0df63");
  topStrip.addColorStop(1, "#80bb38");
  ctx.fillStyle = topStrip;
  ctx.fillRect(0, PLAYFIELD_HEIGHT, CANVAS_WIDTH, 16);

  const soil = ctx.createLinearGradient(
    0,
    PLAYFIELD_HEIGHT + 16,
    0,
    CANVAS_HEIGHT,
  );
  soil.addColorStop(0, "#8a5e2f");
  soil.addColorStop(1, "#684123");
  ctx.fillStyle = soil;
  ctx.fillRect(0, PLAYFIELD_HEIGHT + 16, CANVAS_WIDTH, GROUND_HEIGHT - 16);

  const stripeOffset = -wrap(distance * 0.8, 52);
  for (let x = stripeOffset; x < CANVAS_WIDTH + 52; x += 52) {
    ctx.fillStyle = "#9f6d38";
    ctx.fillRect(x, PLAYFIELD_HEIGHT + 24, 30, 6);
    ctx.fillStyle = "#7a5028";
    ctx.fillRect(x + 10, PLAYFIELD_HEIGHT + 40, 18, 5);
  }

  ctx.fillStyle = "rgba(255, 255, 255, 0.22)";
  ctx.fillRect(0, PLAYFIELD_HEIGHT + 2, CANVAS_WIDTH, 2);
};

const drawBird = (
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  rotation: number,
  timeMs: number,
  eegPulse: boolean,
): void => {
  ctx.save();
  ctx.translate(x, y);
  ctx.rotate(rotation);

  if (eegPulse) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.26)";
    ctx.beginPath();
    ctx.arc(0, 0, BIRD_RADIUS + 12, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#d57b2f";
  ctx.beginPath();
  ctx.moveTo(-BIRD_RADIUS - 10, -2);
  ctx.lineTo(-BIRD_RADIUS - 24, -10);
  ctx.lineTo(-BIRD_RADIUS - 24, 8);
  ctx.closePath();
  ctx.fill();

  const body = ctx.createRadialGradient(-4, -8, 4, 2, 0, BIRD_RADIUS + 6);
  body.addColorStop(0, "#ffd966");
  body.addColorStop(1, "#f6b11d");
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(0, 0, BIRD_RADIUS, 0, Math.PI * 2);
  ctx.fill();

  const flapCycle = Math.sin(timeMs / 75);
  const wingRotation = -0.72 + flapCycle * 0.44;
  ctx.save();
  ctx.translate(-6, 6);
  ctx.rotate(wingRotation);
  ctx.fillStyle = "#f39e2f";
  ctx.beginPath();
  ctx.ellipse(0, 0, 14, 9, 0.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  ctx.fillStyle = "#ea7a1b";
  ctx.beginPath();
  ctx.moveTo(BIRD_RADIUS - 1, 1);
  ctx.lineTo(BIRD_RADIUS + 16, -2);
  ctx.lineTo(BIRD_RADIUS + 16, 6);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(6, -6, 5.4, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#17223b";
  ctx.beginPath();
  ctx.arc(7.6, -6, 2.3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "rgba(255, 255, 255, 0.65)";
  ctx.beginPath();
  ctx.arc(8.2, -7.2, 0.9, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "rgba(99, 63, 17, 0.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(0, 0, BIRD_RADIUS, 0, Math.PI * 2);
  ctx.stroke();

  ctx.restore();
};

const drawCenterPanel = (
  ctx: CanvasRenderingContext2D,
  title: string,
  subtitle: string,
) => {
  const panelWidth = 440;
  const panelHeight = 120;
  const panelX = (CANVAS_WIDTH - panelWidth) / 2;
  const panelY = (PLAYFIELD_HEIGHT - panelHeight) / 2;

  ctx.fillStyle = "rgba(5, 17, 34, 0.65)";
  ctx.fillRect(panelX, panelY, panelWidth, panelHeight);
  ctx.strokeStyle = "rgba(165, 219, 255, 0.5)";
  ctx.lineWidth = 2;
  ctx.strokeRect(panelX, panelY, panelWidth, panelHeight);

  ctx.fillStyle = "#f5fbff";
  ctx.textAlign = "center";
  ctx.font = "700 34px 'Trebuchet MS', 'Segoe UI', sans-serif";
  ctx.fillText(title, CANVAS_WIDTH / 2, panelY + 48);
  ctx.font = "500 19px 'Trebuchet MS', 'Segoe UI', sans-serif";
  ctx.fillStyle = "rgba(222, 242, 255, 0.95)";
  ctx.fillText(subtitle, CANVAS_WIDTH / 2, panelY + 82);
};

export default function FlappyBirdPage() {
  const { blink, isStreaming, executeStreamAction } = useBCIStream();

  const [phase, setPhase] = useState<GamePhase>("menu");
  const [controlMode, setControlMode] = useState<ControlMode>("manual");
  const [score, setScore] = useState(0);
  const [bestScore, setBestScore] = useState<number>(() => loadBestScore());
  const [eegPane, setEegPane] = useState<EegPaneState>(() => createInitialEegPaneState());

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const runtimeRef = useRef<RuntimeState>(createRuntimeState());
  const animationFrameRef = useRef<number | null>(null);
  const lastFrameMsRef = useRef<number | null>(null);
  const phaseRef = useRef<GamePhase>("menu");
  const modeRef = useRef<ControlMode>("manual");
  const scoreRef = useRef(0);
  const blinkCountRef = useRef(blink.blinkCount);
  const blinkPulseAtMsRef = useRef(0);
  const isStreamingRef = useRef(isStreaming);
  const eegLastBlinkAtMsRef = useRef<number | null>(null);
  const eegWavePhaseRef = useRef(0);
  const eegActiveHemisphereRef = useRef<EegHemisphere>("left");
  const eegBlinkRateWindowRef = useRef({
    windowStartMs: performance.now(),
    blinkCount: 0,
    packetRateHz: 0,
  });

  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  useEffect(() => {
    modeRef.current = controlMode;
  }, [controlMode]);

  useEffect(() => {
    isStreamingRef.current = isStreaming;
  }, [isStreaming]);

  const setPhaseState = useCallback((next: GamePhase) => {
    phaseRef.current = next;
    setPhase(next);
  }, []);

  const setScoreState = useCallback((next: number) => {
    if (scoreRef.current === next) {
      return;
    }
    scoreRef.current = next;
    runtimeRef.current.score = next;
    setScore(next);
  }, []);

  const pushEegSample = useCallback(
    (sample: number, patch: Partial<Omit<EegPaneState, "waveSamples">> = {}) => {
      setEegPane((current) => ({
        ...current,
        ...patch,
        waveSamples: [
          ...current.waveSamples.slice(-(EEG_WAVE_POINTS - 1)),
          clampSignal(sample),
        ],
      }));
    },
    [],
  );

  const finishRun = useCallback(() => {
    if (phaseRef.current === "gameover") {
      return;
    }
    const finalScore = scoreRef.current;
    setPhaseState("gameover");
    setBestScore((currentBest) => {
      if (finalScore <= currentBest) {
        return currentBest;
      }
      storeBestScore(finalScore);
      return finalScore;
    });
  }, [setPhaseState]);

  const flap = useCallback(() => {
    if (phaseRef.current !== "running") {
      return;
    }
    runtimeRef.current.birdVelocity = FLAP_VELOCITY;
    blinkPulseAtMsRef.current = performance.now();
  }, []);

  const beginRun = useCallback(
    (mode: ControlMode, initialFlap: boolean) => {
      runtimeRef.current = createRuntimeState();
      lastFrameMsRef.current = null;
      blinkCountRef.current = blink.blinkCount;
      blinkPulseAtMsRef.current = 0;
      scoreRef.current = 0;
      setScore(0);
      setControlMode(mode);
      modeRef.current = mode;
      setPhaseState("running");

      if (initialFlap) {
        runtimeRef.current.birdVelocity = FLAP_VELOCITY;
      }

      if (mode === "eeg" && !isStreamingRef.current) {
        void executeStreamAction("start");
      }
    },
    [blink.blinkCount, executeStreamAction, setPhaseState],
  );

  const togglePause = useCallback(() => {
    if (phaseRef.current === "running") {
      setPhaseState("paused");
      return;
    }
    if (phaseRef.current === "paused") {
      lastFrameMsRef.current = null;
      setPhaseState("running");
    }
  }, [setPhaseState]);

  const updateSimulation = useCallback(
    (deltaMs: number) => {
      if (phaseRef.current !== "running") {
        return;
      }

      const runtime = runtimeRef.current;
      const dt = Math.min(deltaMs, MAX_FRAME_DELTA_MS) / 1000;
      const pipeSpeed = getPipeSpeed(scoreRef.current);
      const gapSize = getPipeGap(scoreRef.current);

      runtime.spawnAccumulatorMs += deltaMs;
      runtime.distance += pipeSpeed * dt;
      runtime.birdVelocity = Math.min(
        MAX_FALL_SPEED,
        runtime.birdVelocity + GRAVITY * dt,
      );
      runtime.birdY += runtime.birdVelocity * dt;

      while (runtime.spawnAccumulatorMs >= PIPE_SPAWN_INTERVAL_MS) {
        runtime.spawnAccumulatorMs -= PIPE_SPAWN_INTERVAL_MS;
        spawnPipe(runtime, gapSize);
      }

      const birdTop = runtime.birdY - BIRD_RADIUS;
      const birdBottom = runtime.birdY + BIRD_RADIUS;
      let crashed = birdTop <= 0 || birdBottom >= PLAYFIELD_HEIGHT;

      for (const pipe of runtime.pipes) {
        pipe.x -= pipeSpeed * dt;

        if (!pipe.scored && pipe.x + PIPE_WIDTH < BIRD_X - BIRD_RADIUS) {
          pipe.scored = true;
          setScoreState(scoreRef.current + 1);
        }

        const overlapsX =
          BIRD_X + BIRD_RADIUS > pipe.x && BIRD_X - BIRD_RADIUS < pipe.x + PIPE_WIDTH;
        if (!overlapsX) {
          continue;
        }

        const halfGap = gapSize / 2;
        const topPipeBottom = pipe.gapCenterY - halfGap;
        const bottomPipeTop = pipe.gapCenterY + halfGap;
        if (birdTop < topPipeBottom || birdBottom > bottomPipeTop) {
          crashed = true;
        }
      }

      runtime.pipes = runtime.pipes.filter((pipe) => pipe.x + PIPE_WIDTH > -26);

      if (crashed) {
        finishRun();
      }
    },
    [finishRun, setScoreState],
  );

  const drawFrame = useCallback((ctx: CanvasRenderingContext2D, nowMs: number) => {
    const runtime = runtimeRef.current;
    const activePhase = phaseRef.current;
    const activeMode = modeRef.current;

    drawBackground(ctx, runtime.distance, nowMs);
    drawPipes(ctx, runtime.pipes, getPipeGap(scoreRef.current));
    drawGround(ctx, runtime.distance);

    const idleYOffset = Math.sin(nowMs / 260) * 8;
    const birdY =
      activePhase === "running" || activePhase === "paused" || activePhase === "gameover"
        ? runtime.birdY
        : PLAYFIELD_HEIGHT * 0.46 + idleYOffset;
    const birdVelocity =
      activePhase === "running" ? runtime.birdVelocity : Math.sin(nowMs / 180) * 140;
    const rotation = clamp(birdVelocity / 620, -0.78, 1.12);
    const eegPulse = activeMode === "eeg" && nowMs - blinkPulseAtMsRef.current < 120;
    drawBird(ctx, BIRD_X, birdY, rotation, nowMs, eegPulse);

    if (activePhase === "menu") {
      drawCenterPanel(ctx, "Flappy Brain", "Space = Manual | E = EEG Blink");
    } else if (activePhase === "paused") {
      drawCenterPanel(ctx, "Paused", "Press P to continue");
    } else if (activePhase === "gameover") {
      drawCenterPanel(ctx, "Crash!", "Press R to retry");
    }
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    const renderLoop = (timestampMs: number) => {
      if (lastFrameMsRef.current === null) {
        lastFrameMsRef.current = timestampMs;
      }
      const deltaMs = timestampMs - lastFrameMsRef.current;
      lastFrameMsRef.current = timestampMs;

      updateSimulation(deltaMs);
      drawFrame(ctx, timestampMs);

      animationFrameRef.current = window.requestAnimationFrame(renderLoop);
    };

    animationFrameRef.current = window.requestAnimationFrame(renderLoop);
    return () => {
      if (animationFrameRef.current !== null) {
        window.cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [drawFrame, updateSimulation]);

  useEffect(() => {
    const previousBlinkCount = blinkCountRef.current;
    blinkCountRef.current = blink.blinkCount;
    const blinkDelta = blink.blinkCount - previousBlinkCount;
    if (blinkDelta <= 0) {
      return;
    }

    const now = performance.now();
    eegLastBlinkAtMsRef.current = now;
    blinkPulseAtMsRef.current = now;
    eegActiveHemisphereRef.current =
      blink.blinkCount % 2 === 0 ? "right" : "left";
    const activeHemisphere = eegActiveHemisphereRef.current;

    const blinkRateWindow = eegBlinkRateWindowRef.current;
    blinkRateWindow.blinkCount += blinkDelta;
    const elapsed = now - blinkRateWindow.windowStartMs;
    if (elapsed >= 1000) {
      blinkRateWindow.packetRateHz = (blinkRateWindow.blinkCount * 1000) / elapsed;
      blinkRateWindow.blinkCount = 0;
      blinkRateWindow.windowStartMs = now;
    }

    const direction = activeHemisphere === "right" ? 1 : -1;
    pushEegSample(direction * 0.88, {
      leftPower: activeHemisphere === "left" ? 0.93 : 0.4,
      rightPower: activeHemisphere === "right" ? 0.93 : 0.4,
      confidence: 1,
      packetRateHz: blinkRateWindow.packetRateHz,
      activeHemisphere,
    });

    if (modeRef.current !== "eeg" || phaseRef.current !== "running") {
      return;
    }
    flap();
  }, [blink.blinkCount, flap, pushEegSample]);

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      const now = performance.now();
      const blinkRateWindow = eegBlinkRateWindowRef.current;
      const elapsed = now - blinkRateWindow.windowStartMs;
      if (elapsed >= 1000) {
        blinkRateWindow.packetRateHz = (blinkRateWindow.blinkCount * 1000) / elapsed;
        blinkRateWindow.blinkCount = 0;
        blinkRateWindow.windowStartMs = now;
      }

      const lastBlinkAt = eegLastBlinkAtMsRef.current;
      const pulseStrength =
        lastBlinkAt === null
          ? 0
          : clamp01(1 - (now - lastBlinkAt) / EEG_BLINK_DECAY_MS);
      eegWavePhaseRef.current += 0.34;
      const phase = eegWavePhaseRef.current;
      const activeHemisphere = eegActiveHemisphereRef.current;
      const direction = activeHemisphere === "right" ? 1 : -1;
      const sample = clampSignal(
        direction * pulseStrength * 0.42 +
          Math.sin(phase) * (0.14 + pulseStrength * 0.28) +
          Math.sin(phase * 0.47) * 0.09,
      );

      pushEegSample(sample, {
        leftPower:
          activeHemisphere === "left"
            ? 0.34 + pulseStrength * 0.58
            : 0.22 + pulseStrength * 0.12,
        rightPower:
          activeHemisphere === "right"
            ? 0.34 + pulseStrength * 0.58
            : 0.22 + pulseStrength * 0.12,
        confidence: pulseStrength,
        packetRateHz: blinkRateWindow.packetRateHz,
        activeHemisphere,
      });
    }, EEG_WAVE_TICK_MS);

    return () => window.clearInterval(intervalId);
  }, [pushEegSample]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();

      if (event.code === "Space") {
        event.preventDefault();
      }

      if (key === "escape") {
        setPhaseState("menu");
        return;
      }

      if (phaseRef.current === "menu") {
        if (event.code === "Space") {
          beginRun("manual", true);
        } else if (key === "e") {
          beginRun("eeg", false);
        }
        return;
      }

      if (phaseRef.current === "running") {
        if (key === "p") {
          togglePause();
          return;
        }
        if (key === "r") {
          beginRun(modeRef.current, modeRef.current === "manual");
          return;
        }
        if (event.code === "Space" && modeRef.current === "manual") {
          flap();
        }
        return;
      }

      if (phaseRef.current === "paused") {
        if (key === "p") {
          togglePause();
          return;
        }
        if (key === "r") {
          beginRun(modeRef.current, modeRef.current === "manual");
          return;
        }
        if (event.code === "Space" && modeRef.current === "manual") {
          lastFrameMsRef.current = null;
          setPhaseState("running");
          flap();
        }
        return;
      }

      if (phaseRef.current === "gameover") {
        if (key === "r") {
          beginRun(modeRef.current, modeRef.current === "manual");
          return;
        }
        if (event.code === "Space") {
          beginRun("manual", true);
          return;
        }
        if (key === "e") {
          beginRun("eeg", false);
        }
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [beginRun, flap, setPhaseState, togglePause]);

  const modeLabel = controlMode === "manual" ? "Manual (Space)" : "EEG Blink";
  const modeBadgeClass =
    controlMode === "manual"
      ? "border-amber-200 bg-amber-50 text-amber-700"
      : "border-sky-200 bg-sky-50 text-sky-700";

  return (
    <div className="min-h-[calc(100vh-4rem)] bg-linear-to-b from-cyan-100 via-sky-100 to-emerald-100 p-4 sm:p-6">
      <div className="mx-auto max-w-6xl">
        <header className="mb-4 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700/80">
              Game 4
            </p>
            <h1 className="text-3xl font-extrabold text-slate-900 sm:text-4xl">
              Flappy Brain
            </h1>
            <p className="mt-1 text-sm text-slate-600 sm:text-base">
              Blink in EEG mode to flap, or play manually with Space.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2 text-xs font-semibold sm:text-sm">
            <span className={`rounded-full border px-3 py-1 ${modeBadgeClass}`}>{modeLabel}</span>
            <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-zinc-700">
              Score {score}
            </span>
            <span className="rounded-full border border-zinc-200 bg-white px-3 py-1 text-zinc-700">
              Best {bestScore}
            </span>
            <span
              className={`rounded-full border px-3 py-1 ${
                isStreaming
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : "border-zinc-200 bg-zinc-50 text-zinc-500"
              }`}
            >
              EEG {isStreaming ? "Live" : "Idle"}
            </span>
            <span
              className={`rounded-full border px-3 py-1 ${
                blink.detected
                  ? "border-sky-300 bg-sky-100 text-sky-700"
                  : "border-zinc-200 bg-zinc-50 text-zinc-500"
              }`}
            >
              Blinks {blink.blinkCount}
            </span>
          </div>
        </header>

        <div className="relative overflow-hidden rounded-3xl border border-sky-200 bg-slate-900 shadow-[0_24px_60px_rgba(14,47,76,0.3)]">
          <div className="aspect-[16/9] w-full">
            <canvas
              ref={canvasRef}
              width={CANVAS_WIDTH}
              height={CANVAS_HEIGHT}
              className="h-full w-full"
            />
          </div>

          <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center justify-between p-3 sm:p-4">
            <div className="rounded-lg border border-white/25 bg-slate-900/55 px-3 py-2 text-[11px] font-semibold text-white sm:text-xs">
              {controlMode === "manual"
                ? "Manual: press Space to flap"
                : isStreaming
                  ? "EEG: blink to flap"
                  : "EEG: stream idle, waiting for blinks"}
            </div>
            <div className="rounded-lg border border-white/25 bg-slate-900/55 px-3 py-2 text-[11px] font-semibold text-white sm:text-xs">
              {phase === "running"
                ? "P to pause"
                : phase === "paused"
                  ? "Paused"
                  : phase === "gameover"
                    ? "R to restart"
                : "Esc returns to menu"}
            </div>
          </div>

          <EEGStreamModal
            waveSamples={eegPane.waveSamples}
            leftPower={eegPane.leftPower}
            rightPower={eegPane.rightPower}
            confidence={eegPane.confidence}
            packetRateHz={eegPane.packetRateHz}
            activeHemisphere={eegPane.activeHemisphere}
            mode="features"
            positionClassName="bottom-3 left-3 sm:bottom-4 sm:left-4"
          />

          <div className="absolute bottom-3 right-3 z-20 flex flex-wrap gap-2 sm:bottom-4 sm:right-4">
            <button
              type="button"
              onClick={() => beginRun("manual", true)}
              className="rounded-md border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-white/30"
            >
              Manual
            </button>
            <button
              type="button"
              onClick={() => beginRun("eeg", false)}
              className="rounded-md border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-white/30"
            >
              EEG Blink
            </button>
            <button
              type="button"
              onClick={togglePause}
              disabled={phase === "menu" || phase === "gameover"}
              className="rounded-md border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-white/30 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {phase === "paused" ? "Resume" : "Pause"}
            </button>
            <button
              type="button"
              onClick={() => setPhaseState("menu")}
              className="rounded-md border border-white/35 bg-white/20 px-3 py-1.5 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-white/30"
            >
              Menu
            </button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 text-sm sm:grid-cols-3">
          <div className="rounded-xl border border-sky-200 bg-white/80 p-3 text-slate-700 shadow-sm">
            <p className="font-semibold text-slate-900">Manual Controls</p>
            <p className="mt-1">Press `Space` to flap and `P` to pause.</p>
          </div>
          <div className="rounded-xl border border-sky-200 bg-white/80 p-3 text-slate-700 shadow-sm">
            <p className="font-semibold text-slate-900">EEG Blink Mode</p>
            <p className="mt-1">
              Every detected blink triggers one jump. Use the global stream button if EEG is idle.
            </p>
          </div>
          <div className="rounded-xl border border-sky-200 bg-white/80 p-3 text-slate-700 shadow-sm">
            <p className="font-semibold text-slate-900">Restart Flow</p>
            <p className="mt-1">Press `R` after a crash or `Esc` to return to menu.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
