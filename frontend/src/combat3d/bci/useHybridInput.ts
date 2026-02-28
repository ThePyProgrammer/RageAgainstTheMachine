import { useCallback, useEffect, useRef, useState } from "react";
import {
  useBCIDiscreteInput,
  type BCIDiscreteState,
  type Combat3DControlMode,
} from "./useBCIDiscreteInput";
import type { InputSample } from "../engine/types";

/* ─── Keyboard state ──────────────────────────────────────────────── */

interface KeyState {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  fire: boolean;
}

const createKeyState = (): KeyState => ({
  up: false,
  down: false,
  left: false,
  right: false,
  fire: false,
});

/* ─── Tuning constants ────────────────────────────────────────────── */

/** Turn magnitude for keyboard in manual mode (−1..1) */
const KB_TURN = 0.8;
/** Turn magnitude for BCI discrete left/right */
const BCI_TURN = 0.85;

/* ─── Debug payload ───────────────────────────────────────────────── */

export interface HybridInputDebug {
  mode: Combat3DControlMode;
  bci: BCIDiscreteState;
  keysHeld: string[];
  lastSample: InputSample | null;
}

/* ─── Hook ────────────────────────────────────────────────────────── */

/**
 * Unified input system for Combat3D.
 *
 * ## Manual mode
 * All controls from keyboard:
 * - Arrow keys / WASD: throttle + turn
 * - Space: fire
 *
 * ## BCI Hybrid mode
 * - **Rotation** comes from BCI left/right (discrete websocket signals)
 * - **Movement** (forward/backward) from keyboard W/S or ↑/↓
 * - **Shoot** from BCI blink signal
 */
export function useHybridInput(mode: Combat3DControlMode) {
  const bciEnabled = mode === "bci_hybrid";
  const { state: bciState, consumeShoot: consumeBCIShoot } =
    useBCIDiscreteInput(bciEnabled);

  const keyStateRef = useRef<KeyState>(createKeyState());
  const [keysHeld, setKeysHeld] = useState<string[]>([]);
  const lastSampleRef = useRef<InputSample | null>(null);

  // ── Keyboard listeners ──
  useEffect(() => {
    const TRACKED = new Set([
      "ArrowUp",
      "ArrowDown",
      "ArrowLeft",
      "ArrowRight",
      "w",
      "W",
      "a",
      "A",
      "s",
      "S",
      "d",
      "D",
      " ",
    ]);

    const apply = (e: KeyboardEvent, value: boolean) => {
      if (!TRACKED.has(e.key)) return;
      e.preventDefault();
      const keys = keyStateRef.current;
      const k = e.key.toLowerCase();
      if (k === "arrowup" || k === "w") keys.up = value;
      else if (k === "arrowdown" || k === "s") keys.down = value;
      else if (k === "arrowleft" || k === "a") keys.left = value;
      else if (k === "arrowright" || k === "d") keys.right = value;
      else if (k === " ") keys.fire = value;

      // publish for debug
      const held: string[] = [];
      if (keys.up) held.push("up");
      if (keys.down) held.push("down");
      if (keys.left) held.push("left");
      if (keys.right) held.push("right");
      if (keys.fire) held.push("fire");
      setKeysHeld(held);
    };

    const onDown = (e: KeyboardEvent) => apply(e, true);
    const onUp = (e: KeyboardEvent) => apply(e, false);

    window.addEventListener("keydown", onDown, { passive: false });
    window.addEventListener("keyup", onUp);
    return () => {
      window.removeEventListener("keydown", onDown);
      window.removeEventListener("keyup", onUp);
    };
  }, []);

  /**
   * Sample the merged input at the given sim time.
   * Call once per physics tick inside the game loop.
   *
   * Returns an `InputSample` that is directly push-able into `InputQueue`.
   */
  const sample = useCallback(
    (simTimeMs: number): InputSample => {
      const keys = keyStateRef.current;
      let throttle = 0;
      let turn = 0;
      let fire = false;

      if (mode === "manual") {
        // ── Full keyboard ──
        if (keys.up) throttle += 1;
        if (keys.down) throttle -= 1;
        if (keys.left) turn -= KB_TURN;
        if (keys.right) turn += KB_TURN;
        fire = keys.fire;
      } else {
        // ── BCI Hybrid ──
        // Rotation from BCI discrete signals
        if (bciState.rotation === "ccw") turn = -BCI_TURN;
        else if (bciState.rotation === "cw") turn = BCI_TURN;

        // Movement from keyboard only (W/S or ↑/↓)
        if (keys.up) throttle += 1;
        if (keys.down) throttle -= 1;

        // Shoot from BCI blink
        if (bciState.shoot) {
          fire = true;
          consumeBCIShoot();
        }
      }

      const s: InputSample = { timestamp: simTimeMs, throttle, turn, fire };
      lastSampleRef.current = s;
      return s;
    },
    [mode, bciState, consumeBCIShoot],
  );

  /** Whether there is any active input worth pushing */
  const hasActiveInput = useCallback((): boolean => {
    const keys = keyStateRef.current;
    if (mode === "manual") {
      return keys.up || keys.down || keys.left || keys.right || keys.fire;
    }
    // BCI hybrid: keyboard movement OR any BCI signal
    return (
      keys.up ||
      keys.down ||
      bciState.rotation !== null ||
      bciState.shoot
    );
  }, [mode, bciState]);

  /** Provides the raw KeyState ref for the game loop to read directly */
  const keyStateHandle = keyStateRef;

  const debug: HybridInputDebug = {
    mode,
    bci: bciState,
    keysHeld,
    lastSample: lastSampleRef.current,
  };

  return { sample, hasActiveInput, keyStateHandle, debug };
}
