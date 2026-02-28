# UI_COMPONENTS_TAILWIND.md

React UI components using Tailwind CSS.

Location: `/apps/web/src/components/`

## Component List

1. **CalibrationWizard** - Step-by-step calibration UI
2. **GameCanvas** - Canvas rendering (see GAME_LOOP_AND_RENDERING.md)
3. **ScoreBoard** - Display scores
4. **StressMeter** - Visual stress indicator
5. **TauntBubble** - Speech bubble overlay
6. **MenuScreen** - Start/settings menu
7. **DebugOverlay** - Dev info (latency, FPS, etc.)

## 1. CalibrationWizard

File: `/apps/web/src/components/CalibrationWizard.tsx`

```typescript
import React from 'react';

interface CalibrationWizardProps {
  step: 'baseline' | 'left' | 'right' | 'complete';
  trial: number;
  instruction: string;
  progress: number; // 0.0 - 1.0
  quality?: number;
  onRetry?: () => void;
  onContinue?: () => void;
}

export const CalibrationWizard: React.FC<CalibrationWizardProps> = ({
  step,
  trial,
  instruction,
  progress,
  quality,
  onRetry,
  onContinue,
}) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 text-white p-8">
      {/* Progress bar */}
      <div className="w-full max-w-md mb-8">
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-blue-500 transition-all duration-300"
            style={{ width: `${progress * 100}%` }}
          />
        </div>
      </div>

      {/* Instruction */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold mb-4">
          {step === 'baseline' && 'Baseline Calibration'}
          {step === 'left' && `Think UP (Trial ${trial})`}
          {step === 'right' && `Think DOWN (Trial ${trial})`}
          {step === 'complete' && 'Calibration Complete!'}
        </h2>
        <p className="text-xl text-gray-300">{instruction}</p>
      </div>

      {/* Visual cue */}
      {step !== 'complete' && (
        <div className="mb-8">
          {step === 'baseline' && (
            <div className="w-24 h-24 bg-gray-600 rounded-full animate-pulse" />
          )}
          {step === 'left' && (
            <div className="text-6xl animate-bounce">↑</div>
          )}
          {step === 'right' && (
            <div className="text-6xl animate-bounce">↓</div>
          )}
        </div>
      )}

      {/* Quality result */}
      {step === 'complete' && quality !== undefined && (
        <div className="mb-8">
          {quality >= 1.5 ? (
            <div className="text-green-500 text-6xl">✓</div>
          ) : quality >= 1.0 ? (
            <div className="text-yellow-500 text-6xl">⚠</div>
          ) : (
            <div className="text-red-500 text-6xl">✗</div>
          )}
          <p className="text-lg mt-4">
            Signal Quality: {quality.toFixed(2)}
            {quality >= 1.5 && ' (Excellent)'}
            {quality >= 1.0 && quality < 1.5 && ' (Marginal - Retry Suggested)'}
            {quality < 1.0 && ' (Poor - Use Keyboard)'}
          </p>
        </div>
      )}

      {/* Actions */}
      {step === 'complete' && (
        <div className="flex gap-4">
          {quality && quality < 1.5 && (
            <button
              onClick={onRetry}
              className="px-6 py-3 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-semibold transition"
            >
              Retry Calibration
            </button>
          )}
          <button
            onClick={onContinue}
            className="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold transition"
          >
            {quality && quality >= 1.0 ? 'Start Game' : 'Use Keyboard'}
          </button>
        </div>
      )}
    </div>
  );
};
```

## 2. ScoreBoard

File: `/apps/web/src/components/ScoreBoard.tsx`

```typescript
import React from 'react';

interface ScoreBoardProps {
  playerScore: number;
  aiScore: number;
}

export const ScoreBoard: React.FC<ScoreBoardProps> = ({ playerScore, aiScore }) => {
  return (
    <div className="fixed top-4 left-1/2 -translate-x-1/2 flex gap-8 text-white text-4xl font-mono">
      <div className="flex items-center gap-2">
        <span className="text-gray-400">You</span>
        <span className="font-bold">{playerScore}</span>
      </div>
      <div className="text-gray-600">|</div>
      <div className="flex items-center gap-2">
        <span className="font-bold">{aiScore}</span>
        <span className="text-gray-400">AI</span>
      </div>
    </div>
  );
};
```

## 3. StressMeter

File: `/apps/web/src/components/StressMeter.tsx`

```typescript
import React from 'react';

interface StressMeterProps {
  stressLevel: number; // 0.0 - 1.0
}

export const StressMeter: React.FC<StressMeterProps> = ({ stressLevel }) => {
  const percentage = Math.round(stressLevel * 100);
  const color =
    stressLevel > 0.7 ? 'bg-red-500' : stressLevel > 0.4 ? 'bg-yellow-500' : 'bg-green-500';

  return (
    <div className="fixed top-4 right-4 bg-black/80 text-white p-4 rounded-lg shadow-xl min-w-[150px]">
      <div className="text-sm text-gray-400 mb-2">Stress</div>
      <div className="text-2xl font-bold mb-2">{percentage}%</div>
      <div className="w-full h-3 bg-gray-700 rounded-full overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
};
```

## 4. TauntBubble

File: `/apps/web/src/components/TauntBubble.tsx`

```typescript
import React, { useEffect, useState } from 'react';

interface TauntBubbleProps {
  text: string;
  durationMs: number;
  timestamp: number;
  onExpire: () => void;
}

export const TauntBubble: React.FC<TauntBubbleProps> = ({
  text,
  durationMs,
  timestamp,
  onExpire,
}) => {
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    const elapsed = Date.now() - timestamp;
    const remaining = durationMs - elapsed;

    if (remaining <= 0) {
      onExpire();
      return;
    }

    // Start fading at 80% of duration
    const fadeStart = durationMs * 0.8;
    if (elapsed >= fadeStart) {
      const fadeProgress = (elapsed - fadeStart) / (durationMs - fadeStart);
      setOpacity(1 - fadeProgress);
    }

    const timer = setTimeout(onExpire, remaining);
    return () => clearTimeout(timer);
  }, [timestamp, durationMs, onExpire]);

  return (
    <div
      className="fixed top-20 left-1/2 -translate-x-1/2 bg-gray-900 border-2 border-red-500 rounded-lg px-6 py-3 shadow-2xl animate-bounce"
      style={{ opacity }}
    >
      <p className="text-white font-bold text-xl whitespace-nowrap">{text}</p>
      <div className="absolute -bottom-2 left-1/2 -translate-x-1/2 w-4 h-4 bg-gray-900 border-r-2 border-b-2 border-red-500 rotate-45" />
    </div>
  );
};
```

## 5. MenuScreen

File: `/apps/web/src/components/MenuScreen.tsx`

```typescript
import React from 'react';

interface MenuScreenProps {
  onStartCalibration: () => void;
  onStartKeyboard: () => void;
  hasCalibration: boolean;
}

export const MenuScreen: React.FC<MenuScreenProps> = ({
  onStartCalibration,
  onStartKeyboard,
  hasCalibration,
}) => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
      <h1 className="text-6xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-red-500 to-pink-500">
        RAGE AGAINST THE MACHINE
      </h1>
      <p className="text-xl text-gray-400 mb-12">Brain-Controlled Pong</p>

      <div className="flex flex-col gap-4 w-80">
        <button
          onClick={onStartCalibration}
          className="w-full px-8 py-4 bg-red-600 hover:bg-red-700 rounded-lg font-bold text-lg transition transform hover:scale-105"
        >
          {hasCalibration ? 'Re-Calibrate EEG' : 'Start with EEG'}
        </button>

        {hasCalibration && (
          <button
            onClick={onStartKeyboard}
            className="w-full px-8 py-4 bg-blue-600 hover:bg-blue-700 rounded-lg font-bold text-lg transition transform hover:scale-105"
          >
            Play with Saved Calibration
          </button>
        )}

        <button
          onClick={onStartKeyboard}
          className="w-full px-8 py-4 bg-gray-600 hover:bg-gray-700 rounded-lg font-semibold transition"
        >
          Play with Keyboard
        </button>
      </div>

      <p className="mt-12 text-sm text-gray-500 max-w-md text-center">
        Stress metric is for entertainment purposes only and not a medical device.
      </p>
    </div>
  );
};
```

## 6. DebugOverlay

File: `/apps/web/src/components/DebugOverlay.tsx`

```typescript
import React from 'react';

interface DebugOverlayProps {
  fps: number;
  latency: number;
  thonkConnected: boolean;
  calibrationQuality?: number;
}

export const DebugOverlay: React.FC<DebugOverlayProps> = ({
  fps,
  latency,
  thonkConnected,
  calibrationQuality,
}) => {
  return (
    <div className="fixed bottom-4 left-4 bg-black/90 text-green-400 p-3 rounded font-mono text-xs">
      <div>FPS: {fps.toFixed(1)}</div>
      <div>Latency: {latency.toFixed(0)}ms</div>
      <div>
        Thonk: {thonkConnected ? <span className="text-green-500">●</span> : <span className="text-red-500">●</span>}
      </div>
      {calibrationQuality && <div>Cal Quality: {calibrationQuality.toFixed(2)}</div>}
    </div>
  );
};
```

## Tailwind Config

File: `/apps/web/tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      animation: {
        'bounce-slow': 'bounce 2s infinite',
      },
    },
  },
  plugins: [],
};
```

## Styling Guidelines

- **Colors**: Dark theme (gray-900 background, white/gray text, red/pink accents)
- **Fonts**: System mono for scores/debug, sans-serif for UI
- **Animations**: Use Tailwind's built-in (bounce, pulse, transition)
- **Responsiveness**: Fixed canvas size, UI scales with viewport
- **Accessibility**: High contrast, large touch targets (min 44px)
