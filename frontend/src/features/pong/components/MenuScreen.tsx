type MenuScreenProps = {
  hasSavedCalibration: boolean;
  onStartKeyboard: () => void;
  onStartBallMode: () => void;
  onStartEEG: () => void;
  onStartWithSavedCalibration: () => void;
  onCustomize: () => void;
};

export const MenuScreen = ({
  hasSavedCalibration,
  onStartKeyboard,
  onStartBallMode,
  onStartEEG,
  onStartWithSavedCalibration,
  onCustomize,
}: MenuScreenProps) => {
  return (
    <section className="min-h-[calc(100vh-4rem)] bg-neutral-950 text-white flex flex-col items-center justify-center gap-6 p-6">
      <div className="text-center">
        <h1 className="text-4xl font-bold">Rage Against The Machine</h1>
        <p className="text-zinc-400 mt-2">Pong demo mode</p>
      </div>

      <div className="w-full max-w-sm space-y-3">
        <button
          type="button"
          onClick={onStartEEG}
          className="w-full rounded bg-emerald-500 hover:bg-emerald-600 py-3 font-semibold"
        >
          E Start with EEG
        </button>
        <button
          type="button"
          onClick={onStartKeyboard}
          className="w-full rounded bg-blue-600 hover:bg-blue-700 py-3 font-semibold"
        >
          K Play with Keyboard Paddle
        </button>
        <button
          type="button"
          onClick={onStartBallMode}
          className="w-full rounded bg-violet-600 hover:bg-violet-700 py-3 font-semibold"
        >
          B Play with Ball Controls (WASD/Trackpad)
        </button>
        {hasSavedCalibration && (
          <button
            type="button"
            onClick={onStartWithSavedCalibration}
            className="w-full rounded border border-zinc-500 hover:border-zinc-300 py-3"
          >
            Play with Saved Calibration
          </button>
        )}
        <button
          type="button"
          onClick={onCustomize}
          className="w-full rounded border border-zinc-500 hover:border-zinc-300 py-3"
        >
          C Customize
        </button>
      </div>

      <p className="text-xs text-zinc-500">
        K keyboard paddle - B ball (WASD/trackpad) - E calibration - C customization - Esc menu
      </p>
    </section>
  );
};

