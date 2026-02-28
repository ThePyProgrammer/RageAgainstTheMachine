type KeyboardHintsProps = {
  mode: "menu" | "game";
};

export const KeyboardHints = ({ mode }: KeyboardHintsProps) => {
  return (
    <div className="fixed bottom-3 left-1/2 -translate-x-1/2 z-20">
      {mode === "menu" ? (
        <p className="rounded bg-black/75 px-3 py-2 text-white text-xs">
          K keyboard paddle - B ball (WASD/trackpad) - E EEG calibration - C customize - Esc menu
        </p>
      ) : (
        <p className="rounded bg-black/75 px-3 py-2 text-white text-xs">
          LEFT/RIGHT or A/D to move paddle - P pause - O options - Esc menu
        </p>
      )}
    </div>
  );
};

