const TAUNT_LIBRARY = {
  calm: ["Breathe. Your control loop is twitchy.", "Nice, you just lost the signal."],
  spike: ["System detected panic in the cockpit!", "Try not to overcorrect, rebel."],
  dominate: ["That was textbook", "Locked and loaded, boss."],
};

export const pickTaunt = (seed: number, stress: number): string => {
  const bucket = stress >= 0.75 ? TAUNT_LIBRARY.spike : stress <= 0.35 ? TAUNT_LIBRARY.calm : TAUNT_LIBRARY.dominate;
  const index = seed % bucket.length;
  return bucket[index] ?? "Keep going.";
};
