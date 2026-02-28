interface PupilDataDisplayProps {
  averagePupilDiameter: number | null;
}

export function PupilDataDisplay({
  averagePupilDiameter,
}: PupilDataDisplayProps) {
  // Show 0mm if no eyes detected or diameter is null/zero
  const displayDiameter =
    averagePupilDiameter && averagePupilDiameter > 0
      ? averagePupilDiameter.toFixed(2)
      : "0.00";

  return (
    <div className="absolute bottom-4 right-4 bg-black/70 text-white px-4 py-3 rounded-lg backdrop-blur-sm min-w-45">
      <div className="text-center">
        <div className="text-xs text-gray-400 mb-1">Pupil Diameter</div>
        <div className="text-3xl font-bold font-mono">
          <span
            className={
              averagePupilDiameter && averagePupilDiameter > 0
                ? "text-green-400"
                : "text-gray-500"
            }
          >
            {displayDiameter}
          </span>
          <span className="text-lg text-gray-400 ml-1">mm</span>
        </div>
      </div>
    </div>
  );
}
