import { BRAIN_GRADIENT_COLORS } from "@/config/eeg";

const ColorLegend = () => {
  // Create CSS linear gradient string
  const gradientString = `linear-gradient(to right, ${BRAIN_GRADIENT_COLORS.join(", ")})`;

  return (
    <div className="absolute bottom-6 left-1/2 -translate-x-1/2">
      {/* Legend Container */}
      <div className="flex flex-col items-center">
        {/* The Gradient Bar */}
        <div className="relative w-64">
          <div
            className="h-2.5 rounded-full border border-zinc-200/50 shadow-inner"
            style={{ background: gradientString }}
          >
            {/* Tick mark for 0µV baseline */}
            <div className="absolute left-[10%] top-0 w-px h-full bg-white/30 rounded-full" />
          </div>
        </div>

        {/* Min and Max labels */}
        <div className="flex mt-1 text-xs font-medium text-zinc-600">
          Microvolts (µV)
        </div>
      </div>
    </div>
  );
};

export default ColorLegend;
