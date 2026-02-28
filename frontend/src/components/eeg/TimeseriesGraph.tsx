import React, {
  useState,
  useMemo,
  useCallback,
  useRef,
  useEffect,
} from "react";
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  type ChartOptions,
  type TooltipItem,
  type Chart,
  type ChartData,
} from "chart.js";
import { Line } from "react-chartjs-2";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useDevice } from "@/contexts/DeviceContext";
import {
  VERT_SCALES,
  WINDOW_SECONDS,
} from "@/config/eeg";
import type { ChannelRange } from "@/types/eeg";

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
);

// Chart container that properly updates when scale changes
interface ChartContainerProps {
  data: ChartData<"line", (number | { x: number; y: number })[], unknown>;
  currentScale: number | "auto";
  range: ChannelRange;
  windowRange?: { min: number; max: number };
  getChartOptions: (
    scale: number | "auto",
    range: ChannelRange,
    windowRange?: { min: number; max: number },
  ) => ChartOptions<"line">;
}

const ChartContainer: React.FC<ChartContainerProps> = ({
  data,
  currentScale,
  range,
  windowRange,
  getChartOptions,
}) => {
  const chartRef = useRef<{ chart: Chart<"line"> } | null>(null);

  useEffect(() => {
    if (chartRef.current) {
      const chart = chartRef.current.chart;
      const options = getChartOptions(currentScale, range, windowRange);

      // Update only the Y axis bounds
      if (chart && chart.scales && chart.scales.y && options.scales?.y) {
        const yMin = options.scales.y.min;
        const yMax = options.scales.y.max;
        if (typeof yMin === "number" && typeof yMax === "number") {
          chart.scales.y.min = yMin;
          chart.scales.y.max = yMax;
          chart.update("none"); // Update without animation
        }
      }
    }
  }, [currentScale, range, windowRange, getChartOptions]);

  return (
    <div className="w-full h-full px-2">
      <Line
        ref={(instance) => {
          if (instance) {
            chartRef.current = { chart: instance };
          }
        }}
        data={data}
        options={getChartOptions(currentScale, range, windowRange)}
      />
    </div>
  );
};

const TimeseriesGraph: React.FC = () => {
  const { displayData, sampleCount, channelRanges } = useBCIStream();
  const { deviceConfig } = useDevice();

  const channelNames = deviceConfig.channelNames;
  const channelColors = deviceConfig.channelColors;
  const samplingRate = deviceConfig.samplingRate;

  const [vertScale, setVertScale] = useState<number | "auto">(200);
  const [windowSeconds, setWindowSeconds] = useState(5);
  const [channelScales, setChannelScales] = useState<
    Record<string, number | "auto">
  >({});
  const [editingChannel, setEditingChannel] = useState<string | null>(null);
  const [editValue, setEditValue] = useState<string>("");

  // Use the raw data directly
  const throttledData = displayData;

  const currentRunTime = sampleCount / samplingRate;

  const handleWindowChange = useCallback((value: string): void => {
    setWindowSeconds(Number(value));
  }, []);

  const handleVertScaleChange = useCallback((value: string): void => {
    setVertScale(value === "auto" ? "auto" : Number(value));
    setChannelScales({});
  }, []);

  const chartData = useMemo(() => {
    const pointsToShow = windowSeconds * samplingRate;
    const windowedData = throttledData.slice(-pointsToShow);
    const windowRanges: Record<string, { min: number; max: number }> = {};

    return {
      datasets: channelNames.map((ch, idx) => {
        const chKey = `fch${ch}`;
        const channelKey = `ch${ch}`; // Used for metrics key matching

        // Calculate min/max while mapping data (single pass, efficient)
        let min = Infinity;
        let max = -Infinity;

        const data = windowedData.map((d, i) => {
          const y = (d[chKey] as number) ?? 0;
          if (y < min) min = y;
          if (y > max) max = y;
          return { x: i, y };
        });

        // Handle case where data is flat or empty
        if (min === Infinity || min === max) {
          min = -1;
          max = 1;
        }

        // Store range for this channel using channelKey to match metrics
        windowRanges[channelKey] = { min, max };

        return {
          label: ch,
          data: data,
          borderColor: channelColors[idx],
          borderWidth: 1.5,
          pointRadius: 0,

          // Performance Flags
          borderJoinStyle: "round" as const,
          tension: 0.1,
          spanGaps: true,
          normalized: true,
          parsing: false as const,
        };
      }),
      windowRanges,
    };
  }, [throttledData, windowSeconds, channelNames, channelColors, samplingRate]);

  const getChartOptions = (
    scale: number | "auto",
    windowRange?: { min: number; max: number },
  ): ChartOptions<"line"> => {
    let min: number, max: number;

    if (typeof scale === "number") {
      // Manual scale set - use it directly (symmetric around 0)
      min = -Math.abs(scale);
      max = Math.abs(scale);
    } else if (scale === "auto" && windowRange) {
      // Tight Auto-Scale: Use raw min/max from data
      const rawMin = windowRange.min;
      const rawMax = windowRange.max;
      const delta = Math.abs(rawMax - rawMin);

      // Use minimum delta of 2µV to ensure tiny signals are visible
      const effectiveDelta = Math.max(delta, 2);

      min = rawMin - effectiveDelta;
      max = rawMax + effectiveDelta;
    } else {
      // Fallback if no windowRange available yet
      min = -100;
      max = 100;
    }

    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false, // Disable animation for real-time performance
      parsing: false, // Match dataset setting

      plugins: {
        legend: { display: false },
        tooltip: {
          enabled: true,
          mode: "index",
          intersect: false,
          animation: { duration: 0 }, // Instant tooltip
          callbacks: {
            label: (context: TooltipItem<"line">) => {
              const val = context.parsed.y;
              return val !== null ? `${val.toFixed(1)} µV` : "";
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear", // Critical for decimation
          display: false,
          grid: { display: false },
          // Ensure X axis stays stable
          min: 0,
          max: Math.max(
            chartData.datasets[0]?.data.length || 0,
            windowSeconds * samplingRate,
          ),
        },
        y: {
          min,
          max,
          grid: {
            color: (context: { tick: { value: number } }) =>
              context.tick.value === 0 ? "#e5e7eb" : "transparent",
            drawTicks: false,
          },
          ticks: { display: false },
        },
      },
      interaction: {
        mode: "nearest",
        axis: "x",
        intersect: false,
      },
    };
  };

  return (
    <div className="h-full w-full bg-white flex flex-col overflow-hidden font-sans text-zinc-900 border border-zinc-200 shadow-sm">
      {/* --- HEADER --- */}
      <header className="h-14 border-b border-zinc-200 px-4 flex items-center justify-between bg-white shrink-0 z-30 relative">
        {/* Left: Title & Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <h1 className="text-lg font-bold tracking-tight text-zinc-900">
              Time Series
            </h1>
          </div>
        </div>

        {/* Right: Controls */}
        <div className="flex items-center gap-4">
          {/* Scale Selector */}
          <div className="flex items-center gap-3">
            <label className="text-sm font-semibold text-zinc-600">
              Vert Scale:
            </label>
            <Select
              value={String(vertScale)}
              onValueChange={handleVertScaleChange}
            >
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                {VERT_SCALES.map((s) => (
                  <SelectItem key={s} value={String(s)}>
                    {s} µV
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* Window Selector */}
          <div className="flex items-center gap-3">
            <label className="text-sm font-semibold text-zinc-600">
              Window:
            </label>
            <Select
              value={String(windowSeconds)}
              onValueChange={handleWindowChange}
            >
              <SelectTrigger className="w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WINDOW_SECONDS.map((s) => (
                  <SelectItem key={s} value={String(s)}>
                    {s}s
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      </header>

      {/* --- MAIN CONTENT --- */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className="flex-1 flex flex-col">
          {/* 1. Channel List */}
          {channelNames.map((ch, idx) => {
            const channelKey = `ch${ch}`;
            const color = channelColors[idx];
            const range = channelRanges[channelKey] || {
              min: 0,
              max: 0,
              railed: false,
              railedWarn: false,
              rmsUv: 0,
              dcOffsetPercent: 0,
            };
            const windowRange = chartData.windowRanges?.[channelKey];

            const currentScale =
              channelScales[channelKey] !== undefined
                ? channelScales[channelKey]
                : vertScale;

            // Prepare single dataset
            const singleChannelData = {
              datasets: [chartData.datasets[idx]],
            };

            return (
              <div
                key={ch}
                className="flex border-b border-zinc-200 h-28 relative group hover:bg-zinc-50 transition-colors"
              >
                {/* Left: Channel Indicator */}
                <div className="w-15 shrink-0 flex flex-col items-center justify-center border-r border-zinc-200 bg-zinc-50/50 z-10">
                  <div
                    className="w-8 h-8 rounded-full flex items-center justify-center text-sm text-white font-bold shadow-sm mb-1 ring-2 ring-white"
                    style={{ backgroundColor: color }}
                  >
                    {ch.length > 3 ? ch.slice(0, 3) : ch}
                  </div>
                </div>

                {/* Center: Chart */}
                <div className="flex-1 relative min-w-0">
                  {/* Calculate scale values for display */}
                  {(() => {
                    let displayMin: number;
                    let displayMax: number;

                    if (currentScale === "auto" && windowRange) {
                      // Auto mode: use actual min/max from data
                      displayMin = windowRange.min;
                      displayMax = windowRange.max;
                    } else if (currentScale === "auto") {
                      // Fallback auto mode
                      displayMin = range.min || -100;
                      displayMax = range.max || 100;
                    } else {
                      // Manual scale: symmetric
                      const displayScale = Math.abs(currentScale as number);
                      displayMin = -displayScale;
                      displayMax = displayScale;
                    }

                    // Check if this channel is being edited
                    const isEditing = editingChannel === channelKey;

                    return (
                      <>
                        {/* Scale Labels or Input */}
                        {isEditing ? (
                          <div className="absolute left-2 top-1/2 -translate-y-1/2 z-20 flex items-center gap-2 bg-white border-2 border-blue-500 rounded-md px-2 py-1 shadow-md">
                            <Input
                              type="number"
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              onBlur={() => {
                                if (editValue && !isNaN(Number(editValue))) {
                                  setChannelScales((prev) => ({
                                    ...prev,
                                    [channelKey]: Math.max(
                                      10,
                                      Number(editValue),
                                    ),
                                  }));
                                }
                                setEditingChannel(null);
                              }}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") {
                                  if (editValue && !isNaN(Number(editValue))) {
                                    setChannelScales((prev) => ({
                                      ...prev,
                                      [channelKey]: Math.max(
                                        10,
                                        Number(editValue),
                                      ),
                                    }));
                                  }
                                  setEditingChannel(null);
                                } else if (e.key === "Escape") {
                                  setEditingChannel(null);
                                }
                              }}
                              autoFocus
                              className="w-24 h-8"
                              placeholder="Enter µV"
                            />
                            <span className="text-xs text-zinc-600 whitespace-nowrap">
                              µV
                            </span>
                          </div>
                        ) : (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="absolute left-1 top-1 text-xs font-mono pointer-events-auto z-10 h-auto py-0 px-2 group/btn transition-all hover:bg-blue-100 hover:border-2 hover:border-blue-400 hover:shadow-md hover:cursor-edit"
                              onDoubleClick={() => {
                                setEditingChannel(channelKey);
                                setEditValue(
                                  String(Math.round(Math.abs(displayMax))),
                                );
                              }}
                              title="Double-click to edit"
                            >
                              {displayMax >= 0 ? "+" : ""}
                              {Math.round(displayMax)} µV
                              <span className="ml-1 text-[10px] opacity-0 group-hover/btn:opacity-100 transition-opacity">
                                ✎
                              </span>
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="absolute left-1 bottom-1 text-xs font-mono pointer-events-auto z-10 h-auto py-0 px-2 group/btn transition-all hover:bg-blue-100 hover:border-2 hover:border-blue-400 hover:shadow-md hover:cursor-edit"
                              onDoubleClick={() => {
                                setEditingChannel(channelKey);
                                setEditValue(
                                  String(Math.round(Math.abs(displayMin))),
                                );
                              }}
                              title="Double-click to edit"
                            >
                              {displayMin >= 0 ? "+" : ""}
                              {Math.round(displayMin)} µV
                              <span className="ml-1 text-[10px] opacity-0 group-hover/btn:opacity-100 transition-opacity">
                                ✎
                              </span>
                            </Button>
                          </>
                        )}
                      </>
                    );
                  })()}

                  {/* Chart.js Instance */}
                  <ChartContainer
                    key={`${ch}-${currentScale}`}
                    data={singleChannelData}
                    currentScale={currentScale}
                    range={range}
                    windowRange={windowRange}
                    getChartOptions={getChartOptions}
                  />
                </div>

                {/* Right: Metrics */}
                <div className="w-35 shrink-0 bg-white/50 border-l border-zinc-200 flex flex-col justify-center px-5 font-mono text-sm">
                  <div
                    className={`tracking-wide font-bold ${range.railed
                        ? "text-red-600" // RAILED (>90%)
                        : range.railedWarn
                          ? "text-yellow-600" // NEAR RAILED (75-90%)
                          : "text-green-600" // NOT RAILED (<75%)
                      }`}
                  >
                    {range.railed
                      ? "RAILED"
                      : range.railedWarn
                        ? "NEAR RAILED"
                        : "NOT RAILED"}
                  </div>

                  <span
                    className={
                      range.railed || range.railedWarn
                        ? "text-yellow-600 font-bold"
                        : ""
                    }
                  >
                    {range.dcOffsetPercent.toFixed(2)}%
                  </span>
                  <span>
                    {range.rmsUv.toFixed(2)}{" "}
                    <span className="text-xs">µVrms</span>
                  </span>
                </div>
              </div>
            );
          })}

          {/* 2. Integrated Footer (Time Axis) */}
          <div className="flex h-12 -mt-1.5 shrink-0 z-0 pointer-events-none relative">
            <div className="w-14"></div>

            <div className="flex-1 relative">
              <div className="absolute inset-0 flex items-start">
                {Array.from({ length: windowSeconds + 1 }).map((_, i) => {
                  const secondsBack = -windowSeconds + i;
                  const pct = (i / windowSeconds) * 100;
                  if (i === windowSeconds + 1) return null;

                  return (
                    <div
                      key={i}
                      className="absolute top-0 h-full flex flex-col items-center"
                      style={{
                        left: `${pct}%`,
                        width: "1px",
                        overflow: "visible",
                      }}
                    >
                      <div className="h-3 w-px bg-zinc-400 mb-1"></div>
                      <span className="text-xs font-medium text-zinc-500 bg-white/80 px-1 rounded">
                        {secondsBack}s
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="absolute mt-9 -mx-7 inset-x-0 flex items-center justify-between font-mono text-[11px] text-zinc-600 px-2">
                <span className="rounded-lg bg-white/90 border border-zinc-200 px-2 py-0.5 shadow-sm">
                  {(currentRunTime - windowSeconds).toFixed(1)}s
                </span>
                <span className="rounded-lg bg-white/90 border border-zinc-200 px-2 py-0.5 shadow-sm">
                  {currentRunTime.toFixed(1)}s
                </span>
              </div>
            </div>

            <div className="w-35 border-l border-zinc-200/0"></div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TimeseriesGraph;
