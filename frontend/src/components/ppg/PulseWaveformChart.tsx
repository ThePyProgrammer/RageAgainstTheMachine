import { useRef, useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from "chart.js";
import type { ChartOptions } from "chart.js";
import { Line } from "react-chartjs-2";

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
);

interface PulseWaveformChartProps {
  dataPoint: number;
  maxPoints?: number;
  color?: string;
  height?: number;
}

// Chart options to remove axes and make it look like a medical monitor
const options: ChartOptions<"line"> = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false, // Critical for performance
  plugins: {
    legend: {
      display: false,
    },
    tooltip: {
      enabled: false,
    },
  },
  scales: {
    x: {
      display: false, // Hide X axis
      grid: {
        display: false,
      },
    },
    y: {
      display: false, // Hide Y axis
      grid: {
        display: false,
      },
      // Let it auto-scale for BVP as amplitude varies
    },
  },
  elements: {
    point: {
      radius: 0, // Hide points
    },
    line: {
      borderWidth: 2,
      tension: 0.4, // Smooth curve
    },
  },
};

export function PulseWaveformChart({
  dataPoint,
  maxPoints = 100,
  color = "rgb(239, 68, 68)",
  height = 60,
}: PulseWaveformChartProps) {
  // Initialize with zeros
  const [data, setData] = useState<number[]>(new Array(maxPoints).fill(0));
  const lastPointRef = useRef<number>(0);

  useEffect(() => {
    const resetTimer = setTimeout(() => {
      setData(new Array(maxPoints).fill(0));
      lastPointRef.current = 0;
    }, 0);

    return () => clearTimeout(resetTimer);
  }, [maxPoints]);

  useEffect(() => {
    // Only update if value changed to avoid unnecessary renders
    // Using a small epsilon for float comparison might be safer, but direct comparison is faster
    if (dataPoint !== lastPointRef.current) {
      const updateTimer = setTimeout(() => {
        setData((prevData) => [...prevData.slice(1), dataPoint]);
      }, 0);
      lastPointRef.current = dataPoint;

      return () => clearTimeout(updateTimer);
    }
  }, [dataPoint]);

  const chartData = {
    labels: new Array(maxPoints).fill(""),
    datasets: [
      {
        label: "BVP",
        data: data,
        borderColor: color,
        backgroundColor: color,
        fill: false,
      },
    ],
  };

  return (
    <div style={{ height: `${height}px`, width: "100%" }}>
      <Line options={options} data={chartData} />
    </div>
  );
}
