import { useCallback, useEffect, useRef, useState } from "react";
import TimeseriesGraph from "@/components/eeg/TimeseriesGraph";
import { HeadPlot } from "@/components/eeg/HeadPlot3D";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertCircle, Download, Settings2, X } from "lucide-react";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useEmbeddings } from "@/hooks/useClassificationModelEmbeddings";
import { useDevice } from "@/contexts/DeviceContext";
import { DEVICE_CONFIGS } from "@/config/eeg";
import type { DeviceType } from "@/config/eeg";
import type { BlinkDetectionSettings } from "@/types/eeg";

type BlinkSettingKey = Exclude<keyof BlinkDetectionSettings, "selectedChannels">;
type CapturedEEGRow = {
  timestampMs: number;
  channelValuesUv: number[];
};

export default function BCIDashboardPage() {
  const {
    displayData,
    sampleCount,
    errorMessage,
    clearError,
    registerStopEmbeddings,
    isStreaming,
    blink,
    blinkSettings,
    updateBlinkSettings,
    toggleBlinkDetectionChannel,
    resetBlinkSettings,
  } = useBCIStream();
  const { disableEmbeddings } = useEmbeddings();
  const { deviceType, setDeviceType, deviceConfig } = useDevice();
  const [isDismissed, setIsDismissed] = useState(false);
  const [isBlinkDialogOpen, setIsBlinkDialogOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);
  const [capturedSampleCount, setCapturedSampleCount] = useState(0);
  const [captureNotice, setCaptureNotice] = useState<string | null>(null);
  const captureRowsRef = useRef<CapturedEEGRow[]>([]);
  const lastCapturedSampleCountRef = useRef(0);

  // Register embedding stop callback
  useEffect(() => {
    registerStopEmbeddings(async () => {
      await disableEmbeddings();
    });
  }, [registerStopEmbeddings, disableEmbeddings]);

  useEffect(() => {
    if (!isBlinkDialogOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsBlinkDialogOpen(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [isBlinkDialogOpen]);

  const handleDismiss = () => {
    setIsDismissed(true);
    clearError();
  };

  const handleDeviceChange = (value: string) => {
    if (isStreaming) return; // Don't allow switching while streaming
    setDeviceType(value as DeviceType);
  };

  const handleBlinkSettingChange = (key: BlinkSettingKey, rawValue: string) => {
    const parsedValue = Number(rawValue);
    if (!Number.isFinite(parsedValue)) return;
    updateBlinkSettings({
      [key]: parsedValue,
    } as Partial<Omit<BlinkDetectionSettings, "selectedChannels">>);
  };

  const downloadCaptureCsv = useCallback(
    (rows: CapturedEEGRow[]) => {
      if (rows.length === 0) {
        setCaptureNotice("Capture stopped. No samples were recorded.");
        return;
      }

      const header = [
        "timestamp",
        ...deviceConfig.channelNames.map((channelName) => `${channelName}_uV`),
      ];
      const csvRows = rows.map((row) =>
        [
          String(row.timestampMs),
          ...row.channelValuesUv.map((value) => value.toFixed(6)),
        ].join(","),
      );
      const csvContent = [header.join(","), ...csvRows].join("\n");
      const blob = new Blob([csvContent], {
        type: "text/csv;charset=utf-8;",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const dateStamp = new Date().toISOString().replace(/[:.]/g, "-");
      link.href = url;
      link.download = `eeg_capture_${deviceType}_${dateStamp}.csv`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);

      setCaptureNotice(`Downloaded ${rows.length} samples to CSV.`);
    },
    [deviceConfig.channelNames, deviceType],
  );

  const startCapture = useCallback(() => {
    if (!isStreaming) return;

    captureRowsRef.current = [];
    lastCapturedSampleCountRef.current = sampleCount;
    setCapturedSampleCount(0);
    setCaptureNotice(null);
    setIsCapturing(true);
  }, [isStreaming, sampleCount]);

  const stopCaptureAndDownload = useCallback(() => {
    if (!isCapturing) return;

    const capturedRows = captureRowsRef.current;
    setIsCapturing(false);
    setCapturedSampleCount(capturedRows.length);
    captureRowsRef.current = [];
    lastCapturedSampleCountRef.current = sampleCount;
    downloadCaptureCsv(capturedRows);
  }, [downloadCaptureCsv, isCapturing, sampleCount]);

  useEffect(() => {
    if (!isCapturing) return;

    const pendingSampleCount = sampleCount - lastCapturedSampleCountRef.current;
    if (pendingSampleCount <= 0) return;

    const availableSampleCount = Math.min(pendingSampleCount, displayData.length);
    if (availableSampleCount <= 0) {
      lastCapturedSampleCountRef.current = sampleCount;
      return;
    }

    const droppedSampleCount = pendingSampleCount - availableSampleCount;
    const latestPoints = displayData.slice(-availableSampleCount);
    const capturedRows = latestPoints.map((point) => {
      const channelValuesUv = deviceConfig.channelNames.map((channelName) => {
        const filteredValue = point[`fch${channelName}`];
        if (typeof filteredValue === "number" && Number.isFinite(filteredValue)) {
          return filteredValue;
        }
        const rawValue = point[`ch${channelName}`];
        return typeof rawValue === "number" && Number.isFinite(rawValue)
          ? rawValue
          : 0;
      });

      return {
        timestampMs:
          typeof point.timestampMs === "number" && Number.isFinite(point.timestampMs)
            ? point.timestampMs
            : Date.now(),
        channelValuesUv,
      };
    });

    captureRowsRef.current.push(...capturedRows);
    setCapturedSampleCount((previous) => previous + capturedRows.length);
    lastCapturedSampleCountRef.current = sampleCount;

    if (droppedSampleCount > 0) {
      setCaptureNotice(
        `Dropped ${droppedSampleCount} samples from capture window.`,
      );
    }
  }, [displayData, deviceConfig.channelNames, isCapturing, sampleCount]);

  useEffect(() => {
    if (isStreaming || !isCapturing) return;
    stopCaptureAndDownload();
  }, [isCapturing, isStreaming, stopCaptureAndDownload]);

  return (
    <div className="h-full flex flex-col relative">
      {/* Device Selector Bar */}
      <div className="h-12 border-b border-zinc-200 px-4 flex items-center gap-4 bg-zinc-50/80 shrink-0">
        <label className="text-sm font-semibold text-zinc-600">Device:</label>
        <Select
          value={deviceType}
          onValueChange={handleDeviceChange}
          disabled={isStreaming}
        >
          <SelectTrigger className="w-[180px] h-8">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {(Object.keys(DEVICE_CONFIGS) as DeviceType[]).map((key) => (
              <SelectItem key={key} value={key}>
                {DEVICE_CONFIGS[key].name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {isStreaming && (
          <span className="text-xs text-zinc-400 italic">
            Stop streaming to change device
          </span>
        )}
        <div className="ml-auto flex items-center gap-3">
          <span className="text-xs font-semibold text-zinc-500">Capture</span>
          <span
            className={`text-xs font-semibold tabular-nums ${
              isCapturing ? "text-emerald-600" : "text-zinc-500"
            }`}
          >
            {capturedSampleCount} samples
          </span>
          <Button
            variant={isCapturing ? "default" : "outline"}
            size="sm"
            onClick={isCapturing ? stopCaptureAndDownload : startCapture}
            disabled={!isCapturing && !isStreaming}
            className="h-8 cursor-pointer"
            title={
              !isStreaming && !isCapturing
                ? "Start EEG streaming before capture"
                : undefined
            }
          >
            <Download className="h-3.5 w-3.5" />
            {isCapturing ? "Stop & Download" : "Capture CSV"}
          </Button>
          {captureNotice && (
            <span className="text-xs text-zinc-500 max-w-72 truncate" title={captureNotice}>
              {captureNotice}
            </span>
          )}
          <div className="h-5 w-px bg-zinc-200" />
          <span className="text-xs font-semibold text-zinc-500">Blinks</span>
          <span
            className={`h-2.5 w-2.5 rounded-full transition-all duration-150 ${
              blink.detected
                ? "bg-emerald-500 ring-4 ring-emerald-200 scale-125"
                : "bg-zinc-300"
            }`}
          />
          <span className="text-xs text-zinc-500 tabular-nums">
            {blink.blinkCount}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsBlinkDialogOpen(true)}
            className="h-8 cursor-pointer"
          >
            <Settings2 className="h-3.5 w-3.5" />
            Blink Settings
          </Button>
        </div>
      </div>

      {/* Central Error Popup */}
      {errorMessage && !isDismissed && (
        <div className="absolute inset-0 flex items-center justify-center z-50 pointer-events-none">
          <div className="max-w-lg w-full mx-4 pointer-events-auto">
            <Alert
              variant="destructive"
              className="shadow-2xl relative bg-white border-2 p-6"
            >
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 mt-0.5" />
                <div className="flex-1">
                  <AlertTitle className="text-lg mb-2">
                    Board Connection Error
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {errorMessage.includes("BOARD_NOT_READY") ||
                      errorMessage.includes("BOARD_NOT_CREATED")
                      ? "Unable to connect to the BCI board. Please ensure your board is powered on and properly connected via the dongle."
                      : errorMessage}
                  </AlertDescription>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleDismiss}
                  className="h-8 w-8 p-0 hover:bg-red-100 -mt-1 cursor-pointer"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>
            </Alert>
          </div>
        </div>
      )}

      {isBlinkDialogOpen && (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/30 p-4">
          <Card className="w-full max-w-3xl max-h-[85vh] overflow-hidden bg-white">
            <CardHeader className="border-b border-zinc-200">
              <div className="flex items-center justify-between">
                <CardTitle>Blink Detection Settings</CardTitle>
                <Button
                  variant="ghost"
                  size="icon-sm"
                  onClick={() => setIsBlinkDialogOpen(false)}
                  className="cursor-pointer"
                  aria-label="Close blink settings"
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <CardContent className="max-h-[70vh] overflow-y-auto p-5 space-y-6">
              <section className="space-y-2">
                <h2 className="text-sm font-semibold text-zinc-700">
                  Electrodes
                </h2>
                <p className="text-xs text-zinc-500">
                  Select channels used for blink detection.
                </p>
                <div className="grid gap-2 sm:grid-cols-4">
                  {deviceConfig.channelNames.map((channelName) => (
                    <label
                      key={channelName}
                      className="flex items-center gap-2 border rounded-md p-2 text-sm"
                    >
                      <Checkbox
                        checked={blinkSettings.selectedChannels.includes(channelName)}
                        onCheckedChange={() =>
                          toggleBlinkDetectionChannel(channelName)
                        }
                      />
                      <span>{channelName}</span>
                    </label>
                  ))}
                </div>
              </section>

              <section className="space-y-3">
                <h2 className="text-sm font-semibold text-zinc-700">Thresholds</h2>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-1">
                    <Label>Required channels</Label>
                    <Input
                      type="number"
                      min={1}
                      max={Math.max(1, blinkSettings.selectedChannels.length)}
                      step={1}
                      value={blinkSettings.requiredChannels}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "requiredChannels",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Z-score threshold</Label>
                    <Input
                      type="number"
                      min={0.5}
                      max={10}
                      step={0.1}
                      value={blinkSettings.zScoreThreshold}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "zScoreThreshold",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Min amplitude (µV)</Label>
                    <Input
                      type="number"
                      min={0}
                      max={500}
                      step={1}
                      value={blinkSettings.minAmplitudeUv}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "minAmplitudeUv",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Min deviation (µV)</Label>
                    <Input
                      type="number"
                      min={0.1}
                      max={100}
                      step={0.1}
                      value={blinkSettings.minDeviationUv}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "minDeviationUv",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                </div>
              </section>

              <section className="space-y-3">
                <h2 className="text-sm font-semibold text-zinc-700">
                  Timing and Adaptation
                </h2>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-1">
                    <Label>Warmup samples</Label>
                    <Input
                      type="number"
                      min={0}
                      max={deviceConfig.samplingRate * 10}
                      step={1}
                      value={blinkSettings.warmupSamples}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "warmupSamples",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Baseline alpha</Label>
                    <Input
                      type="number"
                      min={0.001}
                      max={0.3}
                      step={0.001}
                      value={blinkSettings.baselineAlpha}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "baselineAlpha",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Cooldown (ms)</Label>
                    <Input
                      type="number"
                      min={0}
                      max={5000}
                      step={10}
                      value={blinkSettings.cooldownMs}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "cooldownMs",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                  <div className="space-y-1">
                    <Label>Flash duration (ms)</Label>
                    <Input
                      type="number"
                      min={50}
                      max={3000}
                      step={10}
                      value={blinkSettings.flashMs}
                      onChange={(event) =>
                        handleBlinkSettingChange(
                          "flashMs",
                          event.target.value,
                        )
                      }
                    />
                  </div>
                </div>
              </section>

              <div className="flex items-center justify-end gap-2">
                <Button
                  variant="outline"
                  onClick={resetBlinkSettings}
                  className="cursor-pointer"
                >
                  Reset Defaults
                </Button>
                <Button
                  onClick={() => setIsBlinkDialogOpen(false)}
                  className="cursor-pointer"
                >
                  Close
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Half*/}
        <div className="w-1/2 h-full">
          <TimeseriesGraph />
        </div>

        {/* Right Half */}
        <div className="w-1/2 h-full flex flex-col">
          <HeadPlot />
        </div>
      </div>
    </div>
  );
}
