import { useState, useEffect } from "react";
import TimeseriesGraph from "@/components/eeg/TimeseriesGraph";
import { HeadPlot } from "@/components/eeg/HeadPlot3D";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { AlertCircle, X } from "lucide-react";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useEmbeddings } from "@/hooks/useClassificationModelEmbeddings";
import { useDevice } from "@/contexts/DeviceContext";
import { DEVICE_CONFIGS } from "@/config/eeg";
import type { DeviceType } from "@/config/eeg";

export default function BCIDashboardPage() {
  const { errorMessage, clearError, registerStopEmbeddings, isStreaming } =
    useBCIStream();
  const { disableEmbeddings } = useEmbeddings();
  const { deviceType, setDeviceType } = useDevice();
  const [isDismissed, setIsDismissed] = useState(false);

  // Register embedding stop callback
  useEffect(() => {
    registerStopEmbeddings(async () => {
      await disableEmbeddings();
    });
  }, [registerStopEmbeddings, disableEmbeddings]);

  const handleDismiss = () => {
    setIsDismissed(true);
    clearError();
  };

  const handleDeviceChange = (value: string) => {
    if (isStreaming) return; // Don't allow switching while streaming
    setDeviceType(value as DeviceType);
  };

  return (
    <div className="h-full flex flex-col relative">
      {/* Device Selector Bar */}
      <div className="h-10 border-b border-zinc-200 px-4 flex items-center gap-4 bg-zinc-50/80 shrink-0">
        <label className="text-sm font-semibold text-zinc-600">Device:</label>
        <Select value={deviceType} onValueChange={handleDeviceChange} disabled={isStreaming}>
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
