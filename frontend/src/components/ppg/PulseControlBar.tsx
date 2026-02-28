import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Video, VideoOff, Camera, Lock, Unlock } from "lucide-react";

interface PulseControlBarProps {
  isStreaming: boolean;
  faceDetected: boolean;
  faceLocked: boolean;
  selectedCamera: number;
  availableCameras: MediaDeviceInfo[];
  onStart: () => void;
  onStop: () => void;
  onToggleLock: () => void;
  onSwitchCamera: () => void;
}

export function PulseControlBar({
  isStreaming,
  faceDetected,
  faceLocked,
  selectedCamera,
  availableCameras,
  onStart,
  onStop,
  onToggleLock,
  onSwitchCamera,
}: PulseControlBarProps) {
  return (
    <Card className="shadow-sm">
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          {/* Left: Title and Status */}
          <div className="flex items-center gap-6">
            <div className="flex items-center gap-3">
              <div>
                <h1 className="text-xl font-bold text-slate-900">
                  Webcam Pulse Detector
                </h1>
                <p className="text-sm text-slate-600">
                  Real-time heart rate monitoring
                </p>
              </div>
            </div>

            {/* Status Badges */}
            <div className="flex items-center gap-2">
              <Badge
                variant={isStreaming ? "default" : "secondary"}
                className="font-medium"
              >
                {isStreaming ? "● Active" : "○ Inactive"}
              </Badge>
              {isStreaming && (
                <>
                  <Badge
                    variant={
                      faceLocked
                        ? "default"
                        : faceDetected
                          ? "default"
                          : "outline"
                    }
                    className={
                      faceLocked
                        ? "font-medium bg-gray-700 text-white"
                        : faceDetected
                          ? "font-medium bg-green-600"
                          : "font-medium"
                    }
                  >
                    {faceLocked
                      ? "🔒 Face Locked"
                      : faceDetected
                        ? "● Face Detected"
                        : "○ Searching"}
                  </Badge>
                  <Badge
                    variant={faceLocked ? "default" : "outline"}
                    className="font-medium"
                  >
                    {faceLocked ? "🔒 Locked" : "🔓 Unlocked"}
                  </Badge>
                  <Badge variant="outline" className="font-medium">
                    Camera {selectedCamera + 1}/{availableCameras.length || 1}
                  </Badge>
                </>
              )}
            </div>
          </div>

          {/* Right: Controls */}
          <div className="flex gap-2">
            {availableCameras.length > 1 && isStreaming && (
              <Button
                onClick={onSwitchCamera}
                variant="outline"
                size="default"
                className="cursor-pointer"
              >
                <Camera className="mr-2 h-4 w-4" />
                Switch Camera
              </Button>
            )}
            {isStreaming && (
              <Button
                onClick={onToggleLock}
                variant="outline"
                size="default"
                className="cursor-pointer"
              >
                {faceLocked ? (
                  <>
                    <Lock className="mr-2 h-4 w-4" />
                    Unlock Face
                  </>
                ) : (
                  <>
                    <Unlock className="mr-2 h-4 w-4" />
                    Lock Face
                  </>
                )}
              </Button>
            )}
            {isStreaming ? (
              <Button
                onClick={onStop}
                size="default"
                className="bg-red-500 text-white hover:bg-red-600 shadow-sm cursor-pointer"
              >
                <VideoOff className="mr-2 h-4 w-4" />
                Stop
              </Button>
            ) : (
              <Button
                onClick={onStart}
                size="default"
                className="text-black shadow-sm border-2 cursor-pointer"
              >
                <Video className="mr-2 h-4 w-4" />
                Start
              </Button>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
