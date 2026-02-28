import { useEffect, useState } from "react";
import type { RefObject } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Video } from "lucide-react";
import { PulseBPMDisplay } from "./PulseBPMDisplay";
import { PupilDataDisplay } from "../pupillometry/PupilDataDisplay";


type Box = { x: number; y: number; w: number; h: number };

interface PulseVideoFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  canvasRef: RefObject<HTMLCanvasElement | null>;
  isStreaming: boolean;
  currentBPM: number;
  currentSpO2: number;
  currentRR: number;
  currentBVP: number;
  faceDetected: boolean;
  faceLocked: boolean;
  faceBox: Box | null;
  foreheadBox: Box | null;
  calibrating?: boolean;
  calibrationProgress?: number;
  calibrationTarget?: number;

  waitTime?: number | null;
  averagePupilDiameter?: number | null;
}

export function PulseVideoFeed({
  videoRef,
  canvasRef,
  isStreaming,
  currentBPM,
  currentSpO2,
  currentRR,
  currentBVP,
  faceDetected,
  faceLocked,
  faceBox,
  foreheadBox,
  calibrating = false,
  calibrationProgress = 0,
  calibrationTarget = 300,

  waitTime = null,
  averagePupilDiameter,
}: PulseVideoFeedProps) {
  const [videoDims, setVideoDims] = useState<{ w: number; h: number }>({
    w: 0,
    h: 0,
  });

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const updateDims = () => {
      setVideoDims({
        w: video.videoWidth || video.clientWidth || 0,
        h: video.videoHeight || video.clientHeight || 0,
      });
    };

    updateDims();
    video.addEventListener("loadedmetadata", updateDims);
    window.addEventListener("resize", updateDims);

    return () => {
      video.removeEventListener("loadedmetadata", updateDims);
      window.removeEventListener("resize", updateDims);
    };
  }, [videoRef]);

  const getBoxStyle = (box: Box | null, color: string) => {
    if (!box || !videoDims.w || !videoDims.h) return undefined;

    return {
      position: "absolute" as const,
      border: `3px solid ${color}`,
      borderRadius: "2px",
      left: `${(box.x / videoDims.w) * 100}%`,
      top: `${(box.y / videoDims.h) * 100}%`,
      width: `${(box.w / videoDims.w) * 100}%`,
      height: `${(box.h / videoDims.h) * 100}%`,
      boxSizing: "border-box" as const,
      pointerEvents: "none" as const,
    };
  };

  return (
    <Card className="flex-1 shadow-sm overflow-hidden bg-slate-950">
      <CardContent className="p-2 h-full w-full flex items-center justify-center relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover rounded-lg"
        />
        <canvas
          ref={canvasRef}
          style={{
            display: "none",
            position: "absolute",
            visibility: "hidden",
          }}
        />

        {isStreaming && faceDetected && (
          <>
            <div style={getBoxStyle(faceBox, "rgba(0, 100, 255, 0.8)")} />
            <div style={getBoxStyle(foreheadBox, "rgba(0, 255, 100, 0.9)")} />
          </>
        )}

        {/* Calibration Status Overlay */}
        {isStreaming && faceLocked && calibrating && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-black/70 text-white px-4 py-2 rounded-lg backdrop-blur-sm">
            <div className="text-sm font-medium">
              Calibrating: {calibrationProgress}/{calibrationTarget} samples
            </div>
            {waitTime !== null && waitTime > 0 && (
              <div className="text-xs text-gray-300 text-center mt-1">
                estimate: {currentBPM > 0 ? `${currentBPM} bpm` : "-- bpm"},
                wait {waitTime.toFixed(1)}s
              </div>
            )}
            <div className="w-48 h-1 bg-gray-600 rounded-full mt-2">
              <div
                className="h-full bg-green-500 rounded-full transition-all duration-200"
                style={{
                  width: `${(calibrationProgress / calibrationTarget) * 100}%`,
                }}
              />
            </div>
          </div>
        )}

        {/* BPM Display Overlay - Always visible when streaming */}
        {isStreaming && (
          <PulseBPMDisplay
            bpm={currentBPM}
            spo2={currentSpO2}
            rr={currentRR}
            bvp={currentBVP}
          />
        )}

        {/* Pupil Data Display - Right side */}
        {isStreaming && (
          <PupilDataDisplay
            averagePupilDiameter={averagePupilDiameter ?? null}
          />
        )}

        {/* Start Prompt */}
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="text-center text-slate-400">
              <Video size={48} className="mx-auto mb-3 opacity-50" />
              <p className="text-base">
                Click "Start" to begin pulse detection
              </p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
