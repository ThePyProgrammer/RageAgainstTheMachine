import { useWebcamPulse } from "@/hooks/useWebcamPulse";
import { PulseControlBar } from "@/components/ppg/PulseControlBar";
import { PulseVideoFeed } from "@/components/ppg/PulseVideoFeed";

export default function PulseDetectorPage() {
  const {
    isStreaming,
    currentBPM,
    currentSpO2,
    currentRR,
    currentBVP,
    faceDetected,
    faceLocked,
    faceBox,
    foreheadBox,
    calibrating,
    calibrationProgress,
    calibrationTarget,
    waitTime,
    selectedCamera,
    availableCameras,
    processingTime,
    framesDropped,
    videoRef,
    canvasRef,
    startWebcam,
    stopWebcam,
    toggleFaceLock,
    switchCamera,
    averagePupilDiameter,
  } = useWebcamPulse();

  return (
    <div className="h-full flex flex-col gap-4 p-4 bg-slate-50">
      <PulseControlBar
        isStreaming={isStreaming}
        faceDetected={faceDetected}
        faceLocked={faceLocked}
        selectedCamera={selectedCamera}
        availableCameras={availableCameras}
        onStart={startWebcam}
        onStop={stopWebcam}
        onToggleLock={toggleFaceLock}
        onSwitchCamera={switchCamera}
      />

      <PulseVideoFeed
        videoRef={videoRef}
        canvasRef={canvasRef}
        isStreaming={isStreaming}
        currentBPM={currentBPM}
        currentSpO2={currentSpO2}
        currentRR={currentRR}
        currentBVP={currentBVP}
        faceDetected={faceDetected}
        faceLocked={faceLocked}
        faceBox={faceBox}
        foreheadBox={foreheadBox}
        calibrating={calibrating}
        calibrationProgress={calibrationProgress}
        calibrationTarget={calibrationTarget}
        waitTime={waitTime}
        averagePupilDiameter={averagePupilDiameter}
      />

      {/* Performance Overlay */}
      {isStreaming && (
        <div className="text-xs text-gray-500 flex gap-4">
          <span>Processing: {processingTime.toFixed(1)}ms</span>
          <span>Frames dropped: {framesDropped}</span>
        </div>
      )}
    </div>
  );
}
