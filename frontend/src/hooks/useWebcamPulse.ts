import { useState, useRef, useEffect, useCallback } from "react";
import { CAMERA_CONFIG, PULSE_CONFIG } from "@/config/ppg";
import { API_ENDPOINTS } from "@/config/api";
import { useSessionStatus } from "@/contexts/SessionStatusContext";
import type { EyeData, PupillometryResponse } from "@/types/ocular";

type Box = { x: number; y: number; w: number; h: number };

type PulseDetectionResponse = {
  faceDetected?: boolean;
  face?: Box;
  forehead?: Box;
  bpm?: number;
  faceLocked?: boolean;
  calibrating?: boolean;
  calibrationProgress?: number;
  calibrationTarget?: number;
  waitTime?: number;
  bvp?: number;
  spo2?: number;
  rr?: number;
};

export function useWebcamPulse() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentBPM, setCurrentBPM] = useState<number>(0);
  const [currentSpO2, setCurrentSpO2] = useState<number>(0);
  const [currentRR, setCurrentRR] = useState<number>(0);
  const [currentBVP, setCurrentBVP] = useState<number>(0);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceLocked, setFaceLocked] = useState(false);
  const [faceBox, setFaceBox] = useState<Box | null>(null);
  const [foreheadBox, setForeheadBox] = useState<Box | null>(null);
  const [calibrating, setCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationTarget, setCalibrationTarget] = useState(300);
  const [waitTime, setWaitTime] = useState<number | null>(null);
  const [selectedCamera, setSelectedCamera] = useState(0);
  const [availableCameras, setAvailableCameras] = useState<MediaDeviceInfo[]>(
    [],
  );
  const [processingTime, setProcessingTime] = useState<number>(0);
  const [framesDropped, setFramesDropped] = useState<number>(0);

  // Pupil Data State
  const [averagePupilDiameter, setAveragePupilDiameter] = useState<
    number | null
  >(null);
  const [leftEye, setLeftEye] = useState<EyeData | null>(null);
  const [rightEye, setRightEye] = useState<EyeData | null>(null);

  const { setWebcam } = useSessionStatus();

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pupilWsRef = useRef<WebSocket | null>(null);
  const faceLockedRef = useRef<boolean>(false);
  const animationFrameRef = useRef<number | null>(null);
  const lastBPMUpdateTime = useRef<number>(0);
  const isProcessing = useRef<boolean>(false);
  const lastPupilSendTime = useRef<number>(0);

  // Get available cameras
  useEffect(() => {
    const getCameras = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const cameras = devices.filter(
          (device) => device.kind === "videoinput",
        );
        setAvailableCameras(cameras);
      } catch (error) {
        console.error("Error getting cameras:", error);
      }
    };
    getCameras();
  }, []);

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(API_ENDPOINTS.PULSE_WS);

    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("✅ WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === "pulse") {
          const data: PulseDetectionResponse = message.data;
          const meta = message.meta;

          // Update performance metrics
          if (meta) {
            setProcessingTime(meta.processing_time_ms);
            setFramesDropped(meta.frames_dropped);
          }

          const detected = Boolean(data.faceDetected);
          const locked = Boolean(data.faceLocked);

          setFaceDetected(detected);
          setFaceLocked(locked);

          // Update session status when face lock state changes
          setWebcam(true, locked);

          // Update boxes
          if (detected) {
            if (data.face) setFaceBox(data.face);
            if (data.forehead) setForeheadBox(data.forehead);
          } else {
            setFaceBox(null);
            setForeheadBox(null);
          }

          // Update calibration status
          setCalibrating(Boolean(data.calibrating));
          if (data.calibrationProgress !== undefined) {
            setCalibrationProgress(data.calibrationProgress);
          }
          if (data.calibrationTarget !== undefined) {
            setCalibrationTarget(data.calibrationTarget);
          }
          setWaitTime(data.waitTime ?? null);

          // Update BVP immediately for smooth charting
          if (typeof data.bvp === "number" && Number.isFinite(data.bvp)) {
            setCurrentBVP(data.bvp);
          }

          const now = Date.now();
          const timeSinceLastUpdate = now - lastBPMUpdateTime.current;

          if (timeSinceLastUpdate >= 250) {
            let updated = false;

            if (typeof data.bpm === "number" && Number.isFinite(data.bpm)) {
              setCurrentBPM(Number(data.bpm.toFixed(2)));
              updated = true;
            }

            if (typeof data.spo2 === "number" && Number.isFinite(data.spo2)) {
              setCurrentSpO2(Number(data.spo2.toFixed(1)));
              updated = true;
            }

            if (typeof data.rr === "number" && Number.isFinite(data.rr)) {
              setCurrentRR(Number(data.rr.toFixed(1)));
              updated = true;
            }

            if (updated) {
              console.debug("Updating UI metrics:", {
                bpm: data.bpm,
                spo2: data.spo2,
                rr: data.rr,
              });
              lastBPMUpdateTime.current = now;
            }
          }

          // Mark as not processing so next frame can be sent
          isProcessing.current = false;
        } else if (message.type === "lock_response") {
          console.log("✅ Face lock response:", message.success);
          // Don't set faceLocked here - wait for next pulse frame to confirm
        } else if (message.type === "reset_response") {
          console.log("✅ Reset response:", message.success);
        } else if (message.type === "error") {
          console.error("❌ WebSocket error:", message.message);
          isProcessing.current = false;
        } else if (message.type === "dropped") {
          // Frame was dropped
          if (message.data && typeof message.data.framesDropped === "number") {
            setFramesDropped(message.data.framesDropped);
          }
          isProcessing.current = false;
        }
      } catch (error) {
        console.error("Error parsing WebSocket message:", error);
        isProcessing.current = false;
      }
    };

    ws.onerror = (error) => {
      console.error("❌ WebSocket error:", error);
      isProcessing.current = false;
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      wsRef.current = null;
    };

    wsRef.current = ws;
  }, [setWebcam]);

  // Connect Pupil WebSocket
  const connectPupilWebSocket = useCallback(() => {
    const ws = new WebSocket(API_ENDPOINTS.PUPILLOMETRY_WS);

    ws.onopen = () => {
      console.log("✅ Pupil WebSocket connected");
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);

        if (message.type === "pupillometry") {
          const data: PupillometryResponse = message.data;

          if (data.averagePupilDiameter) {
            setAveragePupilDiameter(data.averagePupilDiameter);
          }
          if (data.leftEye && data.leftEye.detected) setLeftEye(data.leftEye);
          if (data.rightEye && data.rightEye.detected)
            setRightEye(data.rightEye);
        }
      } catch (error) {
        console.error("Error parsing Pupil WebSocket message:", error);
      }
    };

    ws.onerror = (error) => {
      console.error("❌ Pupil WebSocket error:", error);
    };

    ws.onclose = (event) => {
      console.log("Pupil WebSocket closed", event.code, event.reason);
      pupilWsRef.current = null;
    };

    pupilWsRef.current = ws;
  }, []);

  // Start webcam stream
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: availableCameras[selectedCamera]?.deviceId,
          width: { ideal: CAMERA_CONFIG.width },
          height: { ideal: CAMERA_CONFIG.height },
        },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        setWebcam(true, false); // Set webcam active with face unlocked

        // Connect WebSockets
        connectWebSocket();
        connectPupilWebSocket();
      }
    } catch (error) {
      console.error("Error accessing webcam:", error);
    }
  }, [
    selectedCamera,
    availableCameras,
    connectWebSocket,
    connectPupilWebSocket,
    setWebcam,
  ]);

  // Stop webcam stream
  const stopWebcam = useCallback(async () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.send(JSON.stringify({ type: "reset" }));
      wsRef.current.close();
      wsRef.current = null;
    }

    // Close Pupil WebSocket
    if (pupilWsRef.current) {
      pupilWsRef.current.close();
      pupilWsRef.current = null;
    }

    setIsStreaming(false);
    setFaceDetected(false);
    setFaceLocked(false);
    setCurrentBPM(0);
    setCurrentSpO2(0);
    setCurrentRR(0);
    setCurrentBVP(0);
    setFaceBox(null);
    setForeheadBox(null);
    setCalibrating(false);
    setCalibrationProgress(0);
    setWaitTime(null);
    setAveragePupilDiameter(null);
    setLeftEye(null);
    setRightEye(null);
    lastBPMUpdateTime.current = 0;
    isProcessing.current = false;
    setWebcam(false, false); // Set webcam inactive
  }, [setWebcam]);

  // Process video frames and send to WebSocket
  const processFrame = useCallback(
    function processFrameCallback() {
      if (!videoRef.current || !canvasRef.current || !isStreaming) return;

      // Skip frame if still processing previous one (only throttle pulse processing)
      if (isProcessing.current) {
        animationFrameRef.current = requestAnimationFrame(processFrameCallback);
        return;
      }

      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext("2d");

      if (!ctx) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);

      // Convert canvas to blob and send via WebSocket
      canvas.toBlob(
        (blob) => {
          if (!blob) return;

          // Send to Pulse WebSocket
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            // Mark as processing
            isProcessing.current = true;

            // Send binary frame data
            blob.arrayBuffer().then((buffer) => {
              if (
                wsRef.current &&
                wsRef.current.readyState === WebSocket.OPEN
              ) {
                wsRef.current.send(buffer);
              }
            });
          }

          // Send to Pupil WebSocket (throttled)
          const now = Date.now();
          if (
            pupilWsRef.current &&
            pupilWsRef.current.readyState === WebSocket.OPEN &&
            now - lastPupilSendTime.current > 200 // Throttle to ~5 FPS
          ) {
            lastPupilSendTime.current = now;
            blob.arrayBuffer().then((buffer) => {
              if (
                pupilWsRef.current &&
                pupilWsRef.current.readyState === WebSocket.OPEN
              ) {
                pupilWsRef.current.send(buffer);
              }
            });
          }
        },
        "image/jpeg",
        PULSE_CONFIG.frameProcessingQuality,
      );

      animationFrameRef.current = requestAnimationFrame(processFrameCallback);
    },
    [isStreaming],
  );

  // Start processing when streaming begins
  useEffect(() => {
    if (isStreaming && wsRef.current) {
      processFrame();
    }
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isStreaming, processFrame]);

  // Toggle face lock
  const toggleFaceLock = useCallback(async () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn("⚠️ WebSocket not connected");
      return;
    }

    if (faceLocked) {
      // Unlock - reset
      console.log("🔓 Sending unlock/reset command");
      wsRef.current.send(JSON.stringify({ type: "reset" }));
      setFaceLocked(false);
      setCalibrating(false);
      setCalibrationProgress(0);
      setCurrentBPM(0);
      setCurrentSpO2(0);
      setCurrentRR(0);
      setCurrentBVP(0);
      setWaitTime(null);
      setAveragePupilDiameter(null);
      setLeftEye(null);
      setRightEye(null);
      setWebcam(true, false); // Update to face unlocked
    } else {
      // Lock
      if (faceDetected) {
        console.log("🔒 Sending lock command");
        wsRef.current.send(JSON.stringify({ type: "lock" }));
      } else {
        console.warn("⚠️ No face detected - cannot lock");
      }
    }
  }, [faceLocked, faceDetected, setWebcam]);

  // Switch camera - cycle to next available camera
  const switchCamera = useCallback(async () => {
    const wasStreaming = isStreaming;
    await stopWebcam();

    // Cycle to next camera
    const nextIndex = (selectedCamera + 1) % availableCameras.length;
    setSelectedCamera(nextIndex);

    if (wasStreaming) {
      setTimeout(() => startWebcam(), 100);
    }
  }, [
    isStreaming,
    selectedCamera,
    availableCameras.length,
    stopWebcam,
    startWebcam,
  ]);

  // Sync ref with state
  useEffect(() => {
    faceLockedRef.current = faceLocked;
  }, [faceLocked]);

  return {
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
    leftEye,
    rightEye,
  };
}
