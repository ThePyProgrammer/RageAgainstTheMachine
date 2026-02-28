/* eslint-disable react-refresh/only-export-components */
import { createContext, useContext, useState, useEffect } from "react";
import type { ReactNode } from "react";

interface SessionStatus {
  eegStreaming: boolean;
  bfmProcessor: boolean;
  webcam: boolean;
  faceLocked: boolean;
}

interface SessionStatusContextType {
  status: SessionStatus;
  setEEGStreaming: (active: boolean) => void;
  setBFMProcessor: (active: boolean) => void;
  setWebcam: (active: boolean, faceLocked?: boolean) => void;
}

const SessionStatusContext = createContext<
  SessionStatusContextType | undefined
>(undefined);
const STORAGE_KEY = "bci_session_status";

export function SessionStatusProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<SessionStatus>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try {
          return JSON.parse(stored);
        } catch (e) {
          console.error("Failed to parse stored session status:", e);
        }
      }
    }
    return {
      eegStreaming: false,
      bfmProcessor: false,
      webcam: false,
      faceLocked: false,
    };
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(status));
    }
  }, [status]);

  const setEEGStreaming = (active: boolean) => {
    setStatus((prev) => ({ ...prev, eegStreaming: active }));
  };

  const setBFMProcessor = (active: boolean) => {
    setStatus((prev) => ({ ...prev, bfmProcessor: active }));
  };

  const setWebcam = (active: boolean, faceLocked?: boolean) => {
    setStatus((prev) => ({
      ...prev,
      webcam: active,
      faceLocked: faceLocked !== undefined ? faceLocked : prev.faceLocked,
    }));
  };

  return (
    <SessionStatusContext.Provider
      value={{ status, setEEGStreaming, setBFMProcessor, setWebcam }}
    >
      {children}
    </SessionStatusContext.Provider>
  );
}

export function useSessionStatus() {
  const context = useContext(SessionStatusContext);
  if (context === undefined) {
    throw new Error(
      "useSessionStatus must be used within a SessionStatusProvider",
    );
  }
  return context;
}
