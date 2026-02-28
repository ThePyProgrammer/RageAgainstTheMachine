/* eslint-disable react-refresh/only-export-components */
import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
} from "react";
import type { ReactNode } from "react";
import type { ElectrodeMapping } from "@/types/eeg";

interface ElectrodeMappingContextType {
  activeChannel: string | null;
  setActiveChannel: (channel: string | null) => void;
  mapping: ElectrodeMapping;
  handleMappingChange: (electrode: string, channel: string) => void;
  handleRemoveMapping: (electrode: string) => void;
  handleElectrodeClick: (electrodeId: string, isConfigMode: boolean) => void;
}

const ElectrodeMappingContext = createContext<
  ElectrodeMappingContextType | undefined
>(undefined);
const STORAGE_KEY = "bci_electrode_mapping";

export function ElectrodeMappingProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [activeChannel, setActiveChannel] = useState<string | null>(null);
  const [mapping, setMapping] = useState<ElectrodeMapping>(() => {
    if (typeof window !== "undefined") {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        try {
          return JSON.parse(stored);
        } catch (e) {
          console.error("Failed to parse stored electrode mapping:", e);
        }
      }
    }
    return {};
  });

  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(mapping));
    }
  }, [mapping]);

  const handleMappingChange = useCallback(
    (electrode: string, channel: string) => {
      setMapping((prev) => {
        const next: ElectrodeMapping = { ...prev };
        Object.keys(next).forEach((key) => {
          if (next[key] === channel) delete next[key];
        });
        next[electrode] = channel;
        return next;
      });
      setActiveChannel(null);
    },
    [],
  );

  const handleRemoveMapping = useCallback((electrode: string) => {
    setMapping((prev) => {
      const next: ElectrodeMapping = { ...prev };
      delete next[electrode];
      return next;
    });
  }, []);

  const handleElectrodeClick = useCallback(
    (electrodeId: string, isConfigMode: boolean) => {
      if (!isConfigMode || !activeChannel) return;
      handleMappingChange(electrodeId, activeChannel);
    },
    [activeChannel, handleMappingChange],
  );

  return (
    <ElectrodeMappingContext.Provider
      value={{
        activeChannel,
        setActiveChannel,
        mapping,
        handleMappingChange,
        handleRemoveMapping,
        handleElectrodeClick,
      }}
    >
      {children}
    </ElectrodeMappingContext.Provider>
  );
}

export function useElectrodeMappingContext() {
  const context = useContext(ElectrodeMappingContext);
  if (context === undefined) {
    throw new Error(
      "useElectrodeMappingContext must be used within ElectrodeMappingProvider",
    );
  }
  return context;
}
