/* eslint-disable react-refresh/only-export-components */
import {
    createContext,
    useContext,
    useState,
    useCallback,
    useEffect,
} from "react";
import type { ReactNode } from "react";
import { DEVICE_CONFIGS } from "@/config/eeg";
import type { DeviceType, DeviceConfig } from "@/config/eeg";

interface DeviceContextType {
    deviceType: DeviceType;
    deviceConfig: DeviceConfig;
    setDeviceType: (type: DeviceType) => void;
}

const DeviceContext = createContext<DeviceContextType | undefined>(undefined);
const STORAGE_KEY = "bci_device_type";

export function DeviceProvider({ children }: { children: ReactNode }) {
    const [deviceType, setDeviceTypeRaw] = useState<DeviceType>(() => {
        if (typeof window !== "undefined") {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored && stored in DEVICE_CONFIGS) {
                return stored as DeviceType;
            }
        }
        return "cyton";
    });

    const deviceConfig = DEVICE_CONFIGS[deviceType];

    useEffect(() => {
        if (typeof window !== "undefined") {
            localStorage.setItem(STORAGE_KEY, deviceType);
        }
    }, [deviceType]);

    const setDeviceType = useCallback((type: DeviceType) => {
        setDeviceTypeRaw(type);
    }, []);

    return (
        <DeviceContext.Provider value={{ deviceType, deviceConfig, setDeviceType }}>
            {children}
        </DeviceContext.Provider>
    );
}

export function useDevice() {
    const context = useContext(DeviceContext);
    if (context === undefined) {
        throw new Error("useDevice must be used within a DeviceProvider");
    }
    return context;
}
