import { useCallback, useEffect, useMemo, useState } from "react";
import type { UiSettings } from "@/features/pong/types/pongSettings";
import { defaultSettings, loadSettings, saveSettings } from "./settingsManager";

export const usePongSettings = () => {
  const [settings, setSettings] = useState<UiSettings>(defaultSettings);

  useEffect(() => {
    setSettings(loadSettings());
  }, []);

  const updateSettings = useCallback((next: UiSettings) => {
    setSettings(next);
    saveSettings(next);
  }, []);

  const updatePartial = useCallback((updater: (previous: UiSettings) => UiSettings) => {
    setSettings((previous) => {
      const next = updater(previous);
      saveSettings(next);
      return next;
    });
  }, []);

  const resetSettings = useCallback(() => {
    const next = defaultSettings();
    setSettings(next);
    saveSettings(next);
  }, []);

  return useMemo(
    () => ({
      settings,
      updateSettings,
      updatePartial,
      resetSettings,
    }),
    [settings, updateSettings, updatePartial, resetSettings],
  );
};
