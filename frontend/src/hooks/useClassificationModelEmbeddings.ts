import { useState, useEffect, useCallback } from "react";
import { API_ENDPOINTS } from "@/config/api";

interface EmbeddingShape {
  batch_size: number;
  num_channels: number;
  num_patches: number;
  vocab_size?: number;
  reduced_dims?: number;
}

interface Embedding {
  raw: {
    shape: EmbeddingShape;
  };
  reduced: {
    embeddings_2d: [number, number];
    shape: EmbeddingShape;
  };
}

export function useEmbeddings() {
  const [isEnabled, setIsEnabled] = useState(false);
  const [latestEmbedding, setLatestEmbedding] = useState<Embedding | null>(
    null,
  );
  const [embeddingHistory, setEmbeddingHistory] = useState<Embedding[]>([]);
  const [error, setError] = useState<string | null>(null);

  const enableEmbeddings = useCallback(
    async (
      channelNames: string[],
      channelMapping: Record<string, string>,
      checkpointPath: string,
    ) => {
      try {
        console.log(
          `[Embeddings] Enabling with ${channelNames.length} channels:`,
          channelNames,
        );
        console.log("[Embeddings] Channel mapping:", channelMapping);
        console.log("[Embeddings] Checkpoint path:", checkpointPath);

        const response = await fetch(API_ENDPOINTS.MODEL_CONFIGURE, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            enabled: true,
            checkpoint_path: checkpointPath,
            channel_names: channelNames,
            channel_mapping: channelMapping,
          }),
        });

        if (response.ok) {
          setIsEnabled(true);
          console.log("✓ Embeddings enabled");
          return true;
        } else {
          const data = await response.json();
          setError(data.error || "Failed to enable embeddings");
          return false;
        }
      } catch (e) {
        console.error("Failed to connect to backend:", e);
        return false;
      }
    },
    [],
  );

  const disableEmbeddings = useCallback(async () => {
    try {
      const response = await fetch(API_ENDPOINTS.MODEL_CONFIGURE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: false }),
      });

      if (response.ok) {
        setIsEnabled(false);
        console.log("✓ Embeddings disabled");
        return true;
      }
      return false;
    } catch (e) {
      console.error("Failed to connect to backend:", e);
      return false;
    }
  }, []);

  const fetchLatestEmbedding = useCallback(async () => {
    if (!isEnabled) return null;

    try {
      const response = await fetch(API_ENDPOINTS.MODEL_LATEST);

      if (response.ok) {
        const data = await response.json();
        setLatestEmbedding(data);
        return data;
      }
      return null;
    } catch (e) {
      console.error("Failed to fetch embedding:", e);
      return null;
    }
  }, [isEnabled]);

  const fetchEmbeddingHistory = useCallback(
    async (n?: number) => {
      if (!isEnabled) return [];

      try {
        const url = n
          ? `${API_ENDPOINTS.MODEL_HISTORY}?n=${n}`
          : API_ENDPOINTS.MODEL_HISTORY;
        const response = await fetch(url);

        if (response.ok) {
          const data = await response.json();
          setEmbeddingHistory(data.embeddings || []);
          return data.embeddings;
        }
        return [];
      } catch (e) {
        console.error("Failed to fetch embedding history:", e);
        return [];
      }
    },
    [isEnabled],
  );

  // Poll for new embeddings when enabled
  useEffect(() => {
    if (!isEnabled) return;

    let interval: number | null = null;

    // Wait 3 seconds before starting to poll to let the model warm up
    const initialDelay = setTimeout(() => {
      interval = setInterval(() => {
        fetchLatestEmbedding();
        fetchEmbeddingHistory(100);
      }, 2000); // Poll every 2 seconds
    }, 3000);

    return () => {
      clearTimeout(initialDelay);
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isEnabled, fetchLatestEmbedding, fetchEmbeddingHistory]);

  const clearEmbeddings = useCallback(() => {
    setEmbeddingHistory([]);
    setLatestEmbedding(null);
  }, []);

  return {
    isEnabled,
    latestEmbedding,
    embeddingHistory,
    error,
    enableEmbeddings,
    disableEmbeddings,
    fetchLatestEmbedding,
    fetchEmbeddingHistory,
    clearEmbeddings,
  };
}
