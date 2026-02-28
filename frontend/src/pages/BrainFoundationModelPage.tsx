import { useState } from "react";
import { useEmbeddings } from "@/hooks/useClassificationModelEmbeddings";
import { useBCIStream } from "@/hooks/useBCIStream";
import { useElectrodeMappingContext } from "@/contexts/ElectrodeMappingContext";
import { EmbeddingScatterPlot } from "@/components/eeg/EmbeddingScatterPlot";
import { ModelSelector } from "@/components/eeg/models/ModelSelector";
import { ModelEmbeddingInfo } from "@/components/eeg/models/ModelInfo";
import { AVAILABLE_MODELS } from "@/config/models";

export default function ModelSelectionPage() {
  const {
    latestEmbedding,
    embeddingHistory,
    isEnabled,
    enableEmbeddings,
    disableEmbeddings,
  } = useEmbeddings();
  const { isStreaming } = useBCIStream();
  const { mapping } = useElectrodeMappingContext();
  const [selectedModel, setSelectedModel] = useState("labram");
  const [isLoading, setIsLoading] = useState(false);
  const [streamingError, setStreamingError] = useState<string | null>(null);

  const handleEnableModel = async () => {
    // Check if streaming is running
    if (!isStreaming) {
      setStreamingError(
        "BCI streaming must be running first. Please start streaming on the BCI Dashboard.",
      );
      return;
    }

    // Validate that at least one electrode is mapped to a channel
    const mappedElectrodes = Object.keys(mapping);
    if (mappedElectrodes.length === 0) {
      setStreamingError(
        "No electrodes mapped! Please map at least one electrode to a channel using the 3D head plot on the BCI Dashboard before starting the model.",
      );
      return;
    }

    // Validation passed - proceed with enabling
    setStreamingError(null);
    setIsLoading(true);
    try {
      // Get the full checkpoint path from the model configuration
      const model = AVAILABLE_MODELS.find((m) => m.id === selectedModel);
      const checkpointPath = model?.checkpoint_path || selectedModel;

      await enableEmbeddings(mappedElectrodes, mapping, checkpointPath);
    } catch (error) {
      setStreamingError(
        error instanceof Error ? error.message : "Failed to enable embeddings",
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleDisableModel = async () => {
    setIsLoading(true);
    await disableEmbeddings();
    setIsLoading(false);
  };

  const handleModelSelect = (modelId: string) => {
    // Only allow model selection when disabled
    if (!isEnabled) {
      setSelectedModel(modelId);
    }
  };

  return (
    <div className="h-full flex gap-4 p-4">
      {/* Visualization Panel */}
      <div className="flex-1">
        <EmbeddingScatterPlot
          embeddings={embeddingHistory}
          latest={latestEmbedding}
        />
      </div>

      {/* Control Panel */}
      <aside className="w-85 flex flex-col gap-4 overflow-y-auto">
        <ModelSelector
          selectedModel={selectedModel}
          onModelSelect={handleModelSelect}
          isEnabled={isEnabled}
          isLoading={isLoading}
          onEnable={handleEnableModel}
          onDisable={handleDisableModel}
          streamingError={streamingError}
        />

        {latestEmbedding && <ModelEmbeddingInfo embedding={latestEmbedding} />}
      </aside>
    </div>
  );
}
