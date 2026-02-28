import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Brain,
  Loader2,
  Play,
  Square,
  Sparkles,
  AlertCircle,
} from "lucide-react";
import { AVAILABLE_MODELS } from "@/config/models";

interface ModelSelectorProps {
  selectedModel: string;
  onModelSelect: (modelId: string) => void;
  isEnabled: boolean;
  isLoading: boolean;
  onEnable: () => void;
  onDisable: () => void;
  streamingError?: string | null;
}

export function ModelSelector({
  selectedModel,
  onModelSelect,
  isEnabled,
  isLoading,
  onEnable,
  onDisable,
  streamingError,
}: ModelSelectorProps) {
  const model = AVAILABLE_MODELS.find((m) => m.id === selectedModel);
  const availableModelIds = new Set(["labram"]);
  const isModelAvailable = availableModelIds.has(selectedModel);

  return (
    <Card className="bg-white">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center gap-2">
          <Brain className="h-5 w-5" />
          Model Selection
        </CardTitle>
        <CardDescription className="text-sm">
          Choose an EEG embedding model
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Enable/Disable Button - More Prominent */}
        <Button
          className={`w-full h-12 text-base font-semibold shadow-md hover:shadow-lg cursor-pointer transition-all ${
            isEnabled ? "text-black hover:text-black" : ""
          }`}
          onClick={isEnabled ? onDisable : onEnable}
          disabled={!isModelAvailable || isLoading}
          variant={isEnabled ? "destructive" : "default"}
          size="lg"
        >
          {isLoading ? (
            <>
              <Loader2 className="mr-2 h-5 w-5 animate-spin" />
              Processing...
            </>
          ) : isEnabled ? (
            <>
              <Square className="mr-2 h-5 w-5" />
              Stop Model
            </>
          ) : (
            <>
              <Play className="mr-2 h-5 w-5" />
              Start Model
            </>
          )}
        </Button>

        {/* Streaming Error Alert */}
        {streamingError && (
          <Alert variant="destructive" className="py-2">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription className="text-sm">
              {streamingError}
            </AlertDescription>
          </Alert>
        )}

        {/* Model Dropdown */}
        <div className="space-y-2">
          <label className="text-sm font-medium text-muted-foreground">
            Select Model
          </label>
          <Select
            value={selectedModel}
            onValueChange={onModelSelect}
            disabled={isEnabled}
          >
            <SelectTrigger className="h-11 border-2 hover:border-primary/50 transition-colors bg-white dark:bg-white">
              <SelectValue placeholder="Select a model" />
            </SelectTrigger>
            <SelectContent className="bg-white dark:bg-white z-50">
              {AVAILABLE_MODELS.map((modelOption) => (
                <SelectItem
                  key={modelOption.id}
                  value={modelOption.id}
                  disabled={!availableModelIds.has(modelOption.id)}
                  className="cursor-pointer hover:bg-accent"
                >
                  <div className="flex items-center gap-3 py-1">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{modelOption.name}</span>
                        {!availableModelIds.has(modelOption.id) && (
                          <Badge variant="outline" className="text-xs">
                            Coming Soon
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Model Info */}
        {model && (
          <div
            className={`p-4 rounded-lg border-2 space-y-3 transition-colors ${
              isEnabled
                ? "bg-primary/5 border-primary/30"
                : "bg-muted/30 border-muted"
            }`}
          >
            <div className="flex items-center gap-2">
              {isEnabled && (
                <Sparkles className="h-4 w-4 text-primary animate-pulse" />
              )}
              <span className="font-semibold text-sm">{model.name}</span>
              <Badge
                variant={isEnabled ? "default" : "secondary"}
                className="ml-auto"
              >
                {isEnabled ? "Active" : "Inactive"}
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {model.description}
            </p>
            <div className="text-xs space-y-1.5 pt-2 border-t">
              <div className="flex justify-between">
                <span className="text-muted-foreground">Window size:</span>
                <span className="font-mono font-medium">
                  {model.window_size} samples
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-muted-foreground">Duration:</span>
                <span className="font-mono font-medium">
                  {(model.window_size / 250).toFixed(1)}s
                </span>
              </div>
              {model.checkpoint_path && (
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Checkpoint:</span>
                  <span className="font-mono text-xs truncate max-w-45">
                    {model.checkpoint_path.split("/").pop()}
                  </span>
                </div>
              )}
            </div>
          </div>
        )}

        {!isModelAvailable && selectedModel !== "labram" && (
          <p className="text-xs text-amber-600 bg-amber-50 p-2 rounded">
            This model is not yet available. Select LaBraM to continue.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
