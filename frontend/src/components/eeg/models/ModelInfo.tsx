import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Info } from "lucide-react";

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

interface ModelEmbeddingInfoProps {
  embedding: Embedding;
}

export function ModelEmbeddingInfo({ embedding }: ModelEmbeddingInfoProps) {
  return (
    <Card className="bg-white max-h-[300px] overflow-y-auto">
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Info className="h-4 w-4" />
          Latest Embedding
        </CardTitle>
      </CardHeader>
      <CardContent className="text-xs space-y-3">
        <div>
          <div className="font-semibold mb-1 text-blue-600">Raw Embedding</div>
          <div className="space-y-1 pl-2 border-l-2 border-blue-200">
            <div className="flex justify-between">
              <span className="text-muted-foreground">Channels:</span>
              <span className="font-mono">
                {embedding.raw.shape.num_channels}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Patches:</span>
              <span className="font-mono">
                {embedding.raw.shape.num_patches}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Vocab size:</span>
              <span className="font-mono">
                {embedding.raw.shape.vocab_size}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Total dims:</span>
              <span className="font-mono text-xs">
                {embedding.raw.shape.num_channels *
                  embedding.raw.shape.num_patches *
                  (embedding.raw.shape.vocab_size || 0)}
              </span>
            </div>
          </div>
        </div>

        <div>
          <div className="font-semibold mb-1 text-red-600">Reduced (t-SNE)</div>
          <div className="space-y-1 pl-2 border-l-2 border-red-200">
            <div className="flex justify-between">
              <span className="text-muted-foreground">2D Point:</span>
              <span className="font-mono text-xs">
                [{embedding.reduced.embeddings_2d[0].toFixed(3)},
                {embedding.reduced.embeddings_2d[1].toFixed(3)}]
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Dimensions:</span>
              <span className="font-mono">
                {embedding.reduced.shape.reduced_dims}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
