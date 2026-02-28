import type { Model } from "@/types/classification";

export type ModelType = "eeg" | "pupillometry" | "bvp";

export interface TypedModel extends Model {
  type: ModelType;
}

export const AVAILABLE_MODELS: TypedModel[] = [
  // EEG Models
  {
    id: "labram",
    name: "LaBraM",
    description:
      "Large Brain Model - Pretrained transformer for EEG embeddings",
    checkpoint_path:
      "eeg/models/classification/pretrained/labram/labram-base.pth",
    window_size: 1600,
    type: "eeg",
  },
  {
    id: "eeg-transformer-1",
    name: "EEG Transformer v1",
    description: "Transformer-based EEG classification model",
    checkpoint_path:
      "eeg/models/classification/pretrained/labram/labram-base.pth",
    window_size: 1000,
    type: "eeg",
  },

  // Pupillometry Models
  {
    id: "pupil-mlp-1",
    name: "Pupil MLP",
    description: "Multi-layer perceptron for pupil diameter analysis",
    checkpoint_path: "eeg/models/classification/pupillometry/mlp-v1.pth",
    window_size: 500,
    type: "pupillometry",
  },
  {
    id: "pupil-lstm-1",
    name: "Pupil LSTM",
    description: "LSTM network for temporal pupil patterns",
    checkpoint_path: "eeg/models/classification/pupillometry/lstm-v1.pth",
    window_size: 800,
    type: "pupillometry",
  },
  {
    id: "pupil-transformer-1",
    name: "Pupil Transformer",
    description: "Transformer for advanced pupillometry analysis",
    checkpoint_path:
      "eeg/models/classification/pupillometry/transformer-v1.pth",
    window_size: 600,
    type: "pupillometry",
  },

  // BVP/Heart Rate Models
  {
    id: "bvp-resnet-1",
    name: "BVP ResNet",
    description: "ResNet-based blood volume pulse analysis",
    checkpoint_path: "eeg/models/classification/bvp/resnet-v1.pth",
    window_size: 750,
    type: "bvp",
  },
  {
    id: "bvp-lstm-1",
    name: "BVP LSTM",
    description: "LSTM for heart rate variability analysis",
    checkpoint_path: "eeg/models/classification/bvp/lstm-v1.pth",
    window_size: 900,
    type: "bvp",
  },
  {
    id: "hr-cnn-1",
    name: "Heart Rate CNN",
    description: "CNN for heart rate pattern recognition",
    checkpoint_path: "eeg/models/classification/bvp/hr-cnn-v1.pth",
    window_size: 1000,
    type: "bvp",
  },
  {
    id: "ppg-transformer-1",
    name: "PPG Transformer",
    description: "Transformer for photoplethysmography signal processing",
    checkpoint_path: "eeg/models/classification/bvp/ppg-transformer-v1.pth",
    window_size: 850,
    type: "bvp",
  },
];

// Helper functions to get models by type
export const getModelsByType = (type: ModelType): TypedModel[] => {
  return AVAILABLE_MODELS.filter((model) => model.type === type);
};

export const getEEGModels = () => getModelsByType("eeg");
export const getPupillometryModels = () => getModelsByType("pupillometry");
export const getBVPModels = () => getModelsByType("bvp");
