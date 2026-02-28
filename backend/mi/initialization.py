"""Motor Imagery module initialization and startup routines."""

import logging
import os
from pathlib import Path
from typing import Optional
import warnings

import torch
import yaml

# Suppress MNE verbosity during imports and data loading
os.environ["MNE_LOGGING_LEVEL"] = "ERROR"
import mne

mne.set_log_level("ERROR")

from mi.models.eegnet import EEGClassifier, EEGNet
from mi.models.eegnet_residual import EEGClassifier as ResidualEEGClassifier
from mi.models.eegnet_residual import EEGNetResidual
from mi.services.mi_controller import MotorImageryController, load_test_data
from mi.utils.config_loader import get_project_root

logger = logging.getLogger(__name__)

# Global state - shared with routes
mi_controller: Optional[MotorImageryController] = None
test_data_X = None
test_data_y = None


def load_mi_config():
    """Load MI configuration from YAML file."""
    backend_root = get_project_root()
    config_path = backend_root / "mi" / "config" / "eeg_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_model_exists(config: dict) -> Path:
    """Ensure trained model checkpoint exists.

    Args:
        config: MI configuration dictionary

    Returns:
        Path to model file
    """
    backend_root = get_project_root()
    model_path = (
        backend_root / config["training"]["savedir"] / config["training"]["savename"]
    )

    if not model_path.exists():
        neuralflight_root = backend_root / "external" / "NeuralFlight"
        fallback_path = (
            neuralflight_root
            / config["training"]["savedir"]
            / config["training"]["savename"]
        )
        if fallback_path.exists():
            logger.info(
                "Model not found at %s. Using NeuralFlight checkpoint at %s",
                model_path,
                fallback_path,
            )
            return fallback_path
        raise FileNotFoundError(
            f"Model not found at {model_path} or {fallback_path}. Train the model first."
        )
    logger.info(f"Model found at {model_path}")

    return model_path


def preload_test_data(config: dict) -> tuple:
    """Preload test data during startup.

    Args:
        config: MI configuration dictionary

    Returns:
        Tuple of (X, y) test data
    """
    logger.info("Loading test EEG data (subject 6)...")

    # Suppress warnings during data loading
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = load_test_data(config)

    return X, y


def initialize_mi_controller(config: dict, model_path: Path) -> MotorImageryController:
    """Initialize the MI controller with model and config.

    Args:
        config: MI configuration dictionary
        model_path: Path to model checkpoint

    Returns:
        Initialized MotorImageryController
    """
    logger.info("Initializing Motor Imagery Controller...")

    # Calculate n_samples
    n_samples = int(
        (config["epochs"]["tmax"] - config["epochs"]["tmin"])
        * config["preprocessing"]["sampling_rate"]
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    use_residual = bool(config["model"].get("use_residual", False))
    use_attention = bool(config["model"].get("use_attention", False))
    architecture_name = "EEGNetResidual" if use_residual else "EEGNet"
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", {})
        model_cfg = checkpoint.get("model_config")
        if model_cfg:
            n_channels = model_cfg["n_channels"]
            n_classes = model_cfg["n_classes"]
            n_samples = model_cfg["n_samples"]
            logger.info(
                "Checkpoint found with %d classes (model trained with this many outputs)",
                n_classes,
            )
        else:
            n_channels = config["model"]["input_channels"]
            # Use 2 classes to match NeuralFlight training (left/right only)
            n_classes = 2
            logger.warning(
                "No model_config in checkpoint, defaulting to 2 classes (left/right)"
            )

        # Auto-detect architecture from checkpoint if possible.
        if "fc.weight" in state_dict and "fc1.weight" not in state_dict:
            use_residual = False
            architecture_name = "EEGNet"
        elif "fc1.weight" in state_dict:
            use_residual = True
            architecture_name = "EEGNetResidual"

        logger.info("Using %s architecture for MI model", architecture_name)

        if use_residual:
            model = EEGNetResidual(
                n_channels=n_channels,
                n_classes=n_classes,
                n_samples=n_samples,
                dropout=config["model"].get("dropout", 0.5),
                kernel_length=config["model"].get("kernel_length", 64),
                use_attention=use_attention,
            )
            classifier = ResidualEEGClassifier(model, device=device)
        else:
            model = EEGNet(
                n_channels=n_channels,
                n_classes=n_classes,
                n_samples=n_samples,
                dropout=config["model"].get("dropout", 0.5),
                kernel_length=config["model"].get("kernel_length", 64),
            )
            classifier = EEGClassifier(model, device=device)
    else:
        raise RuntimeError(f"Model checkpoint not found at {model_path}")

    classifier.load(str(model_path))

    # NeuralFlight trains with 2 classes (left/right) even though config has 4
    actual_classes = (
        classifier.model.fc.out_features
        if hasattr(classifier.model, "fc")
        else (
            classifier.model.fc2.out_features if hasattr(classifier.model, "fc2") else 2
        )
    )

    logger.info("Model has %d output classes", actual_classes)

    # Map only the classes that exist in the trained model
    full_command_mapping = config["class_to_command"]
    command_mapping = {
        k: v for k, v in full_command_mapping.items() if k < actual_classes
    }

    full_label_mapping = {
        0: "Left Hand",
        1: "Right Hand",
        2: "Feet",
        3: "Rest",
    }
    label_mapping = {k: v for k, v in full_label_mapping.items() if k < actual_classes}

    controller = MotorImageryController(
        classifier, command_mapping, label_mapping=label_mapping
    )

    logger.info("Motor Imagery Controller initialized")
    return controller


def initialize():
    """Initialize MI module during app startup."""
    global mi_controller, test_data_X, test_data_y

    try:
        logger.info("Initializing Motor Imagery module...")

        # Load configuration
        logger.info("Loading MI configuration...")
        config = load_mi_config()
        logger.info("Configuration loaded")

        # Ensure model exists (create if needed)
        logger.info("Checking model...")
        model_path = ensure_model_exists(config)
        logger.info(f"Model ready at {model_path}")

        # Preload test data (this triggers download if needed)
        logger.info("Loading test data...")
        test_data_X, test_data_y = preload_test_data(config)
        logger.info(f"Test data loaded: {len(test_data_X)} epochs")

        # Initialize controller
        logger.info("Initializing controller...")
        mi_controller = initialize_mi_controller(config, model_path)
        logger.info("Controller initialized")

        logger.info("Motor Imagery module initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing Motor Imagery module: {e}", exc_info=True)
        # Don't raise - allow app to start even if MI fails
        logger.warning("Motor Imagery module will be unavailable")


def get_controller() -> Optional[MotorImageryController]:
    """Get the initialized MI controller.

    Returns:
        MI controller instance or None if not initialized
    """
    return mi_controller


def get_test_data() -> tuple:
    """Get the preloaded test data.

    Returns:
        Tuple of (X, y) or (None, None) if not loaded
    """
    return test_data_X, test_data_y
