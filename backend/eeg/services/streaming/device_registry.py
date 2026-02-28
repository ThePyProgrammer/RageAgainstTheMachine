"""
Registry of supported BCI devices with hardware specifications.
"""

DEVICE_REGISTRY = {
    "cyton": {
        "name": "OpenBCI Cyton",
        "eeg_channel_count": 8,
        "eeg_channel_names": ["1", "2", "3", "4", "5", "6", "7", "8"],
        "aux_channel_count": 0,
        "aux_channel_names": [],
        "accel_channel_count": 3,
        "analog_channel_count": 3,
        "sampling_rate": 250,
        "max_uv": (4.5 / 24) * 1_000_000,  # ADS1299: ±187,500 µV
        "railed_threshold_percent": 0.90,
        "near_railed_threshold_percent": 0.75,
        "streamer": "brainflow",
    },
    "muse_v1": {
        "name": "Muse v1",
        "eeg_channel_count": 4,
        "eeg_channel_names": ["TP9", "AF7", "AF8", "TP10"],
        "aux_channel_count": 1,
        "aux_channel_names": ["AUX"],
        "accel_channel_count": 0,
        "analog_channel_count": 0,
        "sampling_rate": 256,
        "max_uv": 1682.815,  # Muse AFE ±1682 µV range
        "railed_threshold_percent": 0.90,
        "near_railed_threshold_percent": 0.75,
        "streamer": "muse_lsl",
        "lsl_stream_type": "EEG",
    },
}


def get_device_config(device_type: str) -> dict:
    """Get device configuration by type. Raises ValueError if unknown."""
    if device_type not in DEVICE_REGISTRY:
        raise ValueError(
            f"Unknown device type: '{device_type}'. "
            f"Available: {list(DEVICE_REGISTRY.keys())}"
        )
    return DEVICE_REGISTRY[device_type]


def get_available_devices() -> list[dict]:
    """Return list of available devices with their names and types."""
    return [
        {"type": key, "name": cfg["name"]}
        for key, cfg in DEVICE_REGISTRY.items()
    ]
