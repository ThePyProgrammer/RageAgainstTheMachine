from pydantic import BaseModel, Field
from typing import Optional


class EEGChunk(BaseModel):
    """EEG data chunk from streaming."""

    data: list[list[float]]


class Session(BaseModel):
    """EEG streaming session."""

    chunks: list[EEGChunk] = Field(default_factory=list)
    session_id: Optional[str] = None


class EmbeddingConfig(BaseModel):
    """Configuration for enabling embeddings."""

    enabled: bool
    checkpoint_path: str = Field(
        default="eeg/models/classification/pretrained/labram/labram-base.pth"
    )
    channel_names: Optional[list[str]] = Field(default=None)
    channel_mapping: Optional[dict[str, str]] = Field(
        default=None
    )  # {electrode_id: channel_id}
