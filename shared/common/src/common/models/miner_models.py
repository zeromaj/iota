from enum import Enum

from pydantic import BaseModel


class MinerStatus(str, Enum):
    """Enumeration of possible miner statuses."""

    FORWARD = "forward"
    BACKWARD = "backward"
    IDLE = "idle"
    UPLOADING_WEIGHTS = "uploading_weights"
    MERGING_PARTITIONS = "merging_partitions"


class MetadataInfo(BaseModel):
    start_idx: int
    end_idx: int
    start_byte: int
    end_byte: int
    chunk_number: int
    weighting_factor: int | None = None
    weight_path: str | None = None
    weight_metadata_path: str | None = None
    optimizer_state_path: str | None = None
    optimizer_state_metadata_path: str | None = None
    chunk_dtype: str
