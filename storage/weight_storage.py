import uuid
import time
import os
import csv
import shutil
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger

# from utils.s3_interactions import upload_weights
from utils.partitions import Partition

# Constants
WEIGHT_DIR = "weight_cache"
LOG_FILE = "weight_log.csv"
MAX_LOG_ROWS = 500


class MergedPartitionManager(BaseModel):
    partitions: list[Partition] = Field(default_factory=list)

    def get_partition(self, partition: Partition) -> Partition:
        return [p for p in self.partitions if p == partition]

    def get_layer_partitions(self, layer: int) -> list[Partition]:
        layer_partitions = []
        processed_chunk_numbers = set()
        for partition in self.partitions:
            if partition.layer == layer:
                if partition.chunk_number not in processed_chunk_numbers:
                    layer_partitions.append(partition)
                processed_chunk_numbers.add(partition.chunk_number)
        if not layer_partitions:
            logger.error(f"No partitions found for layer: {layer}, available partitions: {self.partitions}")
        return layer_partitions

    def update_layer_partitions(self, layer: int, new_partitions: list[Partition]):
        for partition in reversed(self.partitions):
            if partition.layer == layer:
                self.partitions.remove(partition)

        for partition in new_partitions:
            self.partitions.append(partition)


def cleanup_weight_cache():
    """Clean up the weight cache directory and log file."""
    if os.path.exists(WEIGHT_DIR):
        shutil.rmtree(WEIGHT_DIR)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)


class WeightStore(BaseModel):
    # Dictionary of miner hotkeys to weight file paths
    weights: dict[str, str] = Field(default_factory=dict)  # Maps miner_hotkey to weight file path
    layer_weights: dict[int, Partition] = Field(default_factory=dict)  # Maps layer number to weight file path
    store_uid: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=lambda: time.time())
    model_config = ConfigDict(arbitrary_types_allowed=True)
    merged_partition_manager: MergedPartitionManager = Field(default_factory=MergedPartitionManager)

    def __init__(self, **data):
        # Clean up any existing cache and logs
        cleanup_weight_cache()
        super().__init__(**data)
        # Initialize new log file
        self._init_log_file()

    def _init_log_file(self):
        os.makedirs(WEIGHT_DIR, exist_ok=True)
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "operation", "miner_hotkey"])

    def _log_operation(self, operation: str, miner_hotkey: str):
        # Read existing rows
        rows = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = list(reader)

        # Add new row
        rows.append([time.time(), operation, miner_hotkey])

        # Keep only last MAX_LOG_ROWS
        rows = rows[-MAX_LOG_ROWS:]

        # Write back to file
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "operation", "miner_hotkey"])
            writer.writerows(rows)

    async def upload_weights_and_optimizer(
        self,
        miner_hotkey: str,
        weights_path: str,
        weight_metadata_path: str,
        optimizer_state_path: str,
        optimizer_state_metadata_path: str,
    ):
        """Upload weights for a specific miner."""
        self.weights[miner_hotkey] = (
            weights_path,
            weight_metadata_path,
            optimizer_state_path,
            optimizer_state_metadata_path,
        )

    async def get_layer_partitions(self, layer: int) -> list[Partition]:
        return self.merged_partition_manager.get_layer_partitions(layer)

    async def set_layer_partitions(self, layer: int, partitions: list[Partition]):
        logger.debug(f"Setting layer {layer} partitions: {partitions}")
        self.merged_partition_manager.update_layer_partitions(layer, partitions)
        logger.debug(f"Partitions set: {self.merged_partition_manager.get_layer_partitions(layer)}")

    async def get_miner_weights(self, miner_hotkey: str) -> str:
        """Retrieve weights for a specific miner."""
        if miner_hotkey not in self.weights:
            raise KeyError(f"No weights found for miner {miner_hotkey}")

        file_path = Path(self.weights[miner_hotkey])
        self._log_operation("download", miner_hotkey)
        return file_path

    async def list_miners(self) -> list[str]:
        """List all miners who have uploaded weights."""
        self._log_operation("list", "all")
        return list(self.weights.keys())

    def __del__(self):
        # Cleanup any remaining weight files
        if os.path.exists(WEIGHT_DIR):
            for file in Path(WEIGHT_DIR).glob("*.pt"):
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to delete weight file {file}: {e}")

    async def reset_all_miner_weights(self):
        self.weights = {}
        self.layer_weights = {}
