from typing import Literal
from pydantic import BaseModel, Field
from loguru import logger
import random
from itertools import combinations

from common.models.api_models import SubmittedWeights


def assign_cells_to_pairs(miner_hotkeys: list[str]):
    if len(miner_hotkeys) == 1:
        return {0: (miner_hotkeys[0], None)}
    pairs = list(combinations(miner_hotkeys, 2))
    random.shuffle(pairs)
    return {i: pairs[i] for i in range(len(pairs))}


class ChunkData(BaseModel):
    chunk_start_idx: int | None = None
    chunk_end_idx: int | None = None
    chunk_start_byte: int | None = None
    chunk_end_byte: int | None = None
    chunk_dtype: str | None = "bfloat16"
    chunk_length: int | None = None


class MinerPartition(BaseModel):
    layer: int | None = None
    weight_data: ChunkData = ChunkData()
    optimizer_state_data: ChunkData = ChunkData()
    chunk_number: int | Literal["all"] = None
    miner_hotkey: str | None = None
    weight_path: str | None = None
    weight_metadata_path: str | None = None
    optimizer_state_path: str | None = None
    optimizer_state_metadata_path: str | None = None
    other_miner_hotkey: str | None = None

    def __eq__(self, other: "MinerPartition"):
        return (
            self.layer == other.layer
            and self.chunk_number == other.chunk_number
            and self.miner_hotkey == other.miner_hotkey
        )


async def format_chunk_data(
    metadata: dict,
    chunk_id: int | str,
) -> ChunkData:
    """Download a chunk from the database for a miner that will be used for butterfly all reduce merging.

    Args:
        metadata (dict): The metadata for the tensor.
        chunk_id (int | str): The chunk id

    Returns:
        ChunkData: The chunk data.
    """

    if isinstance(chunk_id, str):
        assert chunk_id == "all"

    # get the chunk in the metadata with the correct chunk_id
    available_chunk_ids = list(int(k) for k in metadata["sections"].keys())
    if chunk_id == "all":
        chunk_start_idx = metadata["sections"][str(min(available_chunk_ids))]["start_idx"]
        chunk_end_idx = metadata["sections"][str(max(available_chunk_ids))]["end_idx"]
        chunk_start_byte = metadata["sections"][str(min(available_chunk_ids))]["start_byte"]
        chunk_end_byte = metadata["sections"][str(max(available_chunk_ids))]["end_byte"]
    else:
        chunk = metadata["sections"][str(chunk_id)]
        chunk_start_idx = chunk["start_idx"]
        chunk_end_idx = chunk["end_idx"]
        chunk_start_byte = chunk["start_byte"]
        chunk_end_byte = chunk["end_byte"]

    chunk_data = ChunkData(
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
        chunk_start_byte=chunk_start_byte,
        chunk_end_byte=chunk_end_byte,
        chunk_dtype=metadata["tensor"]["dtype"].split(".")[-1],
        chunk_length=chunk_end_idx - chunk_start_idx,
    )

    return chunk_data


class PartitionManager(BaseModel):
    """Class for managing partitions for a given layer.

    It handles:
    - Creating partitions for a given layer
    - Getting the chunks for a miner in a given layer
    - Getting the partition paths for a given layer
    - Resetting the partition manager for a given layer
    - Getting the partition for a given miner
    - Getting the layer partitions for a given layer

    Attributes:
        partitions (list[Partition]): A list of partitions for the given layer.
        original_weight_paths (dict[int, list[SubmittedWeights]]): A dictionary of original weight paths for the given layer.
    """

    partitions: list[MinerPartition] = Field(default_factory=list)
    original_weight_paths: dict[int, list[SubmittedWeights]] = Field(default_factory=dict)

    def reset_partition_manager(self, layer: int):
        for partition in reversed(self.partitions):
            if partition.layer == layer:
                self.partitions.remove(partition)

        self.original_weight_paths[layer] = []

    def get_partition_for_miner(self, hotkey: str):
        return [p for p in self.partitions if p.miner_hotkey == hotkey]

    def get_layer_partitions(self, layer: int, completed_only: bool = False):
        if completed_only:
            return [
                p
                for p in self.partitions
                if p.layer == layer and p.weight_path is not None and p.optimizer_state_path is not None
            ]
        else:
            return [p for p in self.partitions if p.layer == layer]

    def create_partition_mappings(
        self,
        submitted_weights: dict[str, tuple[str, str, str, str]],
        layer: int,
        registry: "orchestrator.MinerRegistry",
    ):
        """Create a mapping of miner hotkeys to their assigned partitions.

        Args:
            submitted_weights (dict[str, tuple[str, str]]): A dictionary of miner hotkeys to their submitted weights and metadata.
            layer (int): The layer to create the partition mappings for.
        """
        miner_hotkeys = list(submitted_weights.keys())

        for miner_hotkey, submitted_weight in submitted_weights.items():
            try:
                self.original_weight_paths[layer].append(
                    SubmittedWeights(
                        weights_path=submitted_weight[0],
                        weight_metadata_path=submitted_weight[1],
                        optimizer_state_path=submitted_weight[2],
                        optimizer_state_metadata_path=submitted_weight[3],
                        hotkey=miner_hotkey,
                        weighting_factor=1 + registry.get_miner_data(miner_hotkey).backwards_since_reset,
                    )
                )
            except Exception as e:
                logger.warning(f"Error adding submitted weights for miner {miner_hotkey}: {e}")
                continue

        pairs = assign_cells_to_pairs(miner_hotkeys=miner_hotkeys)
        randomize_order = [0, 1]
        random.shuffle(randomize_order)

        for i, pair in pairs.items():
            if pair[randomize_order[0]] is not None:
                self.partitions.append(
                    MinerPartition(
                        layer=layer,
                        miner_hotkey=pair[randomize_order[0]],
                        other_miner_hotkey=pair[randomize_order[1]],
                        chunk_number=i,
                    )
                )
            if pair[randomize_order[1]] is not None:
                self.partitions.append(
                    MinerPartition(
                        layer=layer,
                        miner_hotkey=pair[randomize_order[1]],
                        other_miner_hotkey=pair[randomize_order[0]],
                        chunk_number=i,
                    )
                )

    def get_partition(self, partition: MinerPartition) -> MinerPartition:
        partitions = [p for p in self.partitions if p == partition]
        if not partitions:
            raise ValueError(f"No partition found for {partitions}")
        if len(partitions) > 1:
            raise ValueError(f"Multiple partitions found for {partitions}")
        return partitions[0]

    def get_chunks_for_miner(self, hotkey: str, layer: int) -> tuple[list[SubmittedWeights], list[int]]:
        """Get the chunks for a miner in a given layer.

        Args:
            hotkey (str): The hotkey of the miner that made the request.
            layer (int): The layer to get chunks for.

        Raises:
            ValueError: If no chunks are found for the miner.

        Returns:
            tuple[list[SubmittedWeights], list[int]]: The original weight paths and chunk numbers for the miner.
        """

        chunk_numbers = [p.chunk_number for p in self.partitions if p.miner_hotkey == hotkey]

        if not chunk_numbers:
            raise ValueError(f"No chunks found for miner {hotkey}")

        return self.original_weight_paths[layer], chunk_numbers

    def get_partition_paths(self, layer: int):
        partition_paths = {}
        for partition in self.get_layer_partitions(layer):
            partition_paths[partition.chunk_number] = partition.weight_path
            partition_paths[partition.chunk_number] = partition.optimizer_state_path
        return partition_paths
