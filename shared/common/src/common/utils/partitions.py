from typing import Literal
from common.models.miner_models import MetadataInfo
from pydantic import BaseModel
from itertools import combinations

import random


def assign_cells_to_pairs(miner_hotkeys: list[str], n_partitions: int) -> dict[int, tuple[str, str]]:
    """Assigns cells to pairs of miners. This is used to assign partitions to miners for butterfly all reduce merging.

    Args:
        miner_hotkeys (list[str]): The list of miner hotkeys.
        n_partitions (int): The number of partitions to assign.

    Returns:
        dict[int, tuple[str, str]]: A dictionary of partition numbers to pairs of miner hotkeys.
    """
    if len(miner_hotkeys) == 1:
        return {0: (miner_hotkeys[0], None)}

    pairs = list(combinations(miner_hotkeys, 2))

    MINIMUM_PARTITIONS_PER_MINER = 2
    miner_counts = {miner: MINIMUM_PARTITIONS_PER_MINER for miner in miner_hotkeys}

    for _ in range(max(0, n_partitions - len(pairs))):
        # Find miners with the lowest count
        selected_miners = []
        for _ in range(MINIMUM_PARTITIONS_PER_MINER):
            min_count = min(miner_counts.values())
            lowest_count_miners = [miner for miner, count in miner_counts.items() if count == min_count]
            # Select two miners randomly from those with the lowest count
            selected_miner = random.choice(lowest_count_miners)
            selected_miners.append(selected_miner)
            miner_counts[selected_miner] += 1

        pairs.append(tuple(selected_miners))

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
    metadata: MetadataInfo,
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
    if chunk_id == "all":
        chunk_start_idx = metadata.start_idx
        chunk_end_idx = metadata.end_idx
        chunk_start_byte = metadata.start_byte
        chunk_end_byte = metadata.end_byte
    else:
        chunk_start_idx = metadata.start_idx
        chunk_end_idx = metadata.end_idx
        chunk_start_byte = metadata.start_byte
        chunk_end_byte = metadata.end_byte

    chunk_data = ChunkData(
        chunk_start_idx=chunk_start_idx,
        chunk_end_idx=chunk_end_idx,
        chunk_start_byte=chunk_start_byte,
        chunk_end_byte=chunk_end_byte,
        chunk_dtype=metadata.chunk_dtype,
        chunk_length=chunk_end_idx - chunk_start_idx,
    )

    return chunk_data
