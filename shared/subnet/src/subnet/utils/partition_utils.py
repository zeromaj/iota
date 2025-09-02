import os

from common.models.miner_models import ChunkMetadata
from common.utils.exceptions import NanInfWarning
from common.utils.partitions import MinerPartition, format_chunk_data
from common.utils.s3_utils import download_file
import torch

import json

from loguru import logger
from subnet.utils.s3_torch import download_tensor
from subnet.utils.vector_utils import check_for_nans_and_infs


async def build_chunk_data(partition: MinerPartition) -> MinerPartition:
    """Builds the chunk data for a partition by downloading the metadata and then creating a ChunkMetadata object.

    Args:
        partition (MinerPartition): The partition to build the chunk data for

    Returns:
        MinerPartition: The partition with the chunk data
    """
    weight_metadata_bytes: bytes = await download_file(presigned_url=partition.weight_metadata_path)
    weight_metadata: dict = json.loads(weight_metadata_bytes)

    optimizer_state_metadata_bytes: bytes = await download_file(presigned_url=partition.optimizer_state_metadata_path)
    optimizer_state_metadata: dict = json.loads(optimizer_state_metadata_bytes)

    wm = weight_metadata["sections"][str(partition.chunk_number)]
    om = optimizer_state_metadata["sections"][str(partition.chunk_number)]

    weight_metadata = ChunkMetadata(
        start_idx=wm["start_idx"],
        end_idx=wm["end_idx"],
        start_byte=wm["start_byte"],
        end_byte=wm["end_byte"],
        chunk_dtype=weight_metadata["tensor"]["dtype"].split(".")[-1],
        tensor_path=partition.weight_path,
        metadata_path=partition.weight_metadata_path,
        chunk_number=partition.chunk_number,
        data_type="weights",
    )
    optimizer_state_metadata = ChunkMetadata(
        start_idx=om["start_idx"],
        end_idx=om["end_idx"],
        start_byte=om["start_byte"],
        end_byte=om["end_byte"],
        chunk_dtype=optimizer_state_metadata["tensor"]["dtype"].split(".")[-1],
        tensor_path=partition.optimizer_state_path,
        metadata_path=partition.optimizer_state_metadata_path,
        chunk_number=partition.chunk_number,
        data_type="optimizer_state",
    )

    # Set the chunk data for the partition
    partition.weight_data = await format_chunk_data(metadata=weight_metadata, chunk_id=partition.chunk_number)
    partition.optimizer_state_data = await format_chunk_data(
        metadata=optimizer_state_metadata, chunk_id=partition.chunk_number
    )

    return partition


async def download_partitions(
    merged_partitions: list[MinerPartition],
    weights: torch.Tensor,
    optimizer_state: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Downloads the weights for a given layer and device.

    Miners/validators have to download the shards of the final merged layer weights. These weights exist across multiple files and are 1D byte strings

    Args:
        layer_idx (int): The layer index to download weights for
        device (str): The device to download weights for
        parser (Callable, optional): A parser function to parse the response from the API. Defaults to None.
        epoch (int, optional): The epoch to download weights for. Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: The weights and optimizer state for the layer
    """
    partition_download_error_counter: int = 0

    try:
        total_parts = len(merged_partitions) if merged_partitions else 0

        if not merged_partitions:
            logger.warning("No merged partitions found❗️")
            return None, None

        logger.info(f"Preparing to download {total_parts} merged partitions")

        # Allocate memory to the full 1d tensor, clone to avoid modifying the original weights in place.

        check_for_nans_and_infs(weights, "current weights for miner", exception_type=NanInfWarning)

        check_for_nans_and_infs(
            optimizer_state,
            "current optimizer state for miner",
            exception_type=NanInfWarning,
        )

        num_weights = weights.numel()
        num_optimizer_state = optimizer_state.numel()

        total_weights_downloaded = 0
        total_optimizer_state_downloaded = 0

        for idx, partition in enumerate(merged_partitions):
            logger.info(f"Downloading merged partition {idx + 1}/{total_parts} (chunk {partition.chunk_number})")
            new_partition: MinerPartition = await build_chunk_data(partition=partition)

            try:
                weight_shard = await download_tensor(
                    path=new_partition.weight_path,
                    device=device,
                )

                shard_optimizer_state = await download_tensor(
                    path=new_partition.optimizer_state_path,
                    device=device,
                )

                weight_state_start_idx = new_partition.weight_data.chunk_start_idx
                weight_state_end_idx = new_partition.weight_data.chunk_end_idx
                optimizer_state_start_idx = new_partition.optimizer_state_data.chunk_start_idx
                optimizer_state_end_idx = new_partition.optimizer_state_data.chunk_end_idx

                logger.debug(
                    f"Weight shard shape: {weight_shard.shape}, expected start idx: {weight_state_start_idx}, expected end idx: {weight_state_end_idx}, range: {weight_state_end_idx-weight_state_start_idx}"
                )
                weights[weight_state_start_idx:weight_state_end_idx] = weight_shard

                logger.debug(
                    f"Shard optimizer state shape: {shard_optimizer_state.shape}, expected start idx: {optimizer_state_start_idx}, expected end idx: {optimizer_state_end_idx}, range: {optimizer_state_end_idx-optimizer_state_start_idx}"
                )
                optimizer_state[optimizer_state_start_idx:optimizer_state_end_idx] = shard_optimizer_state

                total_weights_downloaded += weight_shard.numel()
                total_optimizer_state_downloaded += shard_optimizer_state.numel()

                logger.debug(
                    f"Applied partition {partition.chunk_number}: weights[{weight_state_start_idx}:{weight_state_end_idx}] optimizer[{optimizer_state_start_idx}:{optimizer_state_end_idx}]"
                )

            except Exception as e:
                logger.warning(f"Error downloading partition {partition}: {e}")
                partition_download_error_counter += 1

        logger.debug(
            f"Downloaded {total_parts - partition_download_error_counter} / {total_parts} partitions inside download_and_set_weights_and_optimizer_state for hotkey"
        )
        logger.info(
            f"download_and_set_weights_and_optimizer_state downloaded {total_weights_downloaded} / {num_weights} ({(total_weights_downloaded/num_weights)*100}%) weights and {total_optimizer_state_downloaded} / {num_optimizer_state} ({(total_optimizer_state_downloaded/num_optimizer_state)*100}%) optimizer state"
        )

        logger.success("✅ Successfully downloaded and applied weights and optimizer state to model")
        return weights, optimizer_state

    except Exception:
        raise


def save_model_weights_and_optimizer_state(
    model_weights: torch.Tensor, optimizer_state_dict: dict, hotkey: str, run_id: str
):
    """Saves the model weights and optimizer state to the weights directory."""

    logger.debug(f"Saving model weights and optimizer state for hotkey {hotkey[:8]}")

    try:
        os.makedirs("./weights", exist_ok=True)
        torch.save(model_weights, f"./weights/current_model_weights_{hotkey[:8]}_{run_id}.pt")
        torch.save(optimizer_state_dict, f"./weights/current_model_optimizer_state_dict_{hotkey[:8]}_{run_id}.pt")

        # Remove files from previous runs
        for file in os.listdir("./weights"):
            if "run_id" not in file:
                os.remove(f"./weights/{file}")
    except Exception as e:
        logger.error(f"Error saving model weights and optimizer state: {e}")


def load_model_weights_and_optimizer_state(hotkey: str, run_id: str) -> tuple[torch.Tensor, dict]:
    """Loads the model weights and optimizer state from the weights directory."""

    try:
        logger.debug(f"Loading model weights and optimizer state for hotkey {hotkey[:8]}")
        model_weights = torch.load(f"./weights/current_model_weights_{hotkey[:8]}_{run_id}.pt")
        optimizer_state_dict = torch.load(f"./weights/current_model_optimizer_state_dict_{hotkey[:8]}_{run_id}.pt")
        return model_weights, optimizer_state_dict

    except Exception as e:
        if not os.path.exists("./weights"):
            logger.info("📂 Weights directory doesn't exist, creating it to save model weights and optimizer states!")
        else:
            logger.warning(
                f"❌ Attempted to load saved model weights and optimizer state, but it doesn't exist. If epoch is > 1, it's excepted that you have these saved. It's possible that you will diverge and lose incentive if you don't have these: {e}"
            )

        return None, None
