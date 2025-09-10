import asyncio
import os

from common import settings as common_settings
from common.models.miner_models import ChunkMetadata
from common.utils.exceptions import NanInfWarning
from common.utils.partitions import MinerPartition, format_chunk_data
from common.utils.s3_utils import download_file
from common.utils.s3_utils import filter_exceptions
import torch

import json

from loguru import logger
from subnet.utils.vector_utils import check_for_nans_and_infs
from subnet.utils.s3_torch import download_tensor


def get_cosine_similarity(old_shard: torch.Tensor, new_shard: torch.Tensor) -> float:
    norm_old = torch.norm(old_shard)
    norm_new = torch.norm(new_shard)
    cosine_similarity_weights = torch.cosine_similarity(norm_old, norm_new, dim=0)
    return cosine_similarity_weights


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
    layer: int = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Downloads the weights and optimizer state for a given list of partitions.

    Args:
        merged_partitions (list[MinerPartition]): The merged partitions to download weights and optimizer state for
        weights (torch.Tensor): The current weights to download weights and optimizer state for
        optimizer_state (torch.Tensor): The current optimizer state to download weights and optimizer state for
            It will be flattened and then downloaded as a single tensor.
        device (str): The device to download weights and optimizer state for

    Returns:
        tuple[torch.Tensor, torch.Tensor] | None: The weights and optimizer state for the layer
    """
    partition_download_error_counter: int = 0
    total_parts: int = len(merged_partitions) if merged_partitions else 0

    if not merged_partitions:
        logger.warning("No merged partitions found❗️")
        return None, None

    try:
        # Check for nans and infs in the weights
        check_for_nans_and_infs(weights, "current weights", exception_type=NanInfWarning)

        # Check for nans and infs in the optimizer state
        check_for_nans_and_infs(
            optimizer_state,
            "current optimizer state for miner",
            exception_type=NanInfWarning,
        )

        num_weights = weights.numel()
        num_optimizer_state = optimizer_state.numel()

        total_weights_downloaded: int = 0
        total_optimizer_state_downloaded: int = 0

        # Build all partition data first
        new_partitions: list[MinerPartition] = []
        for partition in merged_partitions:
            new_partition: MinerPartition = await build_chunk_data(partition=partition)
            new_partitions.append(new_partition)

        # Process partitions in batches of DOWNLOAD_BATCH_SIZE
        logger.info(
            f"Starting batched download of {len(new_partitions)} weight/optimizer pairs in batches of {common_settings.DOWNLOAD_BATCH_SIZE}"
        )

        # Batch the downloads (same number for weights and optimizer state, so it's technically 2 * DOWNLOAD_BATCH_SIZE)
        for batch_start in range(0, len(new_partitions), common_settings.DOWNLOAD_BATCH_SIZE):
            batch_end = min(batch_start + common_settings.DOWNLOAD_BATCH_SIZE, len(new_partitions))
            batch_partitions = new_partitions[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start // common_settings.DOWNLOAD_BATCH_SIZE + 1}/{(len(new_partitions) + common_settings.DOWNLOAD_BATCH_SIZE - 1) // common_settings.DOWNLOAD_BATCH_SIZE}: partitions {batch_start} to {batch_end - 1}"
            )

            try:
                downloaded_tensors_weights = await asyncio.gather(
                    *[
                        download_tensor(partition.weight_path, device="cpu", dtype=torch.bfloat16)
                        for partition in batch_partitions
                    ],
                    return_exceptions=True,
                )
                downloaded_tensors_optimizer_state = await asyncio.gather(
                    *[
                        download_tensor(partition.optimizer_state_path, device="cpu", dtype=torch.bfloat16)
                        for partition in batch_partitions
                    ],
                    return_exceptions=True,
                )
                downloaded_tensors_weights, downloaded_tensors_optimizer_state = filter_exceptions(
                    downloaded_tensors_weights, downloaded_tensors_optimizer_state
                )

                # Process downloaded tensors and apply to model
                # Tensors come in pairs: [weight0, opt0, weight1, opt1, ...]
                for weight_shard, optimizer_state_shard, partition in zip(
                    downloaded_tensors_weights, downloaded_tensors_optimizer_state, batch_partitions
                ):
                    # If either of the downloads fail, we want to discard the shards.
                    if isinstance(weight_shard, Exception) or isinstance(optimizer_state_shard, Exception):
                        partition_download_error_counter += 1
                        continue

                    weight_state_start_idx = partition.weight_data.chunk_start_idx
                    weight_state_end_idx = partition.weight_data.chunk_end_idx
                    optimizer_state_start_idx = partition.optimizer_state_data.chunk_start_idx
                    optimizer_state_end_idx = partition.optimizer_state_data.chunk_end_idx

                    try:
                        cosine_similarity_weight_shard = get_cosine_similarity(
                            old_shard=weights[weight_state_start_idx:weight_state_end_idx].clone().to(device),
                            new_shard=weight_shard,
                        )
                        cosine_similarity_optimizer_shard = get_cosine_similarity(
                            old_shard=optimizer_state[optimizer_state_start_idx:optimizer_state_end_idx]
                            .clone()
                            .to(device),
                            new_shard=optimizer_state_shard,
                        )

                        logger.debug(
                            f"Cosine similarities for partition chunk {partition.chunk_number} for layer {layer} for weight shard: {cosine_similarity_weight_shard:.4f} optimizer shard: {cosine_similarity_optimizer_shard:.4f}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error calculating cosine similarity for partition chunk {partition.chunk_number} for layer {layer} for shard: {e}"
                        )

                    logger.debug(
                        f"Weight shard shape: {weight_shard.shape}, expected start idx: {weight_state_start_idx}, expected end idx: {weight_state_end_idx}, range: {weight_state_end_idx - weight_state_start_idx}"
                    )
                    weights[weight_state_start_idx:weight_state_end_idx] = weight_shard

                    logger.debug(
                        f"Shard optimizer state shape: {optimizer_state_shard.shape}, expected start idx: {optimizer_state_start_idx}, expected end idx: {optimizer_state_end_idx}, range: {optimizer_state_end_idx - optimizer_state_start_idx}"
                    )
                    optimizer_state[optimizer_state_start_idx:optimizer_state_end_idx] = optimizer_state_shard

                    total_weights_downloaded += weight_shard.numel()
                    total_optimizer_state_downloaded += optimizer_state_shard.numel()

                    logger.debug(
                        f"Applied partition {partition.chunk_number}: weights[{weight_state_start_idx}:{weight_state_end_idx}] optimizer[{optimizer_state_start_idx}:{optimizer_state_end_idx}]"
                    )

            except Exception as e:
                logger.warning(
                    f"Error in batched download for batch {batch_start // common_settings.DOWNLOAD_BATCH_SIZE + 1}: {e}"
                )
                partition_download_error_counter += common_settings.DOWNLOAD_BATCH_SIZE
                continue

        logger.debug(
            f"Downloaded {total_parts - partition_download_error_counter} / {total_parts} partitions inside download_and_set_weights_and_optimizer_state"
        )
        logger.info(
            f"download_and_set_weights_and_optimizer_state downloaded {total_weights_downloaded} / {num_weights} ({(total_weights_downloaded / num_weights) * 100}%) weights and {total_optimizer_state_downloaded} / {num_optimizer_state} ({(total_optimizer_state_downloaded / num_optimizer_state) * 100}%) optimizer state"
        )

        # Cast the model weights and optimizer state to the correct device.
        weights: torch.Tensor = weights.to(device)
        optimizer_state: torch.Tensor = optimizer_state.to(device)
        return weights, optimizer_state

    except Exception:
        raise


def _model_suffix(hotkey: str, run_id: str, layer_idx: int) -> str:
    return f"{hotkey[:8]}_{run_id}_{layer_idx}"


def save_model_weights_and_optimizer_state(
    model_weights: torch.Tensor, optimizer_state_dict: dict, hotkey: str, run_id: str, layer_idx: int
):
    """Saves the model weights and optimizer state to the weights directory."""

    logger.debug(f"Saving model weights and optimizer state for hotkey {hotkey[:8]}")

    model_suffix = _model_suffix(hotkey=hotkey, run_id=run_id, layer_idx=layer_idx)
    try:
        os.makedirs("./weights", exist_ok=True)
        torch.save(model_weights, f"./weights/current_model_weights_{model_suffix}.pt")
        torch.save(optimizer_state_dict, f"./weights/current_model_optimizer_state_dict_{model_suffix}.pt")

        # Remove files from previous runs
        for file in os.listdir("./weights"):
            if model_suffix not in file:
                os.remove(f"./weights/{file}")
    except Exception as e:
        logger.error(f"Error saving model weights and optimizer state: {e}")


def load_model_weights_and_optimizer_state(hotkey: str, run_id: str, layer_idx: int) -> tuple[torch.Tensor, dict]:
    """Loads the model weights and optimizer state from the weights directory."""

    try:
        model_suffix = _model_suffix(hotkey=hotkey, run_id=run_id, layer_idx=layer_idx)
        logger.debug(f"Loading model weights and optimizer state for suffix {model_suffix}")
        model_weights = torch.load(f"./weights/current_model_weights_{model_suffix}.pt")
        optimizer_state_dict = torch.load(f"./weights/current_model_optimizer_state_dict_{model_suffix}.pt")
        return model_weights, optimizer_state_dict

    except Exception as e:
        if not os.path.exists("./weights"):
            logger.info("📂 Weights directory doesn't exist, creating it to save model weights and optimizer states!")
        else:
            logger.warning(
                f"❌ Attempted to load saved model weights and optimizer state, but it doesn't exist. If epoch is > 1, it's excepted that you have these saved. It's possible that you will diverge and lose incentive if you don't have these: {e}"
            )

        return None, None
