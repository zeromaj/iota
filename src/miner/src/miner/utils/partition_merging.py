import asyncio
from bittensor_wallet import Keypair
from common.models.miner_models import ChunkMetadata
from common.models.api_models import CompleteFileUploadResponse, SubmittedWeightsAndOptimizerPresigned
from common.utils.exceptions import WeightPartitionException
from common.utils.partitions import MinerPartition
from common.utils.s3_utils import filter_exceptions
from loguru import logger
from miner import settings as miner_settings
from miner.utils.utils import download_metadata, extract_filename_from_url, upload_tensor
from subnet.miner_api_client import MinerAPIClient
from subnet.utils.s3_torch import download_weights_or_optimizer_state
import torch

"""
4 Metadata chunks (at weight uploading)
XXXXXXXXXXXX| XXXXXXXXXXXX| XXXXXXXXXXXX| XXXXXXXXXXXX

6 Partitions (because between weight uploading and merging new miners registered, meaning we 'think' more partitions are needed)
XXXXXXXX|XXXXXXXX|XXXXXXXX|XXXXXXXX|XXXXXXXX|XXXXXXXX

-> We get the start & end indices from the metadata, hence
Partition 1: XXXXXXXXXXXX|
Partition 2: XXXXXXXXXXXX|
Partition 3: XXXXXXXXXXXX|
Partition 4: XXXXXXXXXXXX|
Partition 5: Invalid (No metadata chunk found)
Partition 6: Invalid (No metadata chunk found)

--> The total amonunt of data is still consistent and we can just discard the invalid partitions.
"""


async def get_chunk_metadata_for_all_partitions(
    submitted_weights_and_optimizer: SubmittedWeightsAndOptimizerPresigned, partitions: list[MinerPartition]
) -> dict[int, dict[str, ChunkMetadata]]:
    """
    Returns a dictionary as follows:
    {
        0: {  <- partition number
            "weights": ChunkMetadata(...),
            "optimizer_state": ChunkMetadata(...),
        },
        ...
        7: {  <- partition number
            "weights": ChunkMetadata(...),
            "optimizer_state": ChunkMetadata(...),
        },
    }
    """
    try:
        # Download both weight and optimizer state metadata
        weight_metadata: dict = await download_metadata(
            metadata_path=submitted_weights_and_optimizer.weight_metadata_path_presigned
        )
        optimizer_state_metadata: dict = await download_metadata(
            metadata_path=submitted_weights_and_optimizer.optimizer_state_metadata_path_presigned
        )
        if not weight_metadata or not optimizer_state_metadata:
            raise Exception(
                f"No weight or optimizer state metadata found | {submitted_weights_and_optimizer.weights_path_presigned}"
            )
        if (
            "weight" not in submitted_weights_and_optimizer.weight_metadata_path_presigned
            and "weight" not in submitted_weights_and_optimizer.weights_path_presigned
        ):
            raise Exception(
                f"Weight metadata path does not contain 'weight' | {submitted_weights_and_optimizer.weights_path_presigned}"
            )
        if (
            "optimizer_state" not in submitted_weights_and_optimizer.optimizer_state_metadata_path_presigned
            and "optimizer_state" not in submitted_weights_and_optimizer.optimizer_state_path_presigned
        ):
            raise Exception(
                f"Optimizer state metadata path does not contain 'optimizer_state' | {submitted_weights_and_optimizer.optimizer_state_path_presigned}"
            )

        metadata_infos: dict[int, dict] = {}
        for partition in partitions:
            try:
                chunk_number = partition.chunk_number

                # Create separate metadata info for weights and optimizer state
                weight_metadata_info = ChunkMetadata(
                    **weight_metadata["sections"][str(chunk_number)],
                    chunk_number=chunk_number,
                    weighting_factor=submitted_weights_and_optimizer.weighting_factor,
                    tensor_path=submitted_weights_and_optimizer.weights_path_presigned,
                    metadata_path=submitted_weights_and_optimizer.weight_metadata_path_presigned,
                    chunk_dtype=weight_metadata["tensor"]["dtype"].split(".")[-1],
                    data_type="weights",
                )

                optimizer_state_metadata_info = ChunkMetadata(
                    **optimizer_state_metadata["sections"][str(chunk_number)],
                    chunk_number=chunk_number,
                    weighting_factor=submitted_weights_and_optimizer.weighting_factor,
                    tensor_path=submitted_weights_and_optimizer.optimizer_state_path_presigned,
                    metadata_path=submitted_weights_and_optimizer.optimizer_state_metadata_path_presigned,
                    chunk_dtype=optimizer_state_metadata["tensor"]["dtype"].split(".")[-1],
                    data_type="optimizer_state",
                )

                metadata_infos[chunk_number] = {
                    "weights": weight_metadata_info,
                    "optimizer_state": optimizer_state_metadata_info,
                }
            except Exception as e:
                logger.error(f"Error getting chunk metadata for partition {partition.chunk_number}: {e}")
                continue

        return metadata_infos

    except Exception as e:
        logger.exception(
            f"Error getting chunk metadata for all partitions. This is likely due to bad metadata being uploaded but is not a fatal error.: {e}"
        )
        logger.error(f"Bad metadata; WEIGHTS: {weight_metadata} | OPTIMIZER STATE: {optimizer_state_metadata}")
        return None


async def download_partition(
    weight_metadata: ChunkMetadata, optimizer_metadata: ChunkMetadata
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        assert weight_metadata.data_type == "weights", "Weights metadata is not of type weights"
        assert (
            optimizer_metadata.data_type == "optimizer_state"
        ), "Optimizer state metadata is not of type optimizer_state"
        assert (
            weight_metadata.chunk_number == optimizer_metadata.chunk_number
        ), "Weights and optimizer state chunk numbers do not match"
        assert "weight" in weight_metadata.tensor_path, "Weights tensor path does not contain 'weight'"
        assert (
            "optimizer_state" in optimizer_metadata.tensor_path
        ), "Optimizer state tensor path does not contain 'optimizer_state'"
        assert "weight" in weight_metadata.metadata_path, "Weights metadata path does not contain 'weight'"
        assert (
            "optimizer_state" in optimizer_metadata.metadata_path
        ), "Optimizer state metadata path does not contain 'optimizer_state'"

        weights: torch.Tensor = await download_weights_or_optimizer_state(
            metadata_info=weight_metadata,
        )
        optimizer_state: torch.Tensor = await download_weights_or_optimizer_state(
            metadata_info=optimizer_metadata,
        )
    except Exception as e:
        logger.error(
            f"Error download partition with weight metadata: {weight_metadata} and otpimizer metadata: {optimizer_metadata}"
        )
        logger.exception(f"Error downloading weights and optimizer state: {e}")
        raise

    return weights, optimizer_state


def metadata_matches(meta1: dict[int, dict[str, ChunkMetadata]], meta2: dict[int, dict[str, ChunkMetadata]]) -> bool:
    """Check if two metadata dictionaries match."""
    # Check if the chunk numbers are the same
    if set(meta1.keys()) != set(meta2.keys()):
        return False

    # Check that all the start/end indices agree with each other
    for chunk_number in meta1.keys():
        for data_type in ["weights", "optimizer_state"]:
            m1, m2 = meta1[chunk_number][data_type], meta2[chunk_number][data_type]
            if not m1.compatible(m2):
                return False
    return True


async def filter_bad_metadata(
    partitions: list[MinerPartition],
    submitted_weights_and_optimizers: list[SubmittedWeightsAndOptimizerPresigned],
) -> dict[str, dict[int, dict[str, ChunkMetadata]]]:
    """Filter out packets with bad metadata and return valid metadata info objects.

    Returns: dictionary mapping weight path to dictionary mapping chunk number to dictionary mapping data type to ChunkMetadata

    Partitions: 0,1,2,3,4,5,6,7
    Weights that were uploaded: 0,1,2

    Each weight (0,1,2) comes with its own metadata on s3 containing the start and end indices for each partition (0,1,2,3,4,5,6,7)

    All these metadata objects SHOULD agree with each other, but because miners upload them, they might not.

    This function now takes the most 'common' metadata object and filters out the rest.

    So in our case, the return would look as follows:

    {
        weight_0: {
            0: {
                "weights": ChunkMetadata(...),
                "optimizer_state": ChunkMetadata(...),
            },
            ...
            7: {
                "weights": ChunkMetadata(...),
                "optimizer_state": ChunkMetadata(...),
            }
        },
        ...,
        weight_2: {
            0: {
                "weights": ChunkMetadata(...),
                "optimizer_state": ChunkMetadata(...),
            },
            ...
            7: {
                "weights": ChunkMetadata(...),
                "optimizer_state": ChunkMetadata(...),
            }
        }
    }
    """
    # Collect valid metadata from all packets
    valid_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]] = {}
    results = await asyncio.gather(
        *[
            get_chunk_metadata_for_all_partitions(submitted_weights_and_optimizer=s, partitions=partitions)
            for s in submitted_weights_and_optimizers
        ]
    )
    valid_metadata = {
        s.weights_path_presigned: r
        for s, r in zip(submitted_weights_and_optimizers, results, strict=True)
        if r is not None
    }

    if not valid_metadata:
        logger.warning("No valid metadata found")
        return {}

    # Compare all metadata objects to each other and find the most common pattern
    agreement_counts: dict[str, int] = {}
    for path1, meta1 in valid_metadata.items():
        agreement_counts[path1] = sum(
            1 for path2, meta2 in valid_metadata.items() if path1 != path2 and metadata_matches(meta1, meta2)
        )

    # Keep only metadata that matches the most common pattern
    best_path = max(agreement_counts, key=agreement_counts.get)
    best_metadata = valid_metadata[best_path]

    filtered_metadata = {path: meta for path, meta in valid_metadata.items() if metadata_matches(best_metadata, meta)}

    filtered_count = len(valid_metadata) - len(filtered_metadata)
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} packets due to metadata disagreements")

    return filtered_metadata


async def get_weight_partition_info(
    layer: int,
    miner_api_client: MinerAPIClient,
) -> tuple[list[SubmittedWeightsAndOptimizerPresigned], list[MinerPartition]]:
    """
    Get the weight partition info from the orchestrator. This calls two different API endpoints:
    - /miner/get_weight_path_per_layer (weight path for the model layer)
    - /miner/get_partition_indices_by_hotkey (partition indices for the miner)

    Returns:
        tuple[list[SubmittedWeightsPresigned], list[int]]: The weight partition info and the partition ids
    """
    weight_path_per_layer: (
        list[SubmittedWeightsAndOptimizerPresigned] | dict
    ) = await miner_api_client.get_weight_path_per_layer()

    if not weight_path_per_layer:
        raise WeightPartitionException("Unknown error getting weight path per layer")

    logger.debug(f"layer {layer} getting partitions")
    partitions: dict = await miner_api_client.get_partitions()
    logger.debug(f"layer {layer} partitions: {partitions}")

    if not partitions:
        logger.warning(f"No partitions found for layer {layer}")
        return weight_path_per_layer, []

    return weight_path_per_layer, [MinerPartition(**p) for p in partitions]


async def merge_partition_batch(
    batch_partitions: list[MinerPartition],
    downloaded_partitions: list[list[tuple[torch.Tensor, torch.Tensor]]],
    filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]],
    num_metadata_chunks: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, MinerPartition]]:
    merge_results: dict[MinerPartition, tuple[torch.Tensor, torch.Tensor]] = {}
    for partition, downloaded_partition in zip(batch_partitions, downloaded_partitions):
        if num_metadata_chunks is not None:
            if partition.chunk_number >= num_metadata_chunks:
                logger.warning(
                    f"Skipping partition {partition.chunk_number} because it is invalid as it doesn't exist in the metadata chunks"
                )
                continue
        else:
            logger.warning(f"No metadata chunks found. Skipping partition {partition.chunk_number}")
        try:
            logger.debug(f"merging partition {partition.chunk_number}")

            weight_average = None
            optimizer_state_average = None
            weight_counter = 0
            optimizer_state_counter = 0

            print(f"downloaded_partition: {len(downloaded_partition)}")
            print(f"downloaded_partition[0]: {len(downloaded_partition[0])}")
            print(f"downloaded_partition[0][0]: {len(downloaded_partition[0][0])}")
            for metadata, (weights, optimizer_state) in zip(filtered_metadata.values(), downloaded_partition):
                try:
                    if weights is None or optimizer_state is None:
                        logger.warning(f"No weights or optimizer state downloaded. Partitions: {batch_partitions}")
                        raise Exception(f"No weights or optimizer state downloaded. Partitions: {batch_partitions}")

                    # TODO: We will be changing the way that weights and optimizer states are merged.
                    weights_metadata: ChunkMetadata = metadata[partition.chunk_number]["weights"]
                    optimizer_state_metadata: ChunkMetadata = metadata[partition.chunk_number]["optimizer_state"]

                    if weight_average is None:
                        weight_average = weights.to(torch.float32) * weights_metadata.weighting_factor
                        optimizer_state_average = (
                            optimizer_state.to(torch.float32) * optimizer_state_metadata.weighting_factor
                        )

                    else:
                        # create a running sum of weights weighted by the weighting factor
                        weight_average += weights.to(torch.float32) * weights_metadata.weighting_factor
                        optimizer_state_average += (
                            optimizer_state.to(torch.float32) * optimizer_state_metadata.weighting_factor
                        )

                    weight_counter += weights_metadata.weighting_factor
                    optimizer_state_counter += optimizer_state_metadata.weighting_factor

                except Exception as e:
                    logger.exception(f"Error downloading chunk {partition.chunk_number} from {metadata}: {e}")

            if weight_average is None:
                raise Exception(f"No weights downloaded. Partitions: {batch_partitions}")

            # Average the weights
            weight_average /= weight_counter
            weight_average = weight_average.to(torch.bfloat16)
            optimizer_state_average /= optimizer_state_counter
            optimizer_state_average = optimizer_state_average.to(torch.bfloat16)

            merge_results[partition.chunk_number] = (weight_average, optimizer_state_average, partition)
            return merge_results
        except Exception as e:
            logger.exception(f"Failed to get partition {partition.chunk_number}: {e}")


async def get_partition_batch(batch_index: int, partitions: list[MinerPartition]) -> list[MinerPartition]:
    start_index = batch_index * len(partitions) // min(miner_settings.N_PARTITION_BATCHES, len(partitions))
    end_index = (batch_index + 1) * len(partitions) // min(miner_settings.N_PARTITION_BATCHES, len(partitions))
    batch_partitions = partitions[start_index:end_index]
    return batch_partitions


async def download_batch_partitions(
    batch_partitions: list[MinerPartition], filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]]
) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    downloaded_partitions = []

    async def download_partition_weights(partition: MinerPartition) -> list[tuple[torch.Tensor, torch.Tensor]]:
        results: list[tuple[torch.Tensor, torch.Tensor]] = await asyncio.gather(
            *[
                download_partition(
                    weight_metadata=metadata[partition.chunk_number]["weights"],
                    optimizer_metadata=metadata[partition.chunk_number]["optimizer_state"],
                )
                for metadata in filtered_metadata.values()
            ]
        )
        logger.debug(f"Results: {len(results)}")
        logger.debug(f"Results[0]: {len(results[0])}")
        return results

    downloaded_partitions: list[list[tuple[torch.Tensor, torch.Tensor]]] = await asyncio.gather(
        *[download_partition_weights(partition=partition) for partition in batch_partitions],
        return_exceptions=True,
    )
    downloaded_partitions: list[list[tuple[torch.Tensor, torch.Tensor]]] = filter_exceptions(downloaded_partitions)
    logger.debug(f"Downloaded partitions: {len(downloaded_partitions)}")
    if len(downloaded_partitions) > 0:
        logger.debug(f"Downloaded partitions[0]: {len(downloaded_partitions[0])}")
        if len(downloaded_partitions[0]) > 0:
            logger.debug(f"Downloaded partitions[0][0]: {len(downloaded_partitions[0][0])}")
    else:
        logger.error("No partitions successfully downloaded for batch")
    return downloaded_partitions


async def upload_partition_batch(
    miner_api_client: MinerAPIClient,
    merge_results: dict[int, tuple[torch.Tensor, torch.Tensor, MinerPartition]],
    filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]],
    hotkey: Keypair,
) -> list[MinerPartition]:
    weight_uploads = []
    optimizer_state_uploads = []
    final_partitions = []
    for weight_average, optimizer_state_average, _ in merge_results.values():
        weight_uploads.append(
            upload_tensor(
                miner_api_client=miner_api_client,
                tensor=weight_average.detach().cpu(),
                file_type="weights",
                hotkey=hotkey,
            )
        )
        optimizer_state_uploads.append(
            upload_tensor(
                miner_api_client=miner_api_client,
                tensor=optimizer_state_average.detach().cpu(),
                file_type="optimizer_state",
                hotkey=hotkey,
            )
        )

    # Upload all weights at once
    weight_uploads: list[CompleteFileUploadResponse] = filter_exceptions(
        await asyncio.gather(*weight_uploads, return_exceptions=True)
    )
    optimizer_state_uploads: list[CompleteFileUploadResponse] = filter_exceptions(
        await asyncio.gather(*optimizer_state_uploads, return_exceptions=True)
    )

    weights_metadata = list(list(filtered_metadata.values())[0].values())[0]["weights"]
    optimizer_state_metadata = list(list(filtered_metadata.values())[0].values())[0]["optimizer_state"]
    for weight_upload, optimizer_state_upload, (_, _, partition) in zip(
        weight_uploads, optimizer_state_uploads, merge_results.values()
    ):
        partition.weight_path = weight_upload.object_path
        partition.optimizer_state_path = optimizer_state_upload.object_path
        partition.weight_metadata_path = extract_filename_from_url(weights_metadata.metadata_path)
        partition.optimizer_state_metadata_path = extract_filename_from_url(optimizer_state_metadata.metadata_path)

        final_partitions.append(partition)
    return [partition for partition in final_partitions if partition.is_valid()]
