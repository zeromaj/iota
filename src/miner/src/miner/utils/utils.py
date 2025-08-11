import json
import time
from typing import Literal, Optional
from asyncio.exceptions import TimeoutError

import numpy as np
from subnet.utils.s3_torch import download_weights_or_optimizer_state
import torch
from bittensor_wallet import Keypair
from common import settings as common_settings
from common.models.api_models import (
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    FileUploadResponse,
)
from common.utils.partitions import ChunkData, MinerPartition, format_chunk_data
from common.utils.s3_utils import download_file
from loguru import logger
from subnet.miner_api_client import MinerAPIClient


async def create_metadata(weights_tensor: torch.Tensor, num_sections: int) -> dict:
    """Create metadata for a tensor.

    Args:
        weights_tensor (torch.Tensor): The tensor to create metadata for.
        num_sections (int): The number of sections to split the tensor into.

    Returns:
        dict: The metadata for the tensor.
    """
    # Create metadata about the tensor
    tensor_metadata = {
        "dtype": str(weights_tensor.dtype),
        "size": weights_tensor.size(),
        "num_elements": weights_tensor.numel(),
        "element_size": weights_tensor.itemsize,
        "total_bytes": weights_tensor.nbytes,  # this is just weights_tensor.numel() * weights_tensor.itemsize
    }

    # Number of sections to split into (in elements, or indices)
    section_size = weights_tensor.numel() // num_sections
    # Create section metadata
    sections_metadata = {}

    for i in range(int(num_sections)):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else weights_tensor.numel()

        # Calculate corresponding tensor indices
        start_byte = start_idx * tensor_metadata["element_size"]
        end_byte = end_idx * tensor_metadata["element_size"]

        assert start_byte is not None and end_byte is not None, "Start byte and end byte are missing"
        assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"

        sections_metadata[i] = {
            "start_byte": start_byte,
            "end_byte": end_byte,
            "start_idx": start_idx,  # e.g for a (100,100) matrix divided into 10 sections, the indices are: 0-999. 1000 - 1999. 2000 - 2999. 3000 - 3999. 4000 - 4999. 5000 - 5999. 6000 - 6999. 7000 - 7999. 8000 - 8999. 9000 - 9999.
            "end_idx": end_idx,
        }

    # Save full tensor metadata
    full_metadata = {"tensor": tensor_metadata, "sections": sections_metadata}

    return full_metadata


async def download_chunk_of_model(
    layer: int,
    miner_hotkey: str,
    weights_path: str,
    metadata_path: str,
    chunk_id: int | str,
    data_type: Literal["weights", "optimizer_state"],
    data_path: str,
) -> tuple[torch.Tensor, dict]:
    """Download a chunk of the model for a miner.

    Args:
        layer (int): The layer of the model.
        miner_hotkey (str): The hotkey of the miner.
        metadata_path (str): The path to the metadata.
        chunk_id (int | str): The chunk id.
        data_type (Literal["weights", "optimizer_state"]): The type of data to download.
        layer (int): The layer of the model.
        data_path (str): The path to the data.

    Returns:
        tuple[torch.Tensor, dict]: The downloaded chunk and the metadata.
    """

    # download the metadata
    try:
        start_time = time.time()
        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | downloading metadata from {metadata_path}")
        metadata_bytes: bytes = await download_file(presigned_url=metadata_path)
        metadata: dict = json.loads(metadata_bytes)
        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | metadata: {metadata}")

        assert isinstance(metadata, dict), f"Metadata is not a dict: {type(metadata)}"

        # format the chunk data
        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | formatting chunk data")
        chunk_data: ChunkData = await format_chunk_data(metadata=metadata, chunk_id=chunk_id)

        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | chunk_data: {chunk_data}")

        partition = MinerPartition(
            layer=layer,
            chunk_number=chunk_id,
            weight_path=weights_path,
            weight_metadata_path=metadata_path,
            miner_hotkey=miner_hotkey,
            weight_data=chunk_data if data_type == "weights" else ChunkData(),
            optimizer_state_data=chunk_data if data_type == "optimizer_state" else ChunkData(),
        )
        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | partition: {partition}")

        # only download the chunk we need: we want to form an s3 query which includes the start and end indices
        weights = await download_weights_or_optimizer_state(path=data_path, partition=partition, data_type=data_type)
        logger.debug(f"Miner {miner_hotkey[:8]} | layer {layer} | weights: {weights}")

        return weights, metadata
    except TimeoutError as e:
        logger.error(
            f"Timeout error downloading chunk of model. Time taken: {time.time() - start_time} seconds. Started download at {start_time}."
        )
        raise
    except Exception as e:
        logger.exception(f"Error downloading chunk of model: {e}")
        raise


async def upload_file(
    hotkey: Keypair,
    data: bytes,
    file_type: Literal["weights", "optimizer_state", "weights_metadata", "optimizer_state_metadata"],
    file_upload_response: Optional[FileUploadResponse] = None,
) -> str | dict:
    """
    Uploads a file to the orchestrator. To upload, we need to:
    1. Initiate a file upload by getting a FileUploadResponse from the orchestrator
    2. Upload the data using the presigned urls
    3. Complete the file upload

    Args:
        hotkey (Keypair): The hotkey of the miner.
        data (bytes): The data to upload
        file_type (Literal["weights", "optimizer_state"]): The type of file to upload
        file_upload_response (Optional[FileUploadResponse], optional): The response from the orchestrator. Defaults to None.

    Raises:
        ValueError: If the number of parts is greater than the maximum number of parts
        e: If there is an error uploading the file

    Returns:
        str: The path to the uploaded file
    """
    # TODO: We may want to set this to a more optimal value, for now we just make each part 10MB
    try:
        num_parts = int(np.ceil(len(data) / common_settings.MAX_PART_SIZE))
        if num_parts > common_settings.MAX_NUM_PARTS:
            raise ValueError(
                f"Number of parts must be less than {common_settings.MAX_NUM_PARTS}. Your file with {len(data)} bytes doesn't fit within {common_settings.MAX_NUM_PARTS} part of 10MB each"
            )

        if file_upload_response is None:
            # Get presigned urls from orchestrator
            file_upload_response: FileUploadResponse | dict = await MinerAPIClient.initiate_file_upload_request(
                hotkey=hotkey,
                file_upload_request=FileUploadRequest(file_type=file_type, num_parts=num_parts),
            )

            # Need to return to check the parsing of the response
            if isinstance(file_upload_response, dict):
                return file_upload_response

        # Upload data to presigned urls
        parts: list[dict] = await MinerAPIClient.upload_multipart_to_s3(
            urls=file_upload_response.urls, data=data, upload_id=file_upload_response.upload_id
        )

        # Complete file upload. Necessary to notify orchestrator that all parts have been uploaded.
        complete_file_upload_response: CompleteFileUploadResponse | dict = (
            await MinerAPIClient.complete_file_upload_request(
                hotkey=hotkey,
                file_upload_completion_request=FileUploadCompletionRequest(
                    object_name=file_upload_response.object_name,
                    upload_id=file_upload_response.upload_id,
                    parts=parts,
                ),
            )
        )

        if isinstance(complete_file_upload_response, dict):
            return complete_file_upload_response

        return complete_file_upload_response.object_path

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise
