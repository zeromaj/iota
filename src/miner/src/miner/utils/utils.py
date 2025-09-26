import asyncio
import json
import gzip
from urllib.parse import urlparse

from typing import Literal, Optional

from common.utils.cache import async_lru
from common.utils.exceptions import LayerStateException, NanInfException
from common.utils.formulas import calculate_num_parts
from common.utils.shared_states import LayerPhase
from subnet.utils.vector_utils import check_for_nans_and_infs
import torch
from bittensor_wallet import Keypair
from common import settings as common_settings
from common.models.api_models import (
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    FileUploadResponse,
)
from common.utils.s3_utils import download_file
from loguru import logger
from subnet.miner_api_client import MinerAPIClient
from common.models.run_flags import RUN_FLAGS


# OBSOLETE
async def get_start_and_end_indices(tensor_length: int, num_sections: int, target_section: int) -> tuple[int, int]:
    """Get the start and end indices for a tensor.

    Args:
        tensor_length (int): The length of the tensor to get the start and end indices for.
        num_sections (int): The number of sections to split the tensor into.
        target_section (int): The target section to get the start and end indices for.

    Returns:
        tuple[int, int]: The start and end indices for the target section.
    """
    assert target_section < num_sections, "Target section is greater than the number of sections"
    section_size = tensor_length // num_sections
    for i in range(int(min(target_section + 1, num_sections))):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else tensor_length
        assert start_idx is not None and end_idx is not None, "Start idx and end idx are missing"
    return start_idx, end_idx


async def create_metadata(tensor: torch.Tensor, num_sections: int) -> dict:
    """Create metadata for a tensor.

    Args:
        weights_tensor (torch.Tensor): The tensor to create metadata for.
        num_sections (int): The number of sections to split the tensor into.

    Returns:
        dict: The metadata for the tensor.
    """
    # Create metadata about the tensor
    tensor_metadata = {
        "dtype": str(tensor.dtype),
        "size": tensor.size(),
        "num_elements": tensor.numel(),
        "element_size": tensor.itemsize,
        "total_bytes": tensor.nbytes,  # this is just weights_tensor.numel() * weights_tensor.itemsize
    }

    # Number of sections to split into (in elements, or indices)
    section_size = tensor.numel() // num_sections
    # Create section metadata
    sections_metadata = {}

    for i in range(int(num_sections)):
        start_idx = i * section_size
        end_idx = start_idx + section_size if i < num_sections - 1 else tensor.numel()

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


@async_lru(maxsize=5000)
async def download_metadata(metadata_path: str) -> dict:
    """Download metadata from a presigned url.

    Args:
        metadata_path (str): The path to the metadata.

    Returns:
        dict: The metadata.
    """
    metadata_bytes: bytes = await download_file(presigned_url=metadata_path)
    if len(metadata_bytes) > 1_000_000:
        logger.warning(f"Metadata is too large: {len(metadata_bytes)} bytes")
        raise ValueError(f"Metadata is too large: {len(metadata_bytes)} bytes")

    metadata: dict = json.loads(metadata_bytes)
    return metadata


async def upload_file(
    miner_api_client: MinerAPIClient,
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
        miner_api_client (MinerAPIClient): The miner API client.
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
        num_parts = calculate_num_parts(data=data)
        if num_parts > common_settings.MAX_NUM_PARTS:
            raise ValueError(
                f"Number of parts must be less than {common_settings.MAX_NUM_PARTS}. Your file with {len(data)} bytes doesn't fit within {common_settings.MAX_NUM_PARTS} part of 10MB each"
            )

        if file_upload_response is None:
            # Get presigned urls from orchestrator
            file_upload_response: FileUploadResponse | dict = await miner_api_client.initiate_file_upload_request(
                hotkey=hotkey,
                file_upload_request=FileUploadRequest(file_type=file_type, num_parts=num_parts),
            )

            # Need to return to check the parsing of the response
            if isinstance(file_upload_response, dict):
                return file_upload_response

        if RUN_FLAGS.compress_s3_files.isOn():
            data = gzip.compress(data)

        # Upload data to presigned urls
        parts: list[dict] = await MinerAPIClient.upload_multipart_to_s3(
            urls=file_upload_response.urls, data=data, upload_id=file_upload_response.upload_id
        )

        # Complete file upload. Necessary to notify orchestrator that all parts have been uploaded.
        complete_file_upload_response: (
            CompleteFileUploadResponse | dict
        ) = await miner_api_client.complete_file_upload_request(
            hotkey=hotkey,
            file_upload_completion_request=FileUploadCompletionRequest(
                object_name=file_upload_response.object_name,
                upload_id=file_upload_response.upload_id,
                parts=parts,
            ),
        )

        if isinstance(complete_file_upload_response, dict):
            return complete_file_upload_response

        return complete_file_upload_response.object_path

    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise


async def upload_tensor(
    miner_api_client: MinerAPIClient,
    tensor: torch.Tensor,
    hotkey: Keypair,
    file_type: Literal["activation", "weights", "optimizer_state", "local_optimizer_state"] = "activation",
) -> CompleteFileUploadResponse:
    initiate_response: FileUploadResponse | dict = await miner_api_client.initiate_file_upload_request(
        hotkey=hotkey,
        file_upload_request=FileUploadRequest(
            file_type=file_type,
            num_parts=1,
        ),
    )
    assert len(tensor) > 0, "Tensor is empty"

    if not initiate_response:
        raise Exception("Error initiating file upload")

    check_for_nans_and_infs(
        tensor=tensor,
        name=f"Uploading tensor of file type {file_type}",
        exception_type=NanInfException,
    )

    # Reinterpret tensor memory as bytes in a consistent format (bfloat16 → uint8 bytes)
    # Always upload as bfloat16-backed bytes to match the downloader's default expectation.
    tensor_cpu = tensor.detach().to("cpu").to(torch.bfloat16).contiguous()
    data = tensor_cpu.view(torch.uint8).numpy().tobytes()

    if RUN_FLAGS.compress_s3_files.isOn():
        data = gzip.compress(data)

    try:
        parts: list[dict] = await MinerAPIClient.upload_multipart_to_s3(
            urls=initiate_response.urls, data=data, upload_id=initiate_response.upload_id
        )

        response: CompleteFileUploadResponse | dict = await miner_api_client.complete_file_upload_request(
            hotkey=hotkey,
            file_upload_completion_request=FileUploadCompletionRequest(
                object_name=initiate_response.object_name,
                upload_id=initiate_response.upload_id,
                parts=parts,
            ),
        )
        return response

    except Exception as e:
        logger.exception(f"Error uploading multipart to S3: {e}")
        raise


# OBSOLETE
def extract_filename_from_url(url):
    """
    Extract the filename from a URL, handling both regular paths and query parameters.

    Args:
        url: The URL to extract filename from


    Returns:
        str: The extracted filename
    """
    # Parse the URL
    parsed_url = urlparse(url)

    # Get the path component
    path = parsed_url.path

    # Extract filename from path
    filename = path.split("/")[-1]

    return filename


async def wait_for_state(state: LayerPhase, miner_api_client: MinerAPIClient, raise_bad_sync: bool = True):
    while True:
        await asyncio.sleep(5)
        logger.info(f"Waiting for state {state}")
        response = await miner_api_client.get_layer_state_request()
        if response == state.value:
            logger.info(f"Orchestrator is finally in state {state}")
            miner_api_client.layer_state = LayerPhase.from_str(response)
            break
        elif LayerPhase.from_str(response).next() == state:
            continue
        else:
            miner_api_client.layer_state = LayerPhase.TRAINING
            if raise_bad_sync:
                raise LayerStateException(
                    f"Miner is out of sync with the orchestrator. Miner is waiting for orchestrator to be in state {state}, but orchestrator is in state {response}, setting state to training"
                )
