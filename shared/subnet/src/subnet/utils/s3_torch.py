from typing import Literal

import numpy as np
import requests
import torch
from common.utils.partitions import MinerPartition
from loguru import logger
from subnet.utils.vector_utils import check_for_nans_and_infs


def download_activation(path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> torch.Tensor:
    """Download an activation from S3 storage."""
    # Download from S3
    response = requests.get(path)
    loaded_tensor = np.frombuffer(response.content, dtype=np.uint8)
    loaded_tensor = torch.tensor(loaded_tensor).view(dtype).to(device)

    assert isinstance(
        loaded_tensor, torch.Tensor
    ), f"Downloaded tensor is not a torch.Tensor: {type(loaded_tensor)}, path: {path}"
    check_for_nans_and_infs(loaded_tensor, f"activation downloaded from {path}")

    return loaded_tensor


def download_weights_or_optimizer_state(
    path: str,
    partition: MinerPartition = MinerPartition(),
    data_type: Literal["weights", "optimizer_state"] = "weights",
) -> torch.Tensor:
    """Download weights from S3 storage."""

    if data_type == "weights":
        data = partition.weight_data
    elif data_type == "optimizer_state":
        data = partition.optimizer_state_data
    else:
        raise ValueError(f"Invalid type: {data_type}")

    # if partition is not specified, download the full tensor
    if data.chunk_start_byte is None or data.chunk_end_byte is None:
        logger.warning(f"Chunk {partition.chunk_number} has no start or end byte")
        response = requests.get(path)
        if response.status_code > 299:
            logger.warning(f"Chunk {partition.chunk_number} not found at {path} response content: {response}")
            response.raise_for_status()
            raise Exception(f"Chunk {partition.chunk_number} not found at {path} response content: {response}")
    else:
        byte_range = f"bytes={data.chunk_start_byte}-{data.chunk_end_byte - 1}"
        response = requests.get(path, headers={"Range": byte_range})
        if response.status_code > 299:
            logger.warning(f"Chunk {partition.chunk_number} not found at {path} response content: {response}")
            response.raise_for_status()
            raise Exception(f"Chunk {partition.chunk_number} not found at {path} response content: {response}")

    binary_data = response.content
    section_numpy = np.frombuffer(binary_data, dtype=np.uint8)
    section_torch = torch.from_numpy(section_numpy.copy())
    # assumes default dtype if not specified
    section_torch = section_torch.view(getattr(torch, data.chunk_dtype))
    return section_torch
