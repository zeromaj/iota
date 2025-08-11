from time import time
from typing import Literal

import aiohttp
from common.utils.exceptions import NanInfWarning
import numpy as np
import torch
from common.utils.partitions import MinerPartition
from loguru import logger
from subnet.utils.vector_utils import check_for_nans_and_infs
from asyncio.exceptions import TimeoutError


async def download_activation(path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda") -> torch.Tensor:
    """Download an activation from S3 storage."""
    # Download from S3
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(path) as response:
            response.raise_for_status()
            content = await response.read()
            loaded_tensor = np.frombuffer(content, dtype=np.uint8)
            loaded_tensor = torch.tensor(loaded_tensor).view(dtype).to(device)

    assert isinstance(
        loaded_tensor, torch.Tensor
    ), f"Downloaded tensor is not a torch.Tensor: {type(loaded_tensor)}, path: {path}"

    check_for_nans_and_infs(loaded_tensor, f"activation downloaded from {path}", exception_type=NanInfWarning)

    return loaded_tensor


async def download_weights_or_optimizer_state(
    path: str,
    partition: MinerPartition = MinerPartition(),
    data_type: Literal["weights", "optimizer_state"] = "weights",
) -> torch.Tensor:
    """Download weights from S3 storage."""
    try:
        if data_type == "weights":
            data = partition.weight_data
        elif data_type == "optimizer_state":
            data = partition.optimizer_state_data
        else:
            raise ValueError(f"Invalid type: {data_type}")

        timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout
        start_time = time()
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # if partition is not specified, download the full tensor
            if data.chunk_start_byte is None or data.chunk_end_byte is None:
                logger.warning(f"Chunk {partition.chunk_number} has no start or end byte")
                async with session.get(path) as response:
                    if response.status > 299:
                        logger.warning(
                            f"Chunk {partition.chunk_number} not found at {path} response status: {response.status}"
                        )
                        response.raise_for_status()
                        raise Exception(
                            f"Chunk {partition.chunk_number} not found at {path} response status: {response.status}"
                        )
                    binary_data = await response.read()
            else:
                byte_range = f"bytes={data.chunk_start_byte}-{data.chunk_end_byte - 1}"
                async with session.get(path, headers={"Range": byte_range}) as response:
                    if response.status > 299:
                        logger.warning(
                            f"Chunk {partition.chunk_number} not found at {path} response status: {response.status}"
                        )
                        response.raise_for_status()
                        raise Exception(
                            f"Chunk {partition.chunk_number} not found at {path} response status: {response.status}"
                        )
                    binary_data = await response.read()

        logger.debug(f"Received binary data for {path}")
        section_numpy = np.frombuffer(binary_data, dtype=np.uint8)
        section_torch = torch.from_numpy(section_numpy.copy())
        logger.debug("Converted binary data to torch tensor")
        # assumes default dtype if not specified
        section_torch = section_torch.view(getattr(torch, data.chunk_dtype))
        return section_torch
    except TimeoutError as e:
        logger.error(
            f"Timeout error downloading weights or optimizer state: {e}. Time taken: {time() - start_time} seconds. Started download at {start_time}."
        )
        raise
    except Exception as e:
        logger.error(f"Error downloading weights or optimizer state: {e}")
        raise
