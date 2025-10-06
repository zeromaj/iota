import gzip
from time import time
from loguru import logger

import asyncio
from asyncio.exceptions import TimeoutError

import aiohttp
from common.utils.exceptions import NanInfWarning
from common.models.miner_models import ChunkMetadata
from common.models.run_flags import RUN_FLAGS
from subnet.model.utils import log_cuda_memory_usage
from subnet.utils.vector_utils import check_for_nans_and_infs

import torch
import numpy as np


class AioHttpClientWithOpenSession:
    def __init__(self):
        self.SESSION_REFRESH_TIME = 300
        self.session = None
        self.session_created_at = time()

    async def close(self):
        if self.session:
            await self.session.close()

    async def refresh_session_if_needed(self):
        if not self.session or time() - self.session_created_at > self.SESSION_REFRESH_TIME:
            await self.close()
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
            self.session_created_at = time()

    async def get(self, url: str):
        await self.refresh_session_if_needed()
        response = await self.session.get(url)
        return response


async def process_response(response: aiohttp.ClientResponse, dtype: torch.dtype, device: str = "cuda") -> torch.Tensor:
    """Process the response from aiohttp and return a tensor."""
    content = await response.read()
    if RUN_FLAGS.compress_s3_files.isOn():
        content = gzip.decompress(content)
    loaded_tensor = np.frombuffer(content, dtype=np.uint8)
    loaded_tensor = torch.tensor(loaded_tensor).view(dtype).to(device)
    return loaded_tensor


async def download_tensor(
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    max_retries: int = 3,
) -> torch.Tensor:
    """Download bytes and cast into a tensor from S3 storage with retry logic.

    Args:
        path: URL path to download tensor from
        dtype: PyTorch data type for the tensor
        device: Device to load tensor to ('cuda', 'cpu', etc.)
        max_retries: Maximum number of retry attempts
        session: Optional aiohttp session to reuse. If None, creates a new session.

    Returns:
        Downloaded tensor
    """
    log_cuda_memory_usage(note="before downloading tensor")

    # Determine if we need to manage the session ourselves
    for attempt in range(max_retries + 1):
        try:
            # Create new session for single download
            response = await s3_client.get(path)
            response.raise_for_status()
            loaded_tensor = await process_response(response=response, dtype=dtype, device=device)

            assert isinstance(
                loaded_tensor, torch.Tensor
            ), f"Downloaded tensor is not a torch.Tensor: {type(loaded_tensor)}, path: {path}"

            check_for_nans_and_infs(loaded_tensor, f"tensor downloaded from {path}", exception_type=NanInfWarning)

            log_cuda_memory_usage(note="after downloading tensor")

            return loaded_tensor

        except aiohttp.ClientResponseError as e:
            if e.status >= 500 or e.status == 429:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        f"Retryable error (HTTP {e.status}), retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Server error (HTTP {e.status}) downloading tensor from R2: {e}. Failed after {max_retries + 1} attempts. This is likely a temporary R2 issue."
                    )
                    raise
            else:
                logger.error(f"HTTP error downloading tensor: {e}")
                raise

        except Exception as e:
            logger.error(f"Error downloading tensor: {e}")
            raise


async def download_weights_or_optimizer_state(
    metadata_info: ChunkMetadata,
    max_retries: int = 3,
) -> torch.Tensor:
    """Download weights from S3 storage with retry logic."""
    start_time = time()
    timeout = aiohttp.ClientTimeout(total=300)  # 5 minute timeout

    for attempt in range(max_retries + 1):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # if partition is not specified, download the full tensor
                byte_range = f"bytes={metadata_info.start_byte}-{metadata_info.end_byte - 1}"
                async with session.get(metadata_info.tensor_path, headers={"Range": byte_range}) as response:
                    if response.status > 299:
                        response.raise_for_status()
                    binary_data = await response.read()

            if RUN_FLAGS.compress_s3_files.isOn():
                binary_data = gzip.decompress(binary_data)

            section_numpy = np.frombuffer(binary_data, dtype=np.uint8)
            section_torch = torch.from_numpy(section_numpy.copy())
            # assumes default dtype if not specified
            section_torch = section_torch.view(getattr(torch, metadata_info.chunk_dtype))
            return section_torch

        except TimeoutError as e:
            logger.error(
                f"Timeout error downloading weights or optimizer state: {e}. Time taken: {time() - start_time} seconds. Started download at {start_time}."
            )
            raise
        except aiohttp.ClientResponseError as e:
            if e.status >= 500 or e.status == 429:
                if attempt < max_retries:
                    delay = 2**attempt
                    logger.warning(
                        f"Retryable error (HTTP {e.status}), retrying in {delay}s... (attempt {attempt + 1}/{max_retries + 1})"
                    )
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.warning(
                        f"Server error (HTTP {e.status}) downloading weights or optimizer state from R2: {e}. Failed after {max_retries + 1} attempts. This is likely a temporary R2 issue."
                    )
                    raise
            else:
                logger.error(f"HTTP error downloading weights or optimizer state: {e}")
                raise
        except Exception as e:
            logger.error(f"Error downloading weights or optimizer state: {e}")
            raise


s3_client = AioHttpClientWithOpenSession()
