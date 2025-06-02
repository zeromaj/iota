import asyncio
import pytest

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_simple():
    await asyncio.sleep(0.5)


import pytest
import torch
import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.s3_interactions import (
    download_activation,
    download_weights_or_optimizer_state,
    download_metadata,
)
from utils.partitions import Partition
from miner.miner import Miner

pytest_plugins = ("pytest_asyncio",)


@pytest.mark.asyncio
async def test_activation_upload_download():
    miner = await Miner.create(
        wallet_name="test",
        wallet_hotkey="test",
        timeout=10,
        n_layers=3,
    )

    input_tensor = torch.tensor([1, 2, 3])
    activation_path = await miner.upload_activation("123", 1, "forward", input_tensor)
    output_tensor = download_activation(activation_path)
    assert torch.allclose(input_tensor, output_tensor)

    miner.close()


@pytest.mark.asyncio
async def test_weights_upload_download():
    miner = await Miner.create(
        wallet_name="test",
        wallet_hotkey="test",
        timeout=10,
        n_layers=3,
    )

    input_tensor = torch.tensor([1, 2, 3])
    input_tensor = input_tensor.to(torch.bfloat16)
    weights_path = await miner.upload_weights(input_tensor, "123", 1, 1)
    output_tensor = download_weights_or_optimizer_state(weights_path)
    assert torch.allclose(input_tensor, output_tensor)

    miner.close()


@pytest.mark.asyncio
async def test_weights_upload_download_with_partition():
    miner = await Miner.create(
        wallet_name="test",
        wallet_hotkey="test",
        timeout=10,
        n_layers=3,
    )

    input_tensor = torch.tensor([1, 2, 3])
    input_tensor = input_tensor.to(torch.bfloat16)
    weights_path = await miner.upload_weights(input_tensor, "123", 1, 1)

    start_idx = 1
    end_idx = 2
    partition = Partition(
        chunk_dtype="bfloat16",
        chunk_start_byte=start_idx * 2,
        chunk_end_byte=end_idx * 2,
        chunk_start_idx=start_idx,
        chunk_end_idx=end_idx,
    )

    output_tensor = download_weights_or_optimizer_state(weights_path, partition=partition)
    assert torch.allclose(input_tensor[start_idx:end_idx], output_tensor)

    miner.close()


@pytest.mark.asyncio
async def test_metadata_upload_download():
    miner = await Miner.create(
        wallet_name="test",
        wallet_hotkey="test",
        timeout=10,
        n_layers=3,
    )

    metadata = {"message": "Hello from S3!"}
    metadata_path = await miner.upload_metadata(metadata, "123", 1, 1)
    downloaded_metadata = download_metadata(metadata_path)
    assert metadata == downloaded_metadata

    miner.close()
