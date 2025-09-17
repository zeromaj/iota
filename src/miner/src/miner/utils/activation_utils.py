import torch
from common import settings as common_settings
from common.utils.s3_utils import download_file
from loguru import logger

from miner import settings as miner_settings


async def download_sample(download_url: str, tokenizer) -> torch.Tensor:
    """
    Downloads the sample from the given URL and returns it as a tensor.

    Args:
        download_url: The URL of the sample to download.
        tokenizer: The tokenizer to use to decode the sample.
    """
    data = await download_file(presigned_url=download_url)

    # Some objects may be gzip-compressed without content-encoding header in R2; try transparently
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        import gzip

        try:
            text = gzip.decompress(data).decode("utf-8")
        except Exception as e:
            raise Exception(f"Failed to decode sample as utf-8 (and gzip fallback failed): {e}")

    if common_settings.MOCK:
        return torch.randn(size=(common_settings.MINI_BATCH_SIZE, 100), dtype=torch.bfloat16).to(miner_settings.DEVICE)

    sample = torch.tensor(tokenizer.encode(text)).to(miner_settings.DEVICE)
    if len(sample) < common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE:
        raise Exception(
            f"Sample is too short: {len(sample)} < {common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE}"
        )

    sample = sample[: common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE]
    sample = sample.reshape(common_settings.MINI_BATCH_SIZE, common_settings.SEQUENCE_LENGTH)

    logger.info(f"Sample shape: {sample.shape}")
    return sample
