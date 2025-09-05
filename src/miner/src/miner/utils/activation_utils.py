import torch
from common import settings as common_settings
from common.utils.s3_utils import download_file
from loguru import logger

from miner import settings as miner_settings


async def download_sample(download_url: str, tokenizer) -> torch.Tensor:
    data = await download_file(presigned_url=download_url)
    text = data.decode("utf-8")

    if common_settings.MOCK:
        return torch.randn(size=(common_settings.MINI_BATCH_SIZE, 100), dtype=torch.bfloat16).to(miner_settings.DEVICE)

    sample = torch.tensor(tokenizer.encode(text)).to(miner_settings.DEVICE)
    if len(sample) < common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE:
        raise Exception(f"Sample is too short: {len(sample)} < {common_settings.SEQUENCE_LENGTH}")

    sample = sample[: common_settings.SEQUENCE_LENGTH * common_settings.MINI_BATCH_SIZE]
    sample = sample.reshape(common_settings.MINI_BATCH_SIZE, common_settings.SEQUENCE_LENGTH)
    logger.info(f"Sample shape: {sample.shape}")
    # return sample.unsqueeze(0)
    return sample
