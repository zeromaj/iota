from common import settings as common_settings
from loguru import logger
from transformers import AutoTokenizer


def load_tokenizer(tokenizer_name: str = "meta-llama/Llama-3.2-1B") -> AutoTokenizer:
    logger.info(f"Loading tokenizer from {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=common_settings.HF_TOKEN)

        if tokenizer is None:
            raise Exception("Error loading tokenizer")

        logger.success(f"✅ Tokenizer loaded successfully from {tokenizer_name}")
        return tokenizer

    except Exception as e:
        logger.exception(f"Error loading tokenizer: {e}")
        raise
