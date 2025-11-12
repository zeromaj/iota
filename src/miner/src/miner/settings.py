import os
from dotenv import load_dotenv
from loguru import logger

from common import settings as common_settings


DOTENV_PATH = os.getenv("DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=DOTENV_PATH):
    logger.warning("No .env file found for miner settings")


# Wallet
WALLET_NAME = os.getenv("MINER_WALLET", "test")
WALLET_HOTKEY = os.getenv("MINER_HOTKEY", "m1")

MINER_HEALTH_HOST = os.getenv("MINER_HEALTH_HOST", "0.0.0.0")
MINER_HEALTH_PORT = int(os.getenv("MINER_HEALTH_PORT", 9000))
MINER_HEALTH_ENDPOINT = os.getenv("MINER_HEALTH_ENDPOINT", "/health")

LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"

DEVICE = os.getenv("DEVICE", "cpu")

# Training settings
TIMEOUT = int(os.getenv("MINER_TIMEOUT", "300"))  # 5 minutes default
PACK_SAMPLES = os.getenv("PACK_SAMPLES", "True") == "True"  # not for miner's to change
N_PARTITION_BATCHES = int(os.getenv("N_PARTITION_BATCHES", "20"))  # not for miner's to change
PREVIOUS_WEIGHTS = os.getenv("MODEL_DIR", "./weights")

# Activation settings - miners can reduce if they are OOM'ing but can't surpass common settings
MAX_ACTIVATION_CACHE_SIZE = int(os.getenv("MAX_ACTIVATION_CACHE_SIZE", common_settings.MAX_ACTIVATION_CACHE_SIZE))
MAX_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MAX_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MAX_FORWARD_ACTIVATIONS_IN_QUEUE)
)
MIN_FORWARD_ACTIVATIONS_IN_QUEUE = int(
    os.getenv("MIN_FORWARD_ACTIVATIONS_IN_QUEUE", common_settings.MIN_FORWARD_ACTIVATIONS_IN_QUEUE)
)

# Training settings
LOCAL_BATCH_SIZE = int(
    os.getenv("LOCAL_BATCH_SIZE", "2")
)  # Splits the minibatch further into even smaller local batches to avoid running out of memory
PSEUDO_GRADIENTS_BATCH_SIZE = int(os.getenv("PSEUDO_GRADIENTS_BATCH_SIZE", "100"))
