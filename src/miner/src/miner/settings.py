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
N_PARTITION_BATCHES = int(os.getenv("N_PARTITION_BATCHES", "10"))  # not for miner's to change

PREVIOUS_WEIGHTS = os.getenv("MODEL_DIR", "./weights")

# Cache is in-process forwards so we want queue to be at least the size of the cache + some buffer (e.g. 15)
# bcs we want it to hold all the backwards for the cache (10) + a buffer set of next forwards (5)
# Miners: Increases these to process more activations at once. Decrease these if you're getting CUDA out of memory errors.
# the max cache size is enforced in the backend by not sending you more forward activations until you have completed some backwards
ACTIVATION_CACHE_SIZE = int(os.getenv("ACTIVATION_CACHE_SIZE", "2"))
assert (
    ACTIVATION_CACHE_SIZE <= common_settings.MAX_ACTIVATION_CACHE_SIZE
), "ACTIVATION_CACHE_SIZE must be less than or equal to MAX_ACTIVATION_CACHE_SIZE"
MAX_ACTIVATION_QUEUE_SIZE = int(os.getenv("MAX_ACTIVATION_QUEUE_SIZE", "3"))
assert (
    MAX_ACTIVATION_QUEUE_SIZE > ACTIVATION_CACHE_SIZE
), "MAX_ACTIVATION_QUEUE_SIZE must be greater than ACTIVATION_CACHE_SIZE to allow for backwards activations"
