import os

from common.settings import BITTENSOR, MOCK
from dotenv import load_dotenv
from loguru import logger

DOTENV_PATH = os.getenv("DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=DOTENV_PATH):
    logger.warning("No .env file found for validator")

DEVICE = os.getenv("DEVICE", "cpu")

# WEIGHT_SUBMIT_INTERVAL: int = 3600  # submit weight every 1 hour
WEIGHT_SUBMIT_INTERVAL: int = 10 if (MOCK or not BITTENSOR) else 60 * 21  # submit weight every 21 minutes
FETCH_TASKS_INTERVAL: int = 5 if (MOCK or not BITTENSOR) else 60 * 5  # fetch tasks every 5 minutes
ORCHESTRATOR_HEALTH_CHECK_INTERVAL: int = 60  # check orchestrator health every 1 minute

# Validation Thresholds
COSINE_SIMILARITY_THRESHOLD = 0.9
ACTIVATION_MAGNITUDE_THRESHOLD = 0.8
OPTIMIZER_SIMILARITY_THRESHOLD = 0.8

# Health settings
LAUNCH_HEALTH = os.getenv("LAUNCH_HEALTH") == "True"
VALIDATOR_HEALTH_HOST = os.getenv("VALIDATOR_HEALTH_HOST", "0.0.0.0")
VALIDATOR_HEALTH_PORT = int(os.getenv("VALIDATOR_HEALTH_PORT", 9000))
VALIDATOR_HEALTH_ENDPOINT = os.getenv("VALIDATOR_HEALTH_ENDPOINT", "/health")

WALLET_NAME = os.getenv("wallet_name", "test")
WALLET_HOTKEY = os.getenv("wallet_hotkey", "m1")
