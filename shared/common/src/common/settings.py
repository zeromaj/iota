import os
from loguru import logger
from dotenv import load_dotenv

from common.configs import LLAMA32_CONFIG_100M

COMMON_DOTENV_PATH = os.getenv("COMMON_DOTENV_PATH", ".env")
if not load_dotenv(dotenv_path=COMMON_DOTENV_PATH):
    logger.warning("No .env file found for common settings")

# Generic settings
MOCK = os.getenv("MOCK") == "True"
if MOCK:
    logger.warning("RUNNING IN MOCK MODE!")
LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED") == "True"

VALIDATE = os.getenv("VALIDATE") == "True"

# Bittensor settings
__SPEC_VERSION__ = 4065
BITTENSOR = os.getenv("BITTENSOR") == "True"
MAX_NUM_PARTS = int(os.getenv("MAX_NUM_PARTS", 10000))
NETUID = int(os.getenv("NETUID", "141"))
NETWORK = os.getenv("NETWORK", "test")
OWNER_UID = 209

# Orchestrator settings (common)
ORCHESTRATOR_PORT = int(os.getenv("ORCHESTRATOR_PORT", 8000))
ORCHESTRATOR_HOST = os.getenv("ORCHESTRATOR_HOST", "localhost")
ORCHESTRATOR_SCHEMA = os.getenv("ORCHESTRATOR_SCHEME", "http")
ORCHESTRATOR_URL = f"{ORCHESTRATOR_SCHEMA}://{ORCHESTRATOR_HOST}:{ORCHESTRATOR_PORT}"

# S3_ENDPOINT = "https://c3e628c65898759897b33b55e56bb7a0.r2.cloudflarestorage.com"
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://c3e628c65898759897b33b55e56bb7a0.r2.cloudflarestorage.com")
S3_REGION = os.getenv("S3_REGION", "eu-north-1")
S3_BUCKET = os.getenv("S3_BUCKET", "iota-prod-enam")
USE_S3 = os.getenv("USE_S3", True)

MIN_PART_SIZE = 10 * 1024 * 1024  # 10MB
MAX_PART_SIZE = 100 * 1024 * 1024  # 100MB

# System Settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
ACTIVATION_CACHE_TIMEOUT = 60 * 20
MAX_ACTIVATION_CACHE_SIZE = 2
TIMEOUT = 20000000000000000  # TODO: this is a hack to avoid timeouts, we should find a better way to handle this

# Epistula settings
SIGNATURE_TIMEOUT_MS = 10000

ACTIVATION_TIMEOUT = 60

# File size limits
FILE_SIZE_LIMITS = {
    "weights": {"min": 1, "max": 10000 * 1024 * 1024},
    "activation": {"min": 1, "max": 100 * 1024 * 1024},
    "optimizer_state": {"min": 1, "max": 10000 * 1024 * 1024},
    "metadata": {"min": 1, "max": 100 * 1024 * 1024},
}

# LLM Model Settings
MODEL_CFG = LLAMA32_CONFIG_100M if MOCK else LLAMA32_CONFIG_100M  # TODO: change this back to 15B after testing
MODEL_SPLITS = MODEL_CFG.pop("model_splits")
N_LAYERS = len(MODEL_SPLITS)
TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"
MODEL_SIZE = MODEL_CFG["total_global_params"]
HF_TOKEN = os.getenv("HF_TOKEN")

# Model Training Settings
DATASET_NAME = "HuggingFaceFW/fineweb"
SHUFFLE_DATASET = True
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 5 if MOCK else 200
SEQUENCE_LENGTH = 800
PACK_SAMPLES = True
WEIGHT_DECAY = 1e-1
GRAD_CLIP_NORM = 1.0
LEARNING_RATE = 2 * 1e-4
TOTAL_TRAIN_STEPS = 100_000
LR_WARMUP_START_FACTOR = 1  # 5e-3
LR_WARMUP_STEPS = 1
LR_CONST_STEPS = 500
LR_TAIL_STEPS_FRAC = 0.02
LR_FINAL_FACTOR = 0.10
LR_SAW_CYCLE_LENGTH = 1000
TEST_MODE = os.getenv("TEST_MODE") == "True"


# Weights & Biases
USE_WANDB = os.getenv("USE_WANDB") == "True"
WANDB_PROJECT = "pretrain-test"
WANDB_ENTITY = "macrocosmos"
RUN_NAME = "local_3_bottlenecks_8_lr4schedulers2_5e-4_gclip"
WANDB_TOKEN = os.getenv("WANDB_TOKEN")
S3_ACTIVATIONS_PATH = os.getenv("ACTIVATION_S3_PATH", "epistula-activations/activations")

SAMPLES_BUCKET = os.getenv("SAMPLE_S3_BUCKET", "iota-prod-enam")

BURN_FACTOR = 0.80
REQUEST_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", "3"))

# Profiler settings
PROFILER_OUTPUT_DIR = os.getenv("PROFILER_OUTPUT_DIR", "./profiler")
