import sys
import asyncio
from loguru import logger

from miner.new_miner import Miner
from miner import settings as miner_settings
from common import settings as common_settings

# Setup logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSSSSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level> | <magenta>{extra}</magenta>",
    level="DEBUG",
    colorize=True,
)
if common_settings.LOG_FILE_ENABLED:
    logger.add(
        f"../../logs/{miner_settings.WALLET_NAME}.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSSSSS} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        colorize=False,
    )


def main():
    """Main entry point for the miner."""
    logger.info("Starting miner")
    logger.info(f"Wallet: {miner_settings.WALLET_NAME}")
    logger.info(f"Hotkey: {miner_settings.WALLET_HOTKEY}")
    logger.info(f"Device: {miner_settings.DEVICE}")
    logger.info(f"Timeout: {miner_settings.TIMEOUT}s")

    try:
        # Create miner instance
        miner = Miner(wallet_name=miner_settings.WALLET_NAME, wallet_hotkey=miner_settings.WALLET_HOTKEY)

        # Run the miner
        asyncio.run(miner.run_miner())

    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
    except Exception as e:
        logger.error(f"Error running miner: {e}")
        raise


if __name__ == "__main__":
    main()
