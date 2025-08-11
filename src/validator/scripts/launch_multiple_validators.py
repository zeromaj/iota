#!/usr/bin/env python3
"""
Simple script to launch multiple validators concurrently.
"""

import argparse
import asyncio
import os
import sys

from loguru import logger

# Add the validator package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from validator.validator import Validator


async def run_single_validator(wallet_name: str, wallet_hotkey: str, validator_id: int):
    """Run a single validator instance."""
    try:
        logger.info(f"Starting validator {validator_id} with wallet_name={wallet_name}, wallet_hotkey={wallet_hotkey}")
        validator = Validator(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey)
        await validator.run_validator()
    except Exception as e:
        logger.error(f"Error in validator {validator_id}: {e}")
        raise


async def launch_multiple_validators(
    num_validators: int, wallet_name_prefix: str = "validator", wallet_hotkey_prefix: str = "hotkey"
):
    """Launch multiple validators concurrently."""
    tasks = []

    for i in range(num_validators):
        wallet_name = f"{wallet_name_prefix}_{i}"
        wallet_hotkey = f"{wallet_hotkey_prefix}_{i}"

        # Create a task for each validator
        task = asyncio.create_task(run_single_validator(wallet_name, wallet_hotkey, i), name=f"validator_{i}")
        tasks.append(task)

        # Small delay between launches to prevent overwhelming the system
        await asyncio.sleep(0.1)

    logger.info(f"Launched {num_validators} validators. Waiting for completion...")

    try:
        # Wait for all validators to complete
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping all validators...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All validators stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch multiple validators concurrently")
    parser.add_argument("--num-validators", type=int, default=3, help="Number of validators to launch (default: 3)")
    parser.add_argument(
        "--wallet-name-prefix", type=str, default="validator", help="Prefix for wallet names (default: 'validator')"
    )
    parser.add_argument(
        "--wallet-hotkey-prefix", type=str, default="hotkey", help="Prefix for wallet hotkeys (default: 'hotkey')"
    )
    parser.add_argument("--log-level", type=str, default="DEBUG", help="Log level (default: DEBUG)")

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level=args.log_level,
    )
    logger.add(
        "multiple_validators.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="3 days",
    )

    logger.info(f"Launching {args.num_validators} validators...")
    logger.info(f"Wallet name prefix: {args.wallet_name_prefix}")
    logger.info(f"Wallet hotkey prefix: {args.wallet_hotkey_prefix}")

    try:
        asyncio.run(
            launch_multiple_validators(
                num_validators=args.num_validators,
                wallet_name_prefix=args.wallet_name_prefix,
                wallet_hotkey_prefix=args.wallet_hotkey_prefix,
            )
        )
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
