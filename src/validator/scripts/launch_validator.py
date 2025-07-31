#!/usr/bin/env python3
"""
Simple script to launch a single validator instance.
"""

import argparse
import asyncio
import os
import sys

from loguru import logger

# Add the validator package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from validator.validator import Validator


async def run_single_validator(wallet_name: str, wallet_hotkey: str, validator_id: int = 0):
    """Run a single validator instance."""
    try:
        logger.info(f"Starting validator {validator_id} with wallet_name={wallet_name}, wallet_hotkey={wallet_hotkey}")
        validator = Validator(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey)
        await validator.run_validator()
    except Exception as e:
        logger.error(f"Error in validator {validator_id}: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch a single validator instance")
    parser.add_argument(
        "--wallet-name",
        type=str,
        default=os.getenv("WALLET_NAME", "validator"),
        help="Wallet name for the validator (default: 'validator' or WALLET_NAME env var)",
    )
    parser.add_argument(
        "--wallet-hotkey",
        type=str,
        default=os.getenv("WALLET_HOTKEY", "hotkey"),
        help="Wallet hotkey for the validator (default: 'hotkey' or WALLET_HOTKEY env var)",
    )
    parser.add_argument("--validator-id", type=int, default=0, help="Validator ID for logging purposes (default: 0)")
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
        "single_validator.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="3 days",
    )

    logger.info(f"Launching validator {args.validator_id}...")
    logger.info(f"Wallet name: {args.wallet_name}")
    logger.info(f"Wallet hotkey: {args.wallet_hotkey}")

    try:
        asyncio.run(
            run_single_validator(
                wallet_name=args.wallet_name,
                wallet_hotkey=args.wallet_hotkey,
                validator_id=args.validator_id,
            )
        )
    except KeyboardInterrupt:
        logger.info("Script terminated by user")
    except Exception as e:
        logger.exception(f"Script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
