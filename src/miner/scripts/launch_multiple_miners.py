#!/usr/bin/env python3
"""
Simple script to launch multiple miners concurrently.
"""

import argparse
import asyncio
import os
import re
import subprocess
import sys

from loguru import logger

# Add the miner package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
# Add the shared common package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "shared", "common", "src"))

from common.settings import BITTENSOR

from miner.new_miner import Miner


def get_available_hotkeys(wallet_name: str = "swarm-test") -> list[str]:
    """
    Get available hotkeys for a given wallet from btcli w list command.

    Args:
        wallet_name: The wallet name to search for hotkeys

    Returns:
        List of hotkey names
    """
    try:
        logger.info(f"Getting hotkeys for wallet '{wallet_name}' from btcli...")
        result = subprocess.run(["btcli", "w", "list"], capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            logger.error(f"btcli command failed: {result.stderr}")
            return []

        output = result.stdout
        hotkeys = []

        # Look for the wallet section and extract hotkeys
        lines = output.split("\n")
        in_target_wallet = False

        for line in lines:
            # Check if we're entering the target wallet section
            if f"Coldkey {wallet_name}" in line:
                in_target_wallet = True
                continue

            # Check if we're entering a different wallet section
            if in_target_wallet and "Coldkey " in line and wallet_name not in line:
                break

            # Extract hotkey names when in the target wallet section
            if in_target_wallet and "Hotkey " in line:
                # Pattern to match: "│   ├── Hotkey miner-XX"
                match = re.search(r"Hotkey\s+([^\s]+)", line)
                if match:
                    hotkey_name = match.group(1)
                    hotkeys.append(hotkey_name)
                    logger.debug(f"Found hotkey: {hotkey_name}")

        logger.info(f"Found {len(hotkeys)} hotkeys for wallet '{wallet_name}': {hotkeys}")
        return hotkeys

    except subprocess.TimeoutExpired:
        logger.error("btcli command timed out")
        return []
    except subprocess.CalledProcessError as e:
        logger.error(f"btcli command error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error getting hotkeys: {e}")
        return []


async def run_single_miner(wallet_name: str, wallet_hotkey: str, miner_id: int):
    """Run a single miner instance."""
    try:
        logger.info(f"Starting miner {miner_id} with wallet_name={wallet_name}, wallet_hotkey={wallet_hotkey}")
        miner = Miner(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey)
        await miner.run_miner()
    except Exception as e:
        logger.error(f"Error in miner {miner_id}: {e}")
        raise


async def launch_multiple_miners(
    num_miners: int, wallet_name_prefix: str = "miner", wallet_hotkey_prefix: str = "hotkey"
):
    """Launch multiple miners concurrently."""
    tasks = []

    # Get available hotkeys if BITTENSOR mode is enabled
    available_hotkeys = []
    if BITTENSOR:
        available_hotkeys = get_available_hotkeys("swarm-test")
        if not available_hotkeys:
            logger.error("No hotkeys found for wallet 'swarm-test'. Falling back to prefix-based naming.")

    for i in range(num_miners):
        if BITTENSOR and available_hotkeys:
            # Use predefined wallet and hotkeys when BITTENSOR is enabled
            wallet_name = "swarm-test"
            if i < len(available_hotkeys):
                wallet_hotkey = available_hotkeys[i]
            else:
                logger.warning(f"Not enough available hotkeys for miner {i}. Using fallback naming.")
                wallet_hotkey = f"{wallet_hotkey_prefix}_{i}"
        else:
            # Use the original naming convention
            wallet_name = f"{wallet_name_prefix}_{i}"
            wallet_hotkey = f"{wallet_hotkey_prefix}_{i}"

        # Create a task for each miner
        task = asyncio.create_task(run_single_miner(wallet_name, wallet_hotkey, i), name=f"miner_{i}")
        tasks.append(task)

        # Small delay between launches to prevent overwhelming the system
        await asyncio.sleep(0.1)

    logger.info(f"Launched {num_miners} miners. Waiting for completion...")

    try:
        # Wait for all miners to complete
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping all miners...")
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All miners stopped")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Launch multiple miners concurrently")
    parser.add_argument("--num-miners", type=int, default=3, help="Number of miners to launch (default: 3)")
    parser.add_argument(
        "--wallet-name-prefix", type=str, default="miner", help="Prefix for wallet names (default: 'miner')"
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
        "multiple_miners.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="3 days",
    )

    if BITTENSOR:
        logger.info("BITTENSOR mode enabled - using wallet 'swarm-test' with dynamically retrieved hotkeys")
        # Get hotkeys early to show count in logs
        hotkeys = get_available_hotkeys("swarm-test")
        if hotkeys:
            logger.info(f"Available hotkeys: {len(hotkeys)}")
            if args.num_miners > len(hotkeys):
                logger.warning(f"Requested {args.num_miners} miners but only {len(hotkeys)} hotkeys available")
        else:
            logger.warning("No hotkeys found - will fall back to prefix-based naming")
    else:
        logger.info("BITTENSOR mode disabled - using prefix-based naming")

    logger.info(f"Launching {args.num_miners} miners...")
    logger.info(f"Wallet name prefix: {args.wallet_name_prefix}")
    logger.info(f"Wallet hotkey prefix: {args.wallet_hotkey_prefix}")

    try:
        asyncio.run(
            launch_multiple_miners(
                num_miners=args.num_miners,
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
