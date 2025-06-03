import sys
import bittensor as bt

import functools
import concurrent
from typing import List, Optional, Any
from fastapi import HTTPException
import settings
import tenacity


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def get_subtensor():
    if settings.BITTENSOR:
        return bt.subtensor(settings.network)
    else:
        from bittensor.utils.mock.subtensor_mock import Subtensor

        return Subtensor(network=settings.network)


def is_validator(uid: int, metagraph: bt.metagraph, vpermit_rao_limit: int = 10_000) -> bool:
    """Checks if a UID on the subnet is a validator."""
    if settings.network == "test":
        return float(metagraph.S[uid]) >= vpermit_rao_limit
    else:
        return metagraph.validator_permit[uid] and float(metagraph.S[uid]) >= vpermit_rao_limit


def is_miner(uid: int, metagraph: bt.metagraph, vpermit_rao_limit) -> bool:
    """Checks if a UID on the subnet is a miner."""
    # Assume everyone who isn't a validator is a miner.
    # This explicilty disallows validator/miner hybrids.
    # 1) Blacklist known bad coldkeys.

    # if metagraph.coldkeys[uid] in [
    # ]:
    #     bt.logging.trace(f"Ignoring known bad coldkey {metagraph.coldkeys[uid]}.")
    #     return False

    return not is_validator(uid, metagraph, vpermit_rao_limit)


def assert_registered(wallet: bt.wallet, metagraph: bt.metagraph):
    """Exits the process if wallet isn't registered in metagraph"""
    # --- Check for registration.
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"Wallet: {wallet} is not registered on netuid {metagraph.netuid}."
            f" Please register the hotkey using `btcli subnets register` before trying again."
        )
        sys.exit(1)


def get_miner_uids(metagraph: bt.metagraph, my_uid: int, vpermit_rao_limit: int) -> List[int]:
    """Gets the uids of all miners in the metagraph."""
    return sorted(
        [
            uid.item()
            for uid in metagraph.uids
            if is_miner(uid.item(), metagraph, vpermit_rao_limit) and uid.item() != my_uid
        ]
    )


def get_uid(wallet: bt.wallet, metagraph: bt.metagraph) -> Optional[int]:
    """Gets the uid of the wallet in the metagraph or None if not registered."""
    if wallet.hotkey.ss58_address in metagraph.hotkeys:
        return metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    return None


def run_in_thread(func: functools.partial, ttl: int, name=None) -> Any:
    """Runs the provided function on a thread with 'ttl' seconds to complete.

    Args:
        func (functools.partial): Function to be run.
        ttl (int): How long to try for in seconds.

    Returns:
        Any: The value returned by 'func'
    """

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        future = executor.submit(func)
        return future.result(timeout=ttl)
    except concurrent.futures.TimeoutError as e:
        bt.logging.error(f"Failed to complete '{name}' within {ttl} seconds.")
        raise TimeoutError(f"Failed to complete '{name}' within {ttl} seconds.") from e
    finally:
        bt.logging.trace(f"Completed {name}")
        executor.shutdown(wait=False)
        bt.logging.trace(f"{name} cleaned up successfully")


def verify_entity_type(
    signed_by: str,
    metagraph: bt.metagraph,
    required_type: str = None,  # "miner", "validator", or None for any
    vpermit_rao_limit: int = 10 if settings.network == "test" else 50_000,
) -> dict:
    """
    Verify that the signed_by hotkey exists in the metagraph and check its type.

    Args:
        signed_by: The hotkey that signed the request
        metagraph: The bittensor metagraph
        required_type: Required entity type ("miner", "validator", or None)
        vpermit_rao_limit: RAO limit for validator permit

    Returns:
        Dict with entity info including type and uid

    Raises:
        HTTPException: If verification fails
    """
    # Check if hotkey is registered
    if signed_by not in metagraph.hotkeys:
        raise HTTPException(status_code=403, detail=f"Hotkey {signed_by} not registered on subnet")

    # Get UID
    uid = metagraph.hotkeys.index(signed_by)

    # Check entity type
    is_validator_entity = is_validator(uid, metagraph, vpermit_rao_limit)
    is_miner_entity = is_miner(uid, metagraph, vpermit_rao_limit)

    entity_type = "validator" if is_validator_entity else "miner" if is_miner_entity else "unknown"

    # Verify required type if specified
    if required_type:
        if required_type == "validator" and not is_validator_entity:
            raise HTTPException(status_code=403, detail=f"Hotkey {signed_by} is not a validator")
        elif required_type == "miner" and not is_miner_entity:
            raise HTTPException(status_code=403, detail=f"Hotkey {signed_by} is not a miner")

    return {
        "uid": uid,
        "hotkey": signed_by,
        "entity_type": entity_type,
        "is_validator": is_validator_entity,
        "is_miner": is_miner_entity,
    }


class NotRegisteredError(Exception):
    """Custom exception for when a hotkey is not registered."""

    pass


subtensor = get_subtensor()
