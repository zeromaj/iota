import asyncio
import time
import torch
from bittensor import Wallet
from loguru import logger
from pydantic import BaseModel

from common import settings as common_settings
from subnet.miner_api_client import MinerAPIClient
from miner import settings as miner_settings


class ActivationData(BaseModel):
    activation_id: str
    direction: str
    input_activations: torch.Tensor
    output_activations: torch.Tensor | None
    state: dict | None
    upload_time: float

    class Config:
        arbitrary_types_allowed = True


class StateManager(BaseModel):
    wallet: Wallet
    layer: int = 0
    # TODO: move the activation cache out of the state manager and into it's own class
    activation_cache: dict[str, ActivationData] = {}
    backwards_since_reset: int = 0
    training_epoch_when_registered: int = None
    run_id: str = None
    backwards_since_last_optim: int = 0
    local_optimization_steps: int = 0
    activation_cache_lock: asyncio.Lock = asyncio.Lock()

    class Config:
        arbitrary_types_allowed = True

    async def add_to_activation_cache(self, activation_id: str, data: ActivationData):
        async with self.activation_cache_lock:
            self.activation_cache[activation_id] = data

    async def remove_from_activation_cache(self, activation_id: str):
        async with self.activation_cache_lock:
            try:
                activation_data = self.activation_cache[activation_id]
                if activation_data.input_activations is not None:
                    del activation_data.input_activations
                if activation_data.output_activations is not None:
                    del activation_data.output_activations
                del self.activation_cache[activation_id]
                torch.cuda.empty_cache()
            except KeyError:
                logger.warning(f"Activation {activation_id} not found in cache")

    async def activation_cache_is_full(self, miner_api_client: MinerAPIClient) -> bool:
        if ooc := len(self.activation_cache) >= miner_settings.ACTIVATION_CACHE_SIZE:
            logger.info(
                f"Miner {self.wallet.hotkey.ss58_address[:8]} cache full with {len(self.activation_cache)} activations: {self.activation_cache.keys()}"
            )

            # Clean up inactive activations
            self.sync_activation_assignments(miner_api_client=miner_api_client)

            # Update cache_is_full status after cleanup
            ooc = len(self.activation_cache) >= miner_settings.ACTIVATION_CACHE_SIZE

            logger.info(
                f"Miner {self.wallet.hotkey.ss58_address[:8]} cache status: {len(self.activation_cache)}/{miner_settings.ACTIVATION_CACHE_SIZE} activations cached, out_of_cache: {ooc}"
            )
            logger.debug(f"Cache: {[(a.activation_id, a.direction) for a in self.activation_cache.values()]}")
        return ooc

    async def sync_activation_assignments(self, miner_api_client: MinerAPIClient) -> dict[str, bool]:
        # Clean up inactive activations
        activations_to_remove: dict[str, bool] = await miner_api_client.sync_activation_assignments(
            activation_ids=list(self.activation_cache.keys())
        )

        # Remove inactive activations
        async with self.activation_cache_lock:
            for activation_id, is_active in activations_to_remove.items():
                if activation_id in self.activation_cache and not is_active:
                    # Clean up tensors before removing from cache
                    cached_data = self.activation_cache[activation_id]
                    del cached_data  # This will help with garbage collection
                    del self.activation_cache[activation_id]
                    logger.info(f"Removed inactive activation {activation_id} from cache")

    def check_if_timeout(self, timeout: int):
        activations_to_remove: list[str] = []

        if len(self.activation_cache) > 0:
            for activation_id, activation_data in list(self.activation_cache.items()):
                upload_time = activation_data.upload_time
                if upload_time < time.time() - timeout:
                    # Explicitly remove tensor references to help the gc
                    activations_to_remove.append(activation_id)
                    logger.warning(f"🗑️ Removing activation {activation_id} from miner cache due to timeout")

        asyncio.gather(*[self.remove_from_activation_cache(activation_id) for activation_id in activations_to_remove])

    def increment_backward_count(self) -> bool:
        """Increment the backward pass counter and return True if optimization step is needed."""
        self.backwards_since_last_optim += 1
        return self.backwards_since_last_optim >= common_settings.MINI_BATCH_ACCUMULATION_COUNT

    def reset_optimization_counter(self):
        """Reset the backward pass counter after performing optimization step."""
        self.backwards_since_last_optim = 0

        # we can't process backwards activations on forwards processed before the optimization step
        self.activation_cache.clear()

        # Add explicit GPU cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset(self):
        # Clear the cache
        self.activation_cache.clear()
        self.activation_cache = {}

        # Reset the states
        self.backwards_since_reset = 0
        self.training_epoch_when_registered = None
        self.local_optimization_steps = 0
