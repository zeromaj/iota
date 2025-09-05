import time

import torch
from bittensor import Wallet
from common import settings as common_settings
from loguru import logger
from pydantic import BaseModel
from subnet.miner_api_client import MinerAPIClient


class CacheEntry(BaseModel):
    input_activations: torch.Tensor
    output_activations: torch.Tensor
    state: dict
    upload_time: float

    class Config:
        arbitrary_types_allowed = True


class StateManager(BaseModel):
    wallet: Wallet
    layer: int = 0
    cache: dict[str, CacheEntry] = {}
    backwards_since_reset: int = 0
    training_epoch_when_registered: int = None
    num_metadata_chunks: int | None = None
    run_id: str = None
    backwards_since_last_optim: int = 0
    local_optimization_steps: int = 0

    class Config:
        arbitrary_types_allowed = True

    def add_to_cache(self, activation_id: str, data: CacheEntry):
        self.cache[activation_id] = data

    def remove_from_cache(self, activation_id: str):
        del self.cache[activation_id]

    async def out_of_cache(self, miner_api_client: MinerAPIClient) -> bool:
        if ooc := len(self.cache) >= common_settings.MAX_ACTIVATION_CACHE_SIZE:
            logger.info(
                f"Miner {self.wallet.hotkey} cache full with {len(self.cache)} activations: {self.cache.keys()}"
            )

            # Clean up inactive activations
            activations_to_remove: dict[str, bool] = await miner_api_client.sync_activation_assignments(
                activation_ids=list(self.cache.keys())
            )

            # Remove inactive activations
            for activation_id, is_active in activations_to_remove.items():
                if activation_id in self.cache and not is_active:
                    # Clean up tensors before removing from cache
                    cached_data = self.cache[activation_id]
                    del cached_data  # This will help with garbage collection
                    del self.cache[activation_id]
                    logger.info(f"Removed inactive activation {activation_id} from cache")

            # Update out_of_cache status after cleanup
            ooc = len(self.cache) >= common_settings.MAX_ACTIVATION_CACHE_SIZE

            logger.info(
                f"Miner {self.wallet.hotkey} cache status: {len(self.cache)}/{common_settings.MAX_ACTIVATION_CACHE_SIZE} activations cached, out_of_cache: {ooc}. Cache: {self.cache.keys()}"
            )
        return ooc

    def check_if_timeout(self, timeout: int):
        activations_to_remove: list[str] = []

        if len(self.cache) > 0:
            for activation_id, activation_data in list(self.cache.items()):
                upload_time = activation_data.upload_time
                if upload_time < time.time() - timeout:
                    # Explicitly remove tensor references to help the gc
                    activations_to_remove.append(activation_id)

                    logger.warning(f"🗑️ Removed activation {activation_id} from miner cache due to timeout")

        for activation_id in activations_to_remove:
            self.remove_from_cache(activation_id)

    def increment_backward_count(self):
        """Increment the backward pass counter and return True if optimization step is needed."""
        self.backwards_since_last_optim += 1
        return self.backwards_since_last_optim >= common_settings.MINI_BATCH_ACCUMULATION_COUNT

    def reset_optimization_counter(self):
        """Reset the backward pass counter after performing optimization step."""
        self.backwards_since_last_optim = 0

    def reset(self):
        # Clear the cache
        self.cache.clear()
        self.cache = {}

        # Reset the states
        self.backwards_since_reset = 0
        self.training_epoch_when_registered = None
        self.local_optimization_steps = 0
