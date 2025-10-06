import asyncio
from pydantic import BaseModel
import torch
import time
from loguru import logger

from common import settings as common_settings
from subnet.miner_api_client import MinerAPIClient
from miner import settings as miner_settings
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException


class ActivationData(BaseModel):
    activation_id: str
    direction: str
    input_activations: torch.Tensor
    sample_activations: torch.Tensor | None
    output_activations: torch.Tensor | None
    state: dict | None
    upload_time: float

    class Config:
        arbitrary_types_allowed = True


class ActivationCache:
    """
    The ActivationCache is responsible for storing the forward activations that are currently in-process
    so that they are accessible by the backward pass once it is received.
    """

    def __init__(self, miner_api_client: MinerAPIClient):
        self._miner_api_client: MinerAPIClient = miner_api_client
        self._hotkey: str = miner_api_client.hotkey.ss58_address
        self._cache: dict[str, ActivationData] = {}
        self._lock: asyncio.Lock = asyncio.Lock()
        self._sync_lock: asyncio.Lock = asyncio.Lock()

        # Tasks to monitor for exceptions upon reset
        self._removal_tasks: list[asyncio.Task] = []
        self._sync_task: asyncio.Task | None = None

    def __len__(self) -> int:
        """Get the number of activations in the cache."""
        return len(self._cache)

    def __contains__(self, activation_id: str) -> bool:
        """Check if an activation_id is in the cache (enables 'in' operator)."""
        return activation_id in self._cache

    def __getitem__(self, activation_id: str) -> ActivationData:
        """Enable cache[activation_id] syntax for getting items."""
        return self._cache[activation_id]

    def __setitem__(self, activation_id: str, activation_data: ActivationData):
        """Enable cache[activation_id] = data syntax for setting items."""
        self._cache[activation_id] = activation_data

    def __delitem__(self, activation_id: str):
        """Enable del cache[activation_id] syntax for deleting items."""
        self._removal_tasks.append(asyncio.create_task(self.remove(activation_id)))

    async def remove(self, activation_id: str):
        """Remove an activation from the cache."""
        async with self._lock:
            logger.debug(f"🗑️ Removing activation {activation_id} from cache")
            try:
                if activation_id not in self._cache:
                    logger.warning(f"Activation {activation_id} has already been removed from cache")
                    return
                activation_data = self._cache[activation_id]
                if activation_data.input_activations is not None:
                    del activation_data.input_activations
                if activation_data.output_activations is not None:
                    del activation_data.output_activations
                del self._cache[activation_id]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error removing activation {activation_id} from cache: {e}")
                raise

    def is_full(self) -> bool:
        """Check if the cache is full."""
        if len(self._cache) >= miner_settings.ACTIVATION_CACHE_SIZE:
            logger.info(
                f"Miner {self._hotkey[:8]} cache full with {len(self._cache)} activations: {self._cache.keys()}"
            )

            # Clean up inactive activations if not already syncing
            if not self._sync_lock.locked():
                # Run sync concurrently so we don't block the main thread
                self._sync_task = asyncio.create_task(self.sync())

            return True
        return False

    async def sync(self) -> dict[str, bool]:
        """Sync the cache with the orchestrator."""
        async with self._sync_lock:
            try:
                # Clean up inactive activations
                activations_to_remove: dict[str, bool] = await self._miner_api_client.sync_activation_assignments(
                    activation_ids=list(self._cache.keys())
                )

                # Remove inactive activations
                for activation_id, is_active in activations_to_remove.items():
                    if activation_id in self._cache and not is_active:
                        await self.remove(activation_id)
                        logger.info(f"🗑️ Removing inactive activation from cache: {activation_id}")
            except Exception as e:
                logger.warning(f"Error syncing cache: {e}")
                raise

    def cleanup(self):
        """Cleanup the cache of activations that have timed out."""
        if len(self._cache) > 0:
            for activation_id, activation_data in list(self._cache.items()):
                upload_time = activation_data.upload_time
                if upload_time < time.time() - common_settings.ACTIVATION_CACHE_TIMEOUT:
                    # Explicitly remove tensor references to help the gc
                    self._removal_tasks.append(asyncio.create_task(self.remove(activation_id)))
                    logger.warning(f"🗑️ Removing timed out activation from cache: {activation_id}")

    async def reset(self):
        """Reset the cache."""

        # Clear all items in the cache
        activation_ids = list(self._cache.keys())
        for activation_id in activation_ids:
            await self.remove(activation_id)

        # Wait for any sync tasks to complete and raise any exceptions
        try:
            if self._sync_task:
                if not self._sync_task.done():
                    self._sync_task.cancel()
                await asyncio.wait_for(self._sync_task, timeout=1.0)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning("Waiting for cache's sync task to complete has timed out")
            pass
        except (LayerStateException, MinerNotRegisteredException) as e:
            # these will have been handled elsewhere
            pass
        except Exception as e:
            logger.exception(f"Error during cache reset, waiting for sync task to complete: {e}")
            pass  # don't raise, just log - so we'll log an exception instead of an error
        finally:
            self._sync_task = None

        # Wait for any removal tasks to complete and raise any exceptions
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*self._removal_tasks, return_exceptions=True),
                timeout=1.0,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error removing activation: {result}")
                    raise result
        except asyncio.TimeoutError:
            logger.warning("Removal tasks timed out")
            pass
        except (LayerStateException, MinerNotRegisteredException) as e:
            # these will have been handled elsewhere
            pass
        except Exception as e:
            logger.error(f"Error during cache reset, waiting for removal tasks to complete: {e}")
            raise
        finally:
            self._removal_tasks.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
