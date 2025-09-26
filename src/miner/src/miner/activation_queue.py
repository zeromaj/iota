import asyncio
from collections import deque
from loguru import logger
import torch
import time

from common.models.api_models import ActivationResponse
from common import settings as common_settings
from common.utils.exceptions import RateLimitException
from common.models.api_models import GetActivationRequest
from subnet.miner_api_client import MinerAPIClient
from miner.state_manager import StateManager, ActivationData
from miner.utils.activation_utils import download_sample
from subnet.utils.s3_torch import download_tensor
from subnet.model.model_mixin import ModelManager
from miner import settings as miner_settings


class ActivationQueue:
    def __init__(self, miner_api_client: MinerAPIClient, state_manager: StateManager):
        self._miner_api_client: MinerAPIClient = miner_api_client
        self._state_manager: StateManager = state_manager
        self._queue: deque[ActivationData] = deque()
        self._queue_lock: asyncio.Lock = asyncio.Lock()
        self._activation_fetcher_task: asyncio.Task | None = None
        self._model_manager: ModelManager | None = None

    async def peek_next_activation_direction(self) -> str:
        """Peek at the next activation in the queue without removing it."""
        async with self._queue_lock:
            if len(self._queue) > 0:
                return self._queue[0].direction
            else:
                return None

    async def get_activation(self, timeout=-1) -> ActivationData:
        """Get an activation from the queue. If the queue is empty, wait for a new activation to be added."""
        start_time = time.time()
        while True:
            # If the activation fetcher is done, stop it.
            # this will raise any errors that were raised by the activation fetcher
            # (i.e. layer state change errors)
            if self.activation_fetcher_is_done():
                logger.debug("Activation fetcher is done, it should have raised LayerStateException")
                await self.stop_activation_fetcher()  # This should raise LayerStateException
                raise Exception(
                    "Unexpected error: Activation fetcher is done, it should have raised LayerStateException"
                )

            try:
                async with self._queue_lock:
                    logger.debug(
                        f"Activation queue length: {len(self._queue)}: {[a.activation_id for a in self._queue]}"
                    )
                    if len(self._queue) == 0:
                        if timeout > 0 and time.time() - start_time > timeout:
                            raise Exception("Timeout getting activation")
                        logger.debug("Queue is empty, waiting for more activations")
                        await asyncio.sleep(0.1)
                        continue
                    logger.debug(f"Took {time.time() - start_time} seconds to get activation")
                    return self._queue.popleft()
            except Exception as e:
                logger.error(f"Error getting activation: {e}")
                raise

    def activation_fetcher_is_done(self) -> bool:
        """Check if the activation fetcher task has completed."""
        if not self._activation_fetcher_task:
            return True
        return self._activation_fetcher_task.done()

    async def start_activation_fetcher(self, model_manager: ModelManager):
        """Start the activation fetcher task if it's not already running."""
        self._model_manager = model_manager
        if self._activation_fetcher_task and not self._activation_fetcher_task.done():
            return
        self._queue.clear()  # Clear the queue to avoid processing expired activations from previous epoch
        self._activation_fetcher_task = asyncio.create_task(self._fetch_activations())

    async def stop_activation_fetcher(self):
        """Stop the activation fetcher task if it's running and await it."""
        if self._activation_fetcher_task:
            try:
                logger.debug("Awaiting activation fetcher task")
                await self._activation_fetcher_task
                logger.debug("Activation fetcher task completed")
            except Exception as e:
                # Handle the error from the task
                logger.warning(f"Activation fetcher task returned an exception: {e}")
                raise

    async def _fetch_activations(self):
        """Fetch activations from the miner API and add them to the queue."""
        while True:
            # This loop will only break if `get_activation` raises an exception (i.e. LayerStateException)

            # Keep cache clean - TODO: this should probably not be part of the activation queue
            self._state_manager.check_if_timeout(timeout=common_settings.ACTIVATION_CACHE_TIMEOUT)

            queue_slots = miner_settings.MAX_ACTIVATION_QUEUE_SIZE - len(self._queue)
            if queue_slots <= 0:
                await asyncio.sleep(0.1)
                continue

            try:
                logger.debug(f"Requesting {queue_slots} activations")
                response: list[ActivationResponse] = await self._miner_api_client.get_activations(
                    get_activation_request=GetActivationRequest(n_activations=queue_slots)
                )
                logger.debug(f"Received {len(response)} activations")
            except RateLimitException:
                logger.warning("Rate limit exceeded")
                await asyncio.sleep(1)
                continue

            logger.debug(f"Response contains: {[(a.activation_id, a.direction) for a in response]}")
            response = await self._filter_response(response=response)
            logger.debug(
                f"After pruning, downloading {len(response)} activations: {[(a.activation_id, a.direction) for a in response]}"
            )
            download_tasks = [
                self._download_activations(activation_response=activation_response) for activation_response in response
            ]

            for task in asyncio.as_completed(download_tasks):
                activation_response, input_activation = await task
                entry = ActivationData(
                    activation_id=activation_response.activation_id,
                    direction=activation_response.direction,
                    input_activations=input_activation,
                    output_activations=None,
                    state=None,
                    upload_time=time.time(),
                )
                logger.debug(
                    f"Downloaded activation {activation_response.activation_id} going {activation_response.direction}"
                )
                async with self._queue_lock:
                    if activation_response.direction == "backward":
                        self._queue.appendleft(entry)
                    else:
                        self._queue.append(entry)

    async def _download_activations(
        self, activation_response: ActivationResponse
    ) -> tuple[ActivationResponse, torch.Tensor]:
        """Download an activation from the API and return it."""
        # Download the activation
        # TODO: check if we actually need to download samples and tensors DIRECT to CUDA since these are loaded to CUDA
        # during their respective passes either way
        if activation_response.direction == "forward" and self._state_manager.layer == 0:
            input_activations = await download_sample(
                download_url=activation_response.presigned_download_url,
                tokenizer=self._model_manager.tokenizer,
            )
        else:
            input_activations = await download_tensor(
                path=activation_response.presigned_download_url,
                device=miner_settings.DEVICE,
            )
            if not common_settings.MOCK:
                input_activations = input_activations.reshape(
                    common_settings.MINI_BATCH_SIZE,
                    common_settings.SEQUENCE_LENGTH,
                    self._model_manager.model_config.get("bottleneck_dim")
                    or self._model_manager.model_config["emb_dim"],
                )
            else:
                input_activations = input_activations.reshape(
                    common_settings.MINI_BATCH_SIZE,
                    100,
                )
        return (activation_response, input_activations)

    async def _filter_response(self, response: list[ActivationResponse]) -> list[ActivationResponse]:
        """Filter the response to remove any activations that we already have in the cache or queue."""
        # Remove any forward activations that we already have in the cache
        async with self._state_manager.activation_cache_lock:
            response = [
                resp
                for resp in response
                if resp.direction == "backward" or resp.activation_id not in self._state_manager.activation_cache
            ]
        logger.debug(
            f"After filtering with cache, response contains: {[(a.activation_id, a.direction) for a in response]}"
        )

        # Remove any activations that we already have in the queue, with special handling for backward activations
        async with self._queue_lock:
            # Build a map for items in queue
            queue_activation_map = {a.activation_id: (i, a.direction) for i, a in enumerate(self._queue)}

            filtered_response = []
            indices_to_remove = []

            for resp in response:
                activation_id = resp.activation_id
                response_direction = resp.direction

                if activation_id not in queue_activation_map:
                    filtered_response.append(resp)
                else:
                    queue_index, queue_direction = queue_activation_map[activation_id]

                    if response_direction == queue_direction:
                        # Same direction, skip the response activation (keep the one already in queue)
                        continue
                    elif response_direction == "backward" and queue_direction == "forward":
                        # Backward activation takes priority over forward activation
                        # Mark the forward activation for removal from queue and keep the backward response
                        indices_to_remove.append(queue_index)
                        filtered_response.append(resp)
                    else:
                        # Forward response with backward in queue, skip the response (keep backward in queue)
                        continue

            # Remove marked activations from queue (in reverse order to maintain indices)
            for index in sorted(indices_to_remove, reverse=True):
                removed_activation = self._queue[index]
                del self._queue[index]
                logger.debug(
                    f"Removed forward activation {removed_activation.activation_id} from queue to make way for backward activation"
                )

            response = filtered_response

        return response
