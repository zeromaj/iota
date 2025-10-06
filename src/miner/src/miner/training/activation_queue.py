import asyncio
from collections import deque
from loguru import logger
import torch
import time
from pydantic import BaseModel

from common.models.api_models import ActivationResponse, GetTargetsRequest, GetActivationRequest
from common import settings as common_settings
from common.utils.exceptions import RateLimitException
from subnet.miner_api_client import MinerAPIClient
from miner.training.activation_cache import ActivationCache, ActivationData
from miner.state_manager import StateManager
from miner.utils.activation_utils import download_sample
from subnet.utils.s3_torch import download_tensor
from subnet.model.model_mixin import ModelManager
from miner import settings as miner_settings
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException


class DownloadedData(BaseModel):
    activation_response: ActivationResponse
    input_activations: torch.Tensor
    sample_activations: torch.Tensor | None = None

    class Config:
        arbitrary_types_allowed = True


class ActivationQueue:
    """
    The ActivationQueue is responsible for fetching and storing activations before they are ready
    to be processed by the training phase. We specify the number of forward activations but not that
    backward activations since those are tied to the miner who processed the forward activation.
    """

    def __init__(
        self, miner_api_client: MinerAPIClient, state_manager: StateManager, activation_cache: ActivationCache
    ):
        self._miner_api_client: MinerAPIClient = miner_api_client
        self._state_manager: StateManager = state_manager
        self._cache: ActivationCache = activation_cache

        self._queue_lock: asyncio.Lock = asyncio.Lock()
        self._forward_queue: deque[ActivationData] = deque()
        self._backward_queue: deque[ActivationData] = deque()
        self._activation_fetcher_task: asyncio.Task | None = None
        self._model_manager: ModelManager | None = None

    def __len__(self) -> int:
        """Get the number of activations in the queue."""
        return len(self._backward_queue) + len(self._forward_queue)

    def next_activation_is_forward(self) -> bool:
        """Peek at the next activation in the queue without removing it."""
        if len(self._backward_queue) == 0 and len(self._forward_queue) > 0:
            return True
        return False

    async def check_if_training_is_complete(self) -> bool:
        """Check if training is complete by checking if the activation fetcher task has completed."""
        # If the activation fetcher is done, stop it.
        # this will raise any errors that were raised by the activation fetcher
        # (i.e. layer state change errors)
        if self.activation_fetcher_is_done():
            logger.debug("Activation fetcher is done")
            await self.stop_activation_fetcher()  # This should raise LayerStateException
            raise Exception("Unexpected error: Activation fetcher is done, it should have raised LayerStateException")

        logger.debug("Activation fetcher is not done, training will continue")  # produces too many logs
        return False

    async def get_activation(self, timeout=-1) -> ActivationData:
        """Get an activation from the queue. If the queue is empty, wait for a new activation to be added."""
        start_time = time.time()
        while True:
            try:
                # Check if the activation fetcher task has completed with an exception
                await self.check_if_training_is_complete()  # This will raise any exception from the background task

                async with self._queue_lock:
                    if len(self._backward_queue) + len(self._forward_queue) > 0:
                        logger.debug(
                            f"Activation queue length: {len(self._backward_queue) + len(self._forward_queue)}: "
                            f"backward: {[a.activation_id for a in self._backward_queue]} "
                            f"forward: {[a.activation_id for a in self._forward_queue]}"
                        )
                        logger.debug(f"Cache status: {len(self._cache)}")
                        if len(self._backward_queue) > 0:
                            logger.debug(f"Took {time.time() - start_time} seconds to get backward activation")
                            return self._backward_queue.popleft()
                        if len(self._forward_queue) > 0 and not self._cache.is_full():
                            logger.debug(f"Took {time.time() - start_time} seconds to get forward activation")
                            return self._forward_queue.popleft()

                # Wait for more activations
                if timeout > 0 and time.time() - start_time > timeout:
                    raise Exception("Timeout getting activation")
                # logger.debug("Queue is empty, waiting for more activations")  # produces too many logs
                await asyncio.sleep(0.1)
                continue
            except Exception as e:
                logger.error(f"Error getting activation from queue: {e}")
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
            logger.warning("Activation fetcher task already running")
            return
        self._backward_queue.clear()  # Clear the backward queue
        self._forward_queue.clear()  # Clear the forward queue to avoid processing expired activations from previous epoch
        self._activation_fetcher_task = asyncio.create_task(self._fetch_activations())
        logger.debug("Activation fetcher task started")

    async def stop_activation_fetcher(self):
        """Stop the activation fetcher task if it's running and await it."""
        if self._activation_fetcher_task:
            try:
                logger.debug("Awaiting activation fetcher task")
                await self._activation_fetcher_task
                logger.error(
                    "Activation fetcher task completed - this message should never be logged bcs we expect a LayerStateException"
                )
            except Exception as e:
                # Handle the error from the task
                logger.warning(f"Activation fetcher task returned an exception: {e}")
                raise

    async def _fetch_activations(self):
        """Fetch activations from the miner API and add them to the queue."""
        while True:
            # This loop will only break if `get_activation` raises an exception (i.e. LayerStateException)

            # Keep cache clean
            self._cache.cleanup()

            async with self._queue_lock:
                logger.debug(f"Max queue size: {miner_settings.MAX_ACTIVATION_QUEUE_SIZE}")
                queue_status = f"backward: {[a.activation_id for a in self._backward_queue]} forward: {[a.activation_id for a in self._forward_queue]}"
                logger.debug(f"Queue status: {queue_status}")
                queue_slots = (
                    miner_settings.MAX_ACTIVATION_QUEUE_SIZE - len(self._backward_queue) - len(self._forward_queue)
                )

            missing_backwards = len(self._cache) - len(self._backward_queue)
            n_fwd_activations = queue_slots - missing_backwards  # Leave room for missing backwards activations
            if n_fwd_activations < 0:
                n_fwd_activations = 0
            logger.debug(
                f"Queue slots available: {queue_slots}"
                f" -- Missing backwards: {missing_backwards}"
                f" -- Forward activation reqs: {n_fwd_activations}"
            )

            try:
                response: list[ActivationResponse] = await self._miner_api_client.get_activations(
                    get_activation_request=GetActivationRequest(n_fwd_activations=n_fwd_activations)
                )
                logger.debug(f"Received activations: {len(response)}")
            except RateLimitException:
                logger.warning("Rate limit exceeded")
                await asyncio.sleep(1)
                continue
            except LayerStateException as e:
                logger.warning(f"Layer state changing while getting activations: {e}")
                raise
            except MinerNotRegisteredException as e:
                logger.warning(f"Miner no longer registered while getting activations: {e}")
                raise
            except Exception as e:
                logger.exception(f"Error getting activations from orchestrator: {e}")
                raise

            if len(response) == 0:
                logger.warning("No activations received from orchestrator")
                continue

            logger.debug(f"Response contains: {[(a.activation_id, a.direction) for a in response]}")

            # Filter the response
            response = await self._filter_duplicates(response=response)  # do this before we split
            backward_response, forward_response = await self._split_responses(response=response)
            logger.debug(
                f"Forward response prior to excess filtering: {[(a.activation_id, a.direction) for a in forward_response]}"
            )
            forward_response = await self._filter_excess_forwards(forward_response=forward_response)

            if len(backward_response) == 0 and len(forward_response) == 0:
                logger.warning("No activations to download")
                continue

            logger.debug(
                f"After filtering, downloading activations {len(backward_response)} backward: {[(a.activation_id, a.direction) for a in backward_response]}"
            )
            logger.debug(
                f"After filtering, downloading activations: {len(forward_response)} forward: {[(a.activation_id, a.direction) for a in forward_response]}"
            )

            # Download the activations
            download_tasks = [
                asyncio.create_task(self._download_activations(activation_response=r)) for r in backward_response
            ]
            download_tasks.extend(
                [asyncio.create_task(self._download_activations(activation_response=r)) for r in forward_response]
            )
            logger.debug(f"Downloading {len(download_tasks)} activations")

            completed_tasks = set()
            for task in asyncio.as_completed(download_tasks):
                try:
                    downloaded_data: DownloadedData = await task
                    completed_tasks.add(task)
                except asyncio.TimeoutError:
                    logger.warning("Timeout downloading activation -- skipping")
                    continue
                except asyncio.CancelledError:
                    logger.warning("Download task cancelled -- skipping")
                    continue
                except (LayerStateException, MinerNotRegisteredException) as e:
                    logger.warning(f"Anticipated exception has occurred while downloading activations: {e}")
                    self._cancel_tasks(tasks=download_tasks, completed_tasks=completed_tasks)
                    raise
                except Exception as e:
                    logger.error(f"Error downloading activation -- skipping: {e}")
                    continue
                activation_response = downloaded_data.activation_response
                entry = ActivationData(
                    activation_id=activation_response.activation_id,
                    direction=activation_response.direction,
                    input_activations=downloaded_data.input_activations,
                    sample_activations=downloaded_data.sample_activations,
                    output_activations=None,
                    state=None,
                    upload_time=time.time(),
                )
                logger.debug(
                    f"Downloaded activation {activation_response.activation_id} going {activation_response.direction}"
                )
                async with self._queue_lock:
                    if activation_response.direction == "backward":
                        self._backward_queue.append(entry)
                    else:
                        self._forward_queue.append(entry)

    async def _download_activations(self, activation_response: ActivationResponse) -> DownloadedData:
        """Download an activation from the API and return it."""
        # TODO: check if we actually need to download samples and tensors DIRECT to CUDA since these are loaded to CUDA
        # during their respective passes either way
        with logger.contextualize(activation_id=activation_response.activation_id):
            try:
                # Download the input activations
                if activation_response.direction == "forward" and self._state_manager.layer == 0:
                    input_activations = await asyncio.wait_for(
                        download_sample(
                            download_url=activation_response.presigned_download_url,
                            tokenizer=self._model_manager.tokenizer,
                            device="cpu",
                        ),
                        timeout=60,  # TODO: @cassova: this value should change based on activation size
                    )
                else:
                    input_activations = await asyncio.wait_for(
                        download_tensor(
                            path=activation_response.presigned_download_url,
                            device="cpu",
                        ),
                        timeout=60,  # TODO: @cassova: this value should change based on activation size
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

                # Download the sample for last layer miners as well
                sample_activations = None
                if (
                    activation_response.direction == "forward"
                    and self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1
                ):
                    logger.debug("Last layer miner, downloading sample activations")
                    initial_activations_path = await self._miner_api_client.get_targets(
                        get_targets_request=GetTargetsRequest(activation_id=activation_response.activation_id),
                    )
                    sample_activations = await asyncio.wait_for(
                        download_sample(
                            download_url=initial_activations_path,
                            tokenizer=self._model_manager.tokenizer,
                            device="cpu",
                        ),
                        timeout=60,  # TODO: @cassova: this value should change based on activation size
                    )
                return DownloadedData(
                    activation_response=activation_response,
                    input_activations=input_activations,
                    sample_activations=sample_activations,
                )
            except (
                asyncio.TimeoutError,
                asyncio.CancelledError,
                LayerStateException,
                MinerNotRegisteredException,
            ):
                # Just raise these expected errors to be caught by the caller
                raise
            except Exception as e:
                # For these unexpected errors, we want the stack trace
                logger.exception(f"Failed downloading activation {activation_response.activation_id}: {e}")
                raise

    async def _split_responses(
        self, response: list[ActivationResponse]
    ) -> tuple[list[ActivationResponse], list[ActivationResponse]]:
        """Split the response into backward and forward activations."""
        backward_response = [resp for resp in response if resp.direction == "backward"] if response else []
        forward_response = [resp for resp in response if resp.direction == "forward"] if response else []
        return backward_response, forward_response

    async def _filter_duplicates(self, response: list[ActivationResponse]) -> list[ActivationResponse]:
        """Filter the response to remove any activations that we already have in the cache or queue."""
        # Remove any forward activations that we already have in the cache
        async with self._cache._lock:
            response = [
                resp for resp in response if resp.direction == "backward" or resp.activation_id not in self._cache
            ]
        logger.debug(
            f"After filtering with cache, response contains: {[(a.activation_id, a.direction) for a in response]}"
        )

        # Remove any activations that we already have in the queue, with special handling for backward activations
        # ex1. we already have the forward in the queue so we can remove it from the response
        # ex2. we already have the forward in the queue but we received its backward for it so we should remove the forward from the queue and add the backward
        async with self._queue_lock:
            # Build a map for items in queue
            forward_queue_activation_map = {a.activation_id: i for i, a in enumerate(self._forward_queue)}
            backward_queue_activation_map = {a.activation_id: i for i, a in enumerate(self._backward_queue)}

            filtered_response = []
            indices_to_remove = []

            for resp in response:
                activation_id = resp.activation_id
                response_direction = resp.direction

                if (
                    activation_id not in forward_queue_activation_map
                    and activation_id not in backward_queue_activation_map
                ):
                    filtered_response.append(resp)
                else:
                    forward_queue_index = forward_queue_activation_map.get(activation_id, None)
                    backward_queue_index = backward_queue_activation_map.get(activation_id, None)

                    if response_direction == "forward" and forward_queue_index is not None:
                        # Same direction, skip the response activation (keep the one already in queue)
                        continue
                    elif response_direction == "backward" and backward_queue_index is not None:
                        # Same direction, skip the response activation (keep the one already in queue)
                        continue
                    elif response_direction == "backward" and forward_queue_index is not None:
                        # Backward activation takes priority over forward activation - delete the forward and store the backward
                        # Mark the forward activation for removal from queue and keep the backward response
                        indices_to_remove.append(forward_queue_index)
                        filtered_response.append(resp)
                    else:
                        # Forward response with backward in queue, skip the response (keep backward in queue)
                        continue

            # Remove marked activations from forward queue (in reverse order to maintain indices)
            for index in sorted(indices_to_remove, reverse=True):
                removed_activation = self._forward_queue[index]
                del self._forward_queue[index]
                logger.debug(
                    f"Removed forward activation {removed_activation.activation_id} from queue to make way for backward activation"
                )

            response = filtered_response

        return response

    async def _filter_excess_forwards(self, forward_response: list[ActivationResponse]) -> list[ActivationResponse]:
        """Remove excess forward activations to make sure we leave room for backwards activations in the queue."""
        async with self._queue_lock:
            max_forwards_in_process = miner_settings.MAX_ACTIVATION_QUEUE_SIZE
            forwards_in_process = len(self._forward_queue) + len(self._cache)
            received_forwards = len(forward_response)

            # If we have too many forwards in process, discard all forward responses
            if forwards_in_process >= max_forwards_in_process:
                logger.debug("Removing all forward activations from response to make way for backward activations")
                return []

            # If the response contains too many forwards, crop it down to the max
            removal_amount = received_forwards + forwards_in_process - max_forwards_in_process
            if removal_amount > 0:
                forward_response = forward_response[:-removal_amount]
                logger.debug(
                    f"Removed {removal_amount} forward activations from response to make way for backward activations"
                )
            return forward_response

    def _cancel_tasks(self, tasks: list[asyncio.Task], completed_tasks: set[asyncio.Task] = set()):
        # Cancel all tasks
        for t in tasks:
            if t not in completed_tasks and not t.done():
                t.cancel()
