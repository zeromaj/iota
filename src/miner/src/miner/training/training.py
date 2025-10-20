from common.models.run_flags import RUN_FLAGS
from common.utils.timer_logger import TimerLogger
import torch
from loguru import logger
import asyncio
import time

from common import settings as common_settings
from common.utils.exceptions import NanInfException
from subnet.utils.vector_utils import check_for_nans_and_infs
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.training.activation_cache import ActivationData, ActivationCache
from miner.training.activation_queue import ActivationQueue
from miner.training.activation_publisher import ActivationPublisher
from subnet.miner_api_client import MinerAPIClient
from subnet.model.model_mixin import ModelManager
from subnet.model.utils import compute_loss, log_gpu_memory_usage
from subnet.model import gpu_device


class TrainingPhase:
    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        state_manager: StateManager,
        model_manager: ModelManager,
    ):
        self._miner_api_client = miner_api_client
        self._state_manager = state_manager
        self._model_manager = model_manager
        self._hotkey = miner_api_client.hotkey.ss58_address
        self._cache: ActivationCache = ActivationCache(miner_api_client=self._miner_api_client)
        self._queue: ActivationQueue = ActivationQueue(
            miner_api_client=self._miner_api_client,
            state_manager=self._state_manager,
            activation_cache=self._cache,
        )
        self._publisher = ActivationPublisher(miner_api_client=self._miner_api_client)
        self.backwards_since_reset = 0
        self.backwards_since_last_optim = 0
        self.local_optimization_steps = 0

    async def run(self):
        try:
            await self._queue.start_activation_fetcher(model_manager=self._model_manager)

            last_activation_time = time.time()
            while True:
                await asyncio.sleep(0.01)  # yield control back to the event loop
                # Check if training phase is complete
                await self._queue.check_if_training_is_complete()

                if self._cache.is_full() and self._queue.next_activation_is_forward():
                    logger.info(f"Activation cache is full. Waiting for backwards activations: {len(self._cache)}")
                    await asyncio.sleep(1)
                    continue

                # Get next activation to process
                activation = await self._queue.get_activation()
                if activation is None:
                    continue

                with logger.contextualize(
                    activation_id=activation.activation_id,
                    time_since_last_activation=time.time() - last_activation_time,
                ):
                    last_activation_time = time.time()
                    if activation.direction == "forward":
                        await self.forward(activation)
                    elif activation.direction == "backward":
                        await self.backward(activation)
                # Loop until LayerStateException is raised by `get_activation`
        except Exception:
            logger.info("Finishing training phase")
            raise
        finally:
            # TODO: @cassova: determine if we want to add an optimization step here too
            # considering the last activation submission may have failed.
            await self.optimization_reset()
            log_gpu_memory_usage(note="after training phase cleanup")

    async def forward(self, activation_data: ActivationData):
        """
        Performs the forward pass.

        If the layer is 0, it will load the data and upload the initial activation to the API.
        If the layer is not 0, it will download a random forward activation from the API and perform the forward pass.

        The forward pass contains:
        - Downloading the forward activation from the API
        - Performing the forward pass
        - Reporting the loss to the API
        - Performing the backward pass
        """
        with logger.contextualize(cache_size=len(self._cache)):
            async with TimerLogger(
                name="forward",
                metadata={
                    "hotkey": self._hotkey[:8],
                    "activation_id": activation_data.activation_id,
                    "layer": self._state_manager.layer,
                },
            ):
                logger.info(
                    f"🚀 Starting FORWARD pass for layer {self._state_manager.layer} | Processing activation {activation_data.activation_id} | Miner: {self._hotkey[:8]}"
                )
                log_gpu_memory_usage(note="starting training forward pass")
                if self._state_manager.layer == 0:
                    logger.debug(f"Got sample shape: {activation_data.input_activations.shape}")
                else:
                    logger.debug(f"Got activation shape: {activation_data.input_activations.shape}")

                # Perform the actual forward pass

                logger.debug(f"Forwarding activation of size {activation_data.input_activations.shape}")
                self._model_manager.model = self._model_manager.model.to(miner_settings.DEVICE)
                input_activations_gpu = activation_data.input_activations.to(miner_settings.DEVICE)
                output_activations_gpu, state = await self._model_manager._forward(
                    layer=self._state_manager.layer, input_activations=input_activations_gpu
                )
                log_gpu_memory_usage(note="after training forward pass")

                # we'll put a copy of the output activations on CPU
                activation_data.input_activations = input_activations_gpu  # keep it on the gpu while in cache
                activation_data.output_activations = None
                activation_data.state = state
                activation_data.upload_time = time.time()
                self._cache[activation_data.activation_id] = activation_data

                if self._state_manager.layer == self._model_manager.model_metadata["n_splits"] - 1:
                    # Compute loss; if targets download or loss computation fails, skip backward gracefully
                    try:
                        loss = await self.compute_last_layer_loss(
                            activation_data=activation_data, logits=output_activations_gpu
                        )
                    except Exception as e:
                        logger.exception(
                            f"Skipping backward for activation {activation_data.activation_id} due to loss/target fetch error: {e}"
                        )
                        return
                    log_gpu_memory_usage(
                        note=f"after training forward pass cleaning on last layer miner with cach size of {len(self._cache)}"
                    )
                    return await self.backward(activation_data=activation_data, loss=loss)

                # Cleanup GPU memory
                output_activations_cpu = output_activations_gpu.detach().cpu()
                del output_activations_gpu

                # If we are not on the last layer, we just need to upload the activations
                logger.info(
                    f"output activations before upload with shape {output_activations_cpu.shape} for {self._hotkey[:8]} on layer {self._state_manager.layer}"
                )

                self._publisher.publish_activation(
                    tensor=output_activations_cpu,
                    activation_id=activation_data.activation_id,
                    direction="forward",
                    attestation_challenge_blob=activation_data.attestation_challenge_blob,
                    upload_url=activation_data.upload_url,
                    activation_path=activation_data.activation_upload_path,
                )

                log_gpu_memory_usage(note="after training forward pass cleaning on non-last layer miner")
                logger.success(
                    f"✅ Successfully completed FORWARD pass for activation {activation_data.activation_id} on layer {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
                )

    async def backward(self, activation_data: ActivationData, loss: torch.Tensor = None):
        """
        Performs the backward pass.
        """
        with logger.contextualize(cache_size=len(self._cache)):
            async with TimerLogger(
                name="backward", metadata={"hotkey": self._hotkey[:8], "activation_id": activation_data.activation_id}
            ):
                logger.info(
                    f"🔄 Starting BACKWARD pass for activation {activation_data.activation_id} | Layer: {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
                )
                async with TimerLogger(name="moving to gpu"):
                    log_gpu_memory_usage(note="starting training backward pass")

                    # Check if activation is in cache
                    if activation_data.activation_id not in self._cache:
                        logger.warning(
                            f"⚠️ Activation {activation_data.activation_id} not found in cache, skipping backward pass"
                        )
                        return
                    cached_activations = self._cache[activation_data.activation_id]

                    # Move to GPU and enable gradients only for floating point tensors
                    self._model_manager.model = self._model_manager.model.to(miner_settings.DEVICE)
                    input_activations_gpu = cached_activations.input_activations.to(miner_settings.DEVICE)
                    activation_grads_gpu = activation_data.input_activations.to(miner_settings.DEVICE)
                    if loss is None:
                        # Recalculate the output activations - these will be on GPU
                        output_activations_gpu, _ = await self._model_manager._forward(
                            layer=self._state_manager.layer, input_activations=input_activations_gpu
                        )
                    else:
                        # Use the loss from the last layer and copy it to GPU if it's not there already
                        output_activations_gpu = loss.to(miner_settings.DEVICE)

                    log_gpu_memory_usage(note="after preparing activations on training backward pass")

                    logger.info(
                        f"output activations before backward with shape {output_activations_gpu.shape} for {self._hotkey[:8]} on layer {self._state_manager.layer}"
                    )
                async with TimerLogger(name="backward pass"):
                    await self._model_manager._backward(
                        layer=self._state_manager.layer,
                        output_activations=output_activations_gpu,
                        activation_grads=activation_grads_gpu,
                        state=cached_activations.state,
                    )
                log_gpu_memory_usage(note="after backward pass")

            async with TimerLogger(name="publishing_backwards"):
                self.backwards_since_reset += 1
                logger.debug(f"Backwards since reset for miner {self._hotkey[:8]}: {self.backwards_since_reset}")

                # Handle different cases for input activation gradients
                if common_settings.MOCK:
                    input_activation_grads = input_activations_gpu.detach().to(torch.bfloat16).cpu()

                elif self._state_manager.layer == 0:
                    # Get the embedding layer weight grads instead of the input activations grads
                    # This is because input activation grads of the first layer do not exist.
                    emb_weight = self._model_manager.model.tok_emb.weight
                    embedding_dim = (
                        self._model_manager.model_config["bottleneck_dim"]
                        or self._model_manager.model_config["emb_dim"]
                    )
                    # Detach and convert to bfloat16 to ensure we only save the values
                    # NOTE: if we cast to bfloat16 after moving to CPU, there will be extra load if the grad is huge
                    # but if we do it before, we'll be taking up additional GPU memory
                    grad_flattened = emb_weight.grad.detach().cpu().to(torch.bfloat16).flatten()
                    input_activation_grads = grad_flattened[
                        : common_settings.SEQUENCE_LENGTH * embedding_dim * common_settings.MINI_BATCH_SIZE
                    ]
                else:
                    input_activation_grads = input_activations_gpu.grad.detach().cpu()

                log_gpu_memory_usage(note="after moving input activation grads to GPU")

                self._publisher.publish_activation(
                    tensor=input_activation_grads,
                    activation_id=activation_data.activation_id,
                    direction="backward",
                    attestation_challenge_blob=activation_data.attestation_challenge_blob,
                    upload_url=activation_data.upload_url,
                    activation_path=activation_data.activation_upload_path,
                )

            async with TimerLogger(name="cleaning up cache"):
                # Cleanup cache
                del self._cache[activation_data.activation_id]
                del cached_activations

                # Cleanup GPU memory
                del output_activations_gpu, input_activations_gpu, activation_grads_gpu, loss

                with logger.contextualize(cache_size=len(self._cache)):
                    log_gpu_memory_usage(note="after training backward pass cleaning")

                    # Check if we need to perform a local optimization step
                    self.backwards_since_last_optim += 1
                    if self.backwards_since_last_optim >= common_settings.MINI_BATCH_ACCUMULATION_COUNT:
                        logger.info(
                            f"🔄 Miner {self._hotkey[:8]} performing local optimization step after {common_settings.MINI_BATCH_ACCUMULATION_COUNT} backward passes"
                        )
                        learning_rate = await self._miner_api_client.get_learning_rate()
                        await self._model_manager.local_optimization_step(learning_rate=learning_rate)
                        await self.optimization_reset()

                        log_gpu_memory_usage(note="after local optimization step")

                        self.local_optimization_steps += 1
                        logger.success(
                            f"✅ Miner {self._hotkey[:8]} completed local optimization step #{self.local_optimization_steps}"
                        )

                    logger.success(
                        f"✅ Successfully completed BACKWARD pass for activation {activation_data.activation_id} | Layer: {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
                    )

    async def compute_last_layer_loss(self, activation_data: ActivationData, logits: torch.Tensor) -> torch.Tensor:
        """
        Performs the backward pass for the last layer.
        """
        async with TimerLogger(
            name="compute_last_layer_loss",
            metadata={
                "hotkey": self._hotkey[:8],
                "activation_id": activation_data.activation_id,
                "layer": self._state_manager.layer,
            },
        ):
            # Target sample is the initial activations
            targets = activation_data.sample_activations
            logger.debug(f"Downloaded targets with shape {targets.shape} and dtype {targets.dtype}")

            # NOTE: targets are on the CPU at this point
            # the problem is that loss calculation is very heavy on the GPU memory
            # on A4000 a 1B, when on GPU, it took 0.03s to compute the loss - on the CPU performance it took 0.5s
            device = miner_settings.DEVICE
            if device != "cpu":
                gpu_device.synchronize()
                gpu_device.empty_cache()
                proxy_bytes_needed = logits.numel() * logits.element_size() * 5
                avail_memory = gpu_device.available_memory()
                if proxy_bytes_needed > avail_memory:
                    device = "cpu"
                    logger.warning(
                        "Not enough memory available to compute loss on GPU"
                        f" - needed {proxy_bytes_needed / 1024**3:.2f}GB, available {avail_memory / 1024**3:.2f}GB"
                    )

            loss: torch.Tensor = compute_loss(
                mock=common_settings.MOCK,
                logits=logits,
                targets=targets,
                vocab_size=self._model_manager.vocab_size,
                pad_token_id=self._model_manager.eos_token_id,
                pack=miner_settings.PACK_SAMPLES,
                device=device,
            )

            check_for_nans_and_infs(
                tensor=loss, name=f"Loss for miner {self._hotkey[:8]}", exception_type=NanInfException
            )

            logger.info(
                f"📊 Computed loss {loss:.6f} for activation {activation_data.activation_id} | Layer: {self._state_manager.layer} | Miner: {self._hotkey[:8]}"
            )

            # Update cache with loss before attempting to report it to handle API errors gracefully
            activation_data.upload_time = time.time()
            self._cache[activation_data.activation_id] = activation_data

            self._publisher.publish_loss(loss=loss.item(), activation_id=activation_data.activation_id)

            return loss

    async def reset(self):
        """Reset the training phase."""
        logger.debug("🗑️ Performing full reset of training")
        self.backwards_since_reset = 0
        await self.optimization_reset()
        await self._publisher.reset()

    async def optimization_reset(self):
        """Reset the cache and backward pass counter after performing optimization step."""
        logger.debug("🗑️ Resetting after optimization step")
        self.backwards_since_last_optim = 0

        # we can't process backwards activations on forwards processed before the optimization step
        if RUN_FLAGS.keep_cache_on_local_step.isOff():
            await self._cache.reset()
        log_gpu_memory_usage(note="after cache reset")
