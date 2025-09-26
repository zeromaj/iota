import torch
from loguru import logger
import asyncio
import time

from common import settings as common_settings
from common.models.api_models import (
    CompleteFileUploadResponse,
    GetTargetsRequest,
    SubmitActivationRequest,
    LossReportRequest,
)
from common.utils.exceptions import NanInfException
from miner.utils.activation_utils import download_sample
from subnet.utils.vector_utils import check_for_nans_and_infs
from miner.utils.utils import upload_tensor
from miner import settings as miner_settings
from miner.state_manager import ActivationData, StateManager
from subnet.miner_api_client import MinerAPIClient
from subnet.model.model_mixin import ModelManager
from miner.activation_queue import ActivationQueue
from subnet.model.utils import compute_loss


class TrainingPhase:
    def __init__(
        self,
        miner_api_client: MinerAPIClient,
        state_manager: StateManager,
        model_manager: ModelManager,
    ):
        self.miner_api_client = miner_api_client
        self.state_manager = state_manager
        self.model_manager = model_manager
        self.hotkey = miner_api_client.hotkey.ss58_address
        self.activation_queue: ActivationQueue = ActivationQueue(
            miner_api_client=self.miner_api_client, state_manager=self.state_manager
        )

    async def run(self):
        await self.activation_queue.start_activation_fetcher(model_manager=self.model_manager)

        while True:
            direction = await self.activation_queue.peek_next_activation_direction()
            if (
                direction is not None
                and direction == "forward"
                and await self.state_manager.activation_cache_is_full(miner_api_client=self.miner_api_client)
            ):
                logger.info("Activation cache is full. Waiting for backwards activations.")
                await asyncio.sleep(0.1)
                continue

            activation = await self.activation_queue.get_activation()

            if activation.direction == "forward":
                await self.forward(activation)
            elif activation.direction == "backward":
                await self.backward(activation)
            # Loop until LayerStateException is raised by `get_activation`

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
        logger.info(
            f"🚀 Starting FORWARD pass for layer {self.state_manager.layer} | Processing activation {activation_data.activation_id} | Miner: {self.hotkey[:8]}"
        )
        if self.state_manager.layer == 0:
            logger.debug(f"Got sample shape: {activation_data.input_activations.shape}")
        else:
            logger.debug(f"Got activation shape: {activation_data.input_activations.shape}")

        # Perform the actual forward pass

        logger.debug(f"Forwarding activation of size {activation_data.input_activations.shape}")
        output_activations, state = await self.model_manager._forward(
            layer=self.state_manager.layer, input_activations=activation_data.input_activations
        )

        activation_data.output_activations = output_activations
        activation_data.state = state
        activation_data.upload_time = time.time()

        await self.state_manager.add_to_activation_cache(
            activation_id=activation_data.activation_id,
            data=activation_data,
        )

        if self.state_manager.layer == self.model_manager.model_metadata["n_splits"] - 1:
            # Compute loss; if targets download or loss computation fails, skip backward gracefully
            try:
                await self.compute_last_layer_loss(activation_data=activation_data)
            except Exception as e:
                logger.warning(
                    f"Skipping backward for activation {activation_data.activation_id} due to loss/target fetch error: {e}"
                )
                return

            return await self.backward(activation_data=activation_data)

        # If we are not on the last layer, we just need to upload the activations
        logger.info(
            f"output activations before upload with shape {output_activations.shape} for {self.hotkey[:8]} on layer {self.state_manager.layer}"
        )
        upload_response: CompleteFileUploadResponse = await upload_tensor(
            miner_api_client=self.miner_api_client,
            tensor=output_activations.detach().clone(),
            hotkey=self.miner_api_client.hotkey,
        )

        await self.miner_api_client.submit_activation_request(
            submit_activation_request=SubmitActivationRequest(
                activation_id=activation_data.activation_id,
                activation_path=upload_response.object_path,
                direction="forward",
            ),
        )
        logger.info(
            f"✅ Successfully completed FORWARD pass for activation {activation_data.activation_id} on layer {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

    async def backward(self, activation_data: ActivationData):
        """
        Performs the backward pass.
        """
        logger.info(
            f"🔄 Starting BACKWARD pass for activation {activation_data.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

        # Check if activation is in cache
        if activation_data.activation_id not in self.state_manager.activation_cache:
            logger.warning(f"⚠️ Activation {activation_data.activation_id} not found in cache, skipping backward pass")
            return
        cached_activations = self.state_manager.activation_cache[activation_data.activation_id]

        # Move to GPU and enable gradients only for floating point tensors
        input_activations: torch.Tensor = cached_activations.input_activations.to(miner_settings.DEVICE)
        output_activations: torch.Tensor = cached_activations.output_activations.to(miner_settings.DEVICE)

        state = cached_activations.state

        logger.info(
            f"output activations before backward with shape {output_activations.shape} for {self.hotkey[:8]} on layer {self.state_manager.layer}"
        )
        await self.model_manager._backward(
            layer=self.state_manager.layer,
            output_activations=output_activations,
            activation_grads=activation_data.input_activations,
            state=state,
        )

        self.state_manager.backwards_since_reset += 1
        logger.debug(f"Backwards since reset for miner {self.hotkey[:8]}: {self.state_manager.backwards_since_reset}")
        # Handle different cases for input activation gradients
        if common_settings.MOCK:
            input_activation_grads = input_activations.detach().to(torch.bfloat16).cpu()

        elif self.state_manager.layer == 0:
            # Get the embedding layer weight grads instead of the input activations grads
            # This is because input activation grads of the first layer do not exist.
            emb_weight = self.model_manager.model.tok_emb.weight
            embedding_dim = (
                self.model_manager.model_config["bottleneck_dim"]
                if self.model_manager.model_config["bottleneck_dim"] is not None
                else self.model_manager.model_config["emb_dim"]
            )
            grad_flattened = emb_weight.grad.clone().flatten()
            input_activation_grads = grad_flattened[
                : common_settings.SEQUENCE_LENGTH * embedding_dim * common_settings.MINI_BATCH_SIZE
            ]

            # Detach and convert to bfloat16 to ensure we only save the values
            input_activation_grads = input_activation_grads.detach().to(torch.bfloat16).cpu()

        else:
            input_activation_grads = input_activations.grad

        upload_response: CompleteFileUploadResponse = await upload_tensor(
            miner_api_client=self.miner_api_client,
            tensor=input_activation_grads,
            hotkey=self.miner_api_client.hotkey,
        )

        logger.info(
            f"input activation grads before upload with shape {input_activation_grads.shape} for {self.hotkey[:8]} on layer {self.state_manager.layer}"
        )
        await self.miner_api_client.submit_activation_request(
            submit_activation_request=SubmitActivationRequest(
                activation_id=activation_data.activation_id,
                activation_path=upload_response.object_path,
                direction="backward",
            ),
        )
        # Remove from cache
        await self.state_manager.remove_from_activation_cache(activation_data.activation_id)

        # Check if we need to perform a local optimization step
        if self.state_manager.increment_backward_count():
            logger.info(
                f"🔄 Miner {self.hotkey[:8]} performing local optimization step after {common_settings.MINI_BATCH_ACCUMULATION_COUNT} backward passes"
            )
            learning_rate = await self.miner_api_client.get_learning_rate()
            await self.model_manager.local_optimization_step(learning_rate=learning_rate)
            self.state_manager.reset_optimization_counter()

            # Remove all activations from cache
            self.state_manager.activation_cache.clear()

            self.state_manager.local_optimization_steps += 1
            logger.info(
                f"✅ Miner {self.hotkey[:8]} completed local optimization step #{self.state_manager.local_optimization_steps}"
            )

        logger.info(
            f"✅ Successfully completed BACKWARD pass for activation {activation_data.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

    async def compute_last_layer_loss(self, activation_data: ActivationData):
        """
        Performs the backward pass for the last layer.
        """

        initial_activations_path = await self.miner_api_client.get_targets(
            get_targets_request=GetTargetsRequest(activation_id=activation_data.activation_id),
        )

        # Target sample is the initial activations
        # Target sample is the initial activations
        targets = await download_sample(download_url=initial_activations_path, tokenizer=self.model_manager.tokenizer)
        logger.debug(f"Downloaded targets: {targets}")
        logger.debug(f"Targets shape: {targets.shape}")
        logger.debug(f"Targets dtype: {targets.dtype}")

        loss: torch.Tensor = compute_loss(
            mock=common_settings.MOCK,
            logits=activation_data.output_activations,
            targets=targets,
            vocab_size=self.model_manager.vocab_size,
            pad_token_id=self.model_manager.eos_token_id,
            pack=miner_settings.PACK_SAMPLES,
        )

        check_for_nans_and_infs(tensor=loss, name=f"Loss for miner {self.hotkey[:8]}", exception_type=NanInfException)

        logger.info(
            f"📊 Computed loss {loss:.6f} for activation {activation_data.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

        # Update cache with loss before attempting to report it to handle API errors gracefully
        activation_data.output_activations = loss
        activation_data.upload_time = time.time()
        await self.state_manager.add_to_activation_cache(
            activation_id=activation_data.activation_id,
            data=activation_data,
        )

        try:
            loss_copy: torch.Tensor = loss.clone().detach()
            await self.miner_api_client.report_loss(
                loss_report=LossReportRequest(activation_id=activation_data.activation_id, loss=loss_copy.item()),
            )

        except Exception as e:
            logger.error(f"Error reporting loss: {e}")
