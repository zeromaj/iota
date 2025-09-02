import sys
import asyncio
from loguru import logger
import json
import time
import torch
import aiohttp
from bittensor import Wallet

from subnet.common_api_client import CommonAPIClient
from miner.health_server import HealthServerMixin
from miner.utils.activation_utils import download_sample
from miner.utils.partition_merging import (
    download_partition,
    filter_bad_metadata,
    get_weight_partition_info,
)
from miner import settings as miner_settings
from miner.state_manager import CacheEntry, StateManager
from miner.utils.utils import (
    create_metadata,
    extract_filename_from_url,
    upload_file,
    upload_tensor,
    wait_for_state,
)

from common import settings as common_settings
from common.models.api_models import (
    ActivationResponse,
    CompleteFileUploadResponse,
    GetTargetsRequest,
    LossReportRequest,
    MinerRegistrationResponse,
    SubmitActivationRequest,
    SubmittedWeightsAndOptimizerPresigned,
    WeightUpdate,
)
from common.utils.exceptions import (
    APIException,
    LayerStateException,
    MinerNotRegisteredException,
    NanInfException,
    NanInfWarning,
    SpecVersionException,
    SubmittedWeightsError,
)
from common.models.miner_models import ChunkMetadata
from common.utils.partitions import MinerPartition
from common.utils.shared_states import LayerPhase

from subnet.utils.s3_torch import download_tensor
from subnet.utils.vector_utils import check_for_nans_and_infs, flatten_optimizer_state
from subnet.utils.partition_utils import save_model_weights_and_optimizer_state, load_model_weights_and_optimizer_state
from subnet.base.base_neuron import BaseNeuron
from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import _clean_gpu_memory, compute_loss
from subnet.test_client import TestAPIClient


class Miner(BaseNeuron, HealthServerMixin):
    def __init__(self, wallet_name: str | None = None, wallet_hotkey: str | None = None, wallet: Wallet | None = None):
        super().__init__()
        self.init_neuron(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)
        self.state_manager: StateManager = StateManager(wallet=self.wallet)
        self.weights_submitted: bool = False
        self.partitions_submitted: bool = False
        self.miner_api_client: MinerAPIClient = MinerAPIClient(hotkey=self.wallet.hotkey)

    async def run(self):
        logger.info(f"🚀 Starting miner {self.hotkey[:8]} | Timeout: {miner_settings.TIMEOUT}s")

        await self.reset_miner_state()

        while True:
            try:
                if not await CommonAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey):
                    logger.info(f"🔄 Orchestrator health check failed for miner {self.wallet.hotkey.ss58_address[:8]}")
                    await self.reset_miner_state()

                    # A small delay before continuing might be beneficial.
                    await asyncio.sleep(5)
                    continue

                # Final memory check after loading
                if torch.cuda.is_available():
                    allocated_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.debug(f"💾 GPU memory: {allocated_memory:.2f}GB")

                if self.miner_api_client.layer_state == LayerPhase.TRAINING:
                    await self.step()
                    self.weights_submitted = False
                    self.partitions_submitted = False

                elif self.miner_api_client.layer_state == LayerPhase.WEIGHTS_UPLOADING:
                    logger.info(
                        f"\n\n\n\n\n\n\n\n 🔄 Miner in layer {self.state_manager.layer} submitting weights state!\n\n\n\n\n\n\n\n"
                    )
                    if self.weights_submitted:
                        logger.debug(f"Weights already submitted for miner {self.hotkey[:8]}, skipping")
                    else:
                        await self.submit_weights()
                        self.weights_submitted = True
                    logger.info("🔄 Miner submitted weights, switching to merging partitions")
                    await wait_for_state(state=LayerPhase.MERGING_PARTITIONS, miner_api_client=self.miner_api_client)

                elif self.miner_api_client.layer_state == LayerPhase.MERGING_PARTITIONS:
                    logger.info(
                        f"\n\n\n\n\n\n\n\n 🔄 Miner in layer {self.state_manager.layer} merging partitions state!\n\n\n\n\n\n\n\n"
                    )
                    if not self.partitions_submitted:
                        logger.info("🔄 Miner getting weight partition info")
                        weight_path_per_layer, partitions = await get_weight_partition_info(
                            layer=self.state_manager.layer, miner_api_client=self.miner_api_client
                        )

                        if not partitions:
                            logger.info("🔄 Miner has no partitions to merge")
                            continue
                        logger.info("🔄 Miner starting merging partitions")
                        partitions = await self.merge_partitions(
                            weight_path_per_layer=weight_path_per_layer,
                            partitions=partitions,
                        )
                        logger.info("🔄 Miner finished merged partitions")

                        self.partitions_submitted = True
                        await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)
                        await self.download_and_set_weights_and_optimizer_state(
                            device=miner_settings.DEVICE,
                        )

                    else:
                        logger.info(f"🔄 Miner {self.hotkey[:8]} already submitted partitions, skipping...")
                        await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)

                logger.info(
                    f"🔄 Miner {self.hotkey[:8]} in Layer {self.state_manager.layer} is in state: {self.miner_api_client.layer_state}"
                )
                await asyncio.sleep(1.1)

            except LayerStateException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} layer state change...: {e}")
                continue
            except MinerNotRegisteredException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} miner not registered error: {e}")
                continue
            except APIException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} API exception: {e}")
                continue
            except aiohttp.ClientResponseError as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Client response error: {e}")
                continue
            except NanInfWarning as e:
                logger.info(f"⚠️ Miner {self.hotkey[:8]} NaN/Inf warning: {e}")
                continue
            except NanInfException as e:
                logger.error(f"❌ Miner {self.hotkey[:8]} NaN/Inf exception: {e}")
                raise
            except Exception:
                raise

    async def step(self):
        logger.info(
            f"🔄 Miner {self.hotkey[:8]} step | Layer: {self.state_manager.layer} \n"
            f"is_training: {self.miner_api_client.layer_state} \n"
            f"backwards_since_reset: {self.state_manager.backwards_since_reset} \n"
            f"len(cache): {len(self.state_manager.cache)}"
        )

        # Check if any of the activations in the cache have timed out and remove them
        if len(self.state_manager.cache) == common_settings.MAX_ACTIVATION_CACHE_SIZE:
            self.state_manager.check_if_timeout(timeout=common_settings.ACTIVATION_CACHE_TIMEOUT)

        response: ActivationResponse | dict = await self.miner_api_client.get_activation()
        if not response:
            raise Exception("Error getting activation")

        if response.direction == "forward":
            await self.forward(response)
        elif response.direction == "backward":
            await self.backward(response)

    async def forward(self, activation: ActivationResponse | None = None):
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
        if await self.state_manager.out_of_cache(miner_api_client=self.miner_api_client):
            logger.warning(
                f"⚠️ Miner {self.hotkey[:8]} is out of cache ({len(self.state_manager.cache)}/{common_settings.MAX_ACTIVATION_CACHE_SIZE}), skipping forward pass until backwards have been performed"
            )
            await asyncio.sleep(1)
            return

        assert (
            activation.presigned_download_url is not None and activation.presigned_upload_url is not None
        ), f"Activation is required for layer {self.state_manager.layer}, activation: {activation}"

        logger.info(
            f"🚀 Starting FORWARD pass for layer {self.state_manager.layer} | Processing activation {activation.activation_id} | Miner: {self.hotkey[:8]}"
        )
        if self.state_manager.layer == 0:
            # Load text file and tokenize
            input_activations = await download_sample(
                download_url=activation.presigned_download_url, tokenizer=self.model_manager.tokenizer
            )
        else:
            # Download activation from S3
            input_activations = await download_tensor(
                path=activation.presigned_download_url, device=miner_settings.DEVICE
            )
            if not common_settings.MOCK:
                input_activations = input_activations.reshape(
                    -1,
                    common_settings.SEQUENCE_LENGTH,
                    self.model_manager.model_config.get("bottleneck_dim") or self.model_manager.model_config["emb_dim"],
                )

        # Perform the actual forward pass

        logger.debug(f"Forwarding activation of size {input_activations.shape}")
        output_activations, state = await self.model_manager._forward(
            layer=self.state_manager.layer, input_activations=input_activations
        )

        self.state_manager.add_to_cache(
            activation.activation_id,
            CacheEntry(
                input_activations=input_activations,
                output_activations=output_activations,
                state=state,
                upload_time=time.time(),
            ),
        )

        if self.state_manager.layer == self.model_manager.model_metadata["n_splits"] - 1:
            await self.compute_last_layer_loss(
                output_activations=output_activations,
                input_activation_response=activation,
                state=state,
                input_activations=input_activations,
            )
            return await self.backward(activation=activation)

        # If we are not on the last layer, we just need to upload the activations
        upload_response: CompleteFileUploadResponse = await upload_tensor(
            tensor=output_activations.detach().clone(),
            hotkey=self.wallet.hotkey,
        )

        await self.miner_api_client.submit_activation_request(
            submit_activation_request=SubmitActivationRequest(
                activation_id=activation.activation_id,
                activation_path=upload_response.object_path,
                direction="forward",
            ),
        )
        logger.info(
            f"✅ Successfully completed FORWARD pass for activation {activation.activation_id} on layer {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

    async def backward(
        self,
        activation: ActivationResponse,
    ):
        logger.info(
            f"🔄 Starting BACKWARD pass for activation {activation.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

        # Check if activation is in cache
        if activation.activation_id not in self.state_manager.cache:
            logger.warning(f"⚠️ Activation {activation.activation_id} not found in cache, skipping backward pass")
            return
        activation_grads = None
        if (
            self.state_manager.layer != self.model_manager.model_metadata["n_splits"] - 1
            and self.model_manager.model_metadata["n_splits"] > 1
        ):
            # For backward pass, we need to get activations that we have cached forward activations for
            # So we still need to list first, then filter, then randomly select
            activation_grads: torch.Tensor = await download_tensor(
                path=activation.presigned_download_url, device=miner_settings.DEVICE
            )
            if not common_settings.MOCK:
                activation_grads = activation_grads.reshape(
                    -1,
                    common_settings.SEQUENCE_LENGTH,
                    self.model_manager.model_config.get("bottleneck_dim")
                    or self.model_manager.model_config.get("emb_dim"),
                )

        # Get activations from cache and move back to GPU
        cached_activations = self.state_manager.cache[activation.activation_id]

        # Move to GPU and enable gradients only for floating point tensors
        input_activations: torch.Tensor = cached_activations.input_activations.to(miner_settings.DEVICE)
        output_activations: torch.Tensor = cached_activations.output_activations.to(miner_settings.DEVICE)

        state = cached_activations.state

        await self.model_manager._backward(
            layer=self.state_manager.layer,
            output_activations=output_activations,
            activation_grads=activation_grads,
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
            grad_size = (
                self.model_manager.model_config["bottleneck_dim"]
                if self.model_manager.model_config["bottleneck_dim"] is not None
                else self.model_manager.model_config["emb_dim"]
            )
            input_activation_grads = emb_weight.grad[: common_settings.SEQUENCE_LENGTH, :grad_size]

            # Detach and convert to bfloat16 to ensure we only save the values
            input_activation_grads = input_activation_grads.detach().to(torch.bfloat16).cpu()

        else:
            input_activation_grads = input_activations.grad

        upload_response: CompleteFileUploadResponse = await upload_tensor(
            tensor=input_activation_grads,
            hotkey=self.wallet.hotkey,
        )

        response = await self.miner_api_client.submit_activation_request(
            submit_activation_request=SubmitActivationRequest(
                activation_id=activation.activation_id,
                activation_path=upload_response.object_path,
                direction="backward",
            ),
        )
        # Remove from cache
        self.state_manager.remove_from_cache(activation.activation_id)
        logger.info(
            f"✅ Successfully completed BACKWARD pass for activation {activation.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

    async def compute_last_layer_loss(
        self,
        output_activations: torch.Tensor,
        input_activation_response: ActivationResponse,
        state: dict,
        input_activations: torch.Tensor,
    ):
        """
        Performs the backward pass for the last layer.
        """

        initial_activations_path = await self.miner_api_client.get_targets(
            get_targets_request=GetTargetsRequest(activation_id=input_activation_response.activation_id),
        )

        # Target sample is the initial activations
        sample = await download_sample(download_url=initial_activations_path, tokenizer=self.model_manager.tokenizer)

        loss: torch.Tensor = compute_loss(
            mock=common_settings.MOCK,
            logits=output_activations,
            targets=sample,
            vocab_size=self.model_manager.vocab_size,
            pad_token_id=self.model_manager.eos_token_id,
            pack=miner_settings.PACK_SAMPLES,
        )

        check_for_nans_and_infs(tensor=loss, name=f"Loss for miner {self.hotkey[:8]}", exception_type=NanInfException)

        logger.info(
            f"📊 Computed loss {loss:.6f} for activation {input_activation_response.activation_id} | Layer: {self.state_manager.layer} | Miner: {self.hotkey[:8]}"
        )

        # Update cache with loss before attempting to report it to handle API errors gracefully
        self.state_manager.add_to_cache(
            input_activation_response.activation_id,
            CacheEntry(
                input_activations=input_activations,
                output_activations=loss,
                state=state,
                upload_time=time.time(),
            ),
        )

        try:
            loss_copy: torch.Tensor = loss.clone().detach()
            response = await self.miner_api_client.report_loss(
                loss_report=LossReportRequest(
                    activation_id=input_activation_response.activation_id, loss=loss_copy.item()
                ),
            )
            if hasattr(response, "error_name"):
                return

        except Exception as e:
            logger.error(f"Error reporting loss: {e}")

    async def register_loop(self) -> tuple[dict, dict]:
        """
        Register the miner with the orchestrator, acquiring a layer during the process.
        If the miner is not registered, it will try to register every 60 seconds
        """
        while True:
            try:
                if not getattr(self, "registered_on_metagraph", True):
                    logger.warning(
                        f"Miner {self.hotkey[:8]} not registered on metagraph. Sleeping for 1 minute before retry..."
                    )
                    await asyncio.sleep(60)  # 60 seconds

                    # Try to re-register using init_neuron method
                    logger.info(f"🔄 Attempting to re-register miner {self.hotkey[:8]} on metagraph...")
                    self.init_neuron(
                        wallet_name=miner_settings.WALLET_NAME,
                        wallet_hotkey=miner_settings.WALLET_HOTKEY,
                        wallet=self.wallet,
                    )
                    if not getattr(self, "registered_on_metagraph", True):
                        continue
                    else:
                        logger.success(f"✅ Miner {self.hotkey[:8]} registered successfully on metagraph")

                if not common_settings.BITTENSOR:
                    await TestAPIClient.register_to_metagraph(hotkey=self.wallet.hotkey)

                logger.info(f"🔄 Attempting to register miner {self.hotkey[:8]} with orchestrator...")
                response: MinerRegistrationResponse = await self.miner_api_client.register_miner_request()

                assigned_layer = int(response.layer)
                current_epoch = int(response.current_epoch)

                if response.layer is None:
                    raise Exception(
                        f"Miner {self.hotkey[:8]} registered with no layer assigned, this should not happen"
                    )

                self.state_manager.layer = assigned_layer
                self.state_manager.training_epoch_when_registered = current_epoch
                self.state_manager.run_id = response.run_id

                logger.success(
                    f"✅ Miner {self.hotkey[:8]} registered successfully in layer {self.state_manager.layer} on training epoch {current_epoch}"
                )
                return response.model_cfg.model_dump(), response.model_metadata.model_dump()

            except Exception as e:
                logger.exception(f"Error registering miner: {e}")
                await asyncio.sleep(10)

    async def submit_weights(self):
        """
        Uploads the weights to the orchestrator and submits them to the database

        Raises:
            SubmittedWeightsError: If the weights are not submitted successfully
            e: If there is an error submitting the weights
        """
        if self.state_manager.backwards_since_reset == 0:
            logger.warning(f"Backwards since reset for miner {self.hotkey[:8]} is 0, skipping")
            return
        if all([p.grad is None for p in self.model_manager.model.parameters()]):
            logger.warning(f"Gradients are None for miner {self.hotkey[:8]}, skipping")
            logger.debug(f"Gradients: {[p.grad for p in self.model_manager.model.parameters()]}")
            return

        learning_rate = await self.miner_api_client.get_learning_rate()
        await self.model_manager.local_all_reduce(learning_rate=learning_rate)

        flattened_optimizer_state, _, _ = flatten_optimizer_state(
            optimizer=self.model_manager.optimizer, device=miner_settings.DEVICE
        )
        weights = torch.nn.utils.parameters_to_vector(parameters=self.model_manager.model.parameters())

        try:
            num_splits = await self.miner_api_client.get_num_splits()
            self.state_manager.num_metadata_chunks = num_splits
            if not self.state_manager.num_metadata_chunks:
                raise Exception("Error getting number of splits")

            weight_update_dict = {}
            for name, tensor in {"weights": weights, "optimizer_state": flattened_optimizer_state}.items():
                check_for_nans_and_infs(
                    tensor=tensor, name=f"{name} for miner {self.hotkey[:8]}", exception_type=NanInfException
                )

                metadata_name = f"{name}_metadata"
                metadata: dict = await create_metadata(
                    weights_tensor=tensor, num_sections=self.state_manager.num_metadata_chunks
                )

                # Convert tensor to bytes, handling bfloat16 compatibility
                tensor_cpu = tensor.detach().to("cpu").contiguous()
                tensor_cpu = tensor_cpu.view(torch.uint8)
                # Convert bfloat16 to float32 for NumPy compatibility, then to bytes
                tensor_bytes = tensor_cpu.numpy().tobytes()
                logger.debug(
                    f"UPLOADING {name} for miner {self.hotkey[:8]}. Elements: {tensor_cpu.numel()}, Dtype: {tensor_cpu.dtype}, Shape: {tensor_cpu.shape}"
                )

                path: str | dict = await upload_file(data=tensor_bytes, file_type=name, hotkey=self.wallet.hotkey)

                # Upload metadata as activation type since orchestrator doesn't have a metadata type
                metadata_path = await upload_file(
                    data=json.dumps(metadata).encode(), file_type=metadata_name, hotkey=self.wallet.hotkey
                )

                weight_update_dict[name + "_path"] = path
                weight_update_dict[metadata_name + "_path"] = metadata_path

            response: dict = await self.miner_api_client.submit_weights(
                weight_update=WeightUpdate(**weight_update_dict)
            )

            if not response:
                raise SubmittedWeightsError("Error submitting weights")

        except LayerStateException as e:
            logger.debug(f"Layer state exception submitting weights: {e}")
            raise

        except Exception as e:
            logger.error(f"Generic error submitting weights: {e}")
            raise

    async def run_miner(self):
        """
        Run the miner. Responsible for:
        - Starting the healthcheck server
        - Registering the miner
        - Setting up the local model
        - Running the miner loop

        The method runs in a loop and retries on failures with a fixed delay.
        """

        with logger.contextualize(hotkey=self.hotkey[:8], layer=self.state_manager.layer):
            logger.info("🚀 Starting miner 🚀")
            try:
                # Start the healthcheck server
                if miner_settings.LAUNCH_HEALTH:
                    await self._start_health_server()
                    logger.info("🏥 Health server started")
                else:
                    logger.warning(
                        "⚠️ Miner healthcheck API not configured in settings (MINER_HEALTH_PORT missing). Skipping."
                    )

                # Reset the entire miner state, which also downloads the weights and optimizer state.
                await self.run()

            except KeyboardInterrupt:
                logger.info("Gracefully shutting down miner")

            except SpecVersionException:
                logger.error("Spec version mismatch. Please pull the latest code and restart the miner")
                raise

            except Exception as e:
                logger.exception(f"❌ Critical error in run_miner: {e}")
                await asyncio.sleep(5)

            finally:
                logger.info(f"Saving current model weights and optimizer state for miner {self.hotkey[:8]} on shutdown")
                save_model_weights_and_optimizer_state(
                    model_weights=torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()),
                    optimizer_state_dict=self.model_manager.optimizer.state_dict(),
                    hotkey=self.hotkey,
                    run_id=self.state_manager.run_id,
                )
                logger.info("Cleaning up miner on shutdown...")
                try:
                    _clean_gpu_memory()

                    try:
                        await self._stop_health_server()
                        logger.info("🏥 Health server stopped")
                    except Exception as e:
                        logger.error(f"Failed to stop health server: {e}")

                except Exception as e:
                    logger.error(f"Failed to shutdown miner: {e}")

            # Final cleanup when exiting the loop (only reached on KeyboardInterrupt)
            logger.info("🛑 Miner shutdown complete")

            # Miners can sometimes not clean themselves up properly. Therefore, lets force kill the process.
            sys.exit(0)

    async def reset_miner_state(self):
        """
        Reset the entire miner state, including the API client, health server, and all other state.
        """
        logger.info("🔄 Resetting miner entire state between epochs!")

        self.state_manager.reset()

        # TODO: This wont work if we start moving miners across layers depending on the epoch.
        if self.model_manager.model is not None and self.model_manager.optimizer is not None:
            current_model_weights: torch.Tensor = torch.nn.utils.parameters_to_vector(
                self.model_manager.model.parameters()
            )
            current_model_optimizer_state: dict = self.model_manager.optimizer.state_dict()

            # Save the current model weights and optimizer state to a file to be loaded in at the beginning of the next epoch.
            # This is to prevent potentially losing the model weights when stuck on critical miner restarts.
            save_model_weights_and_optimizer_state(
                model_weights=current_model_weights,
                optimizer_state_dict=current_model_optimizer_state,
                hotkey=self.hotkey,
                run_id=self.state_manager.run_id,
            )

        current_model_weights, current_model_optimizer_state = load_model_weights_and_optimizer_state(
            hotkey=self.hotkey, run_id=self.state_manager.run_id
        )

        self.model_manager.reset()

        # We provide the model config and metadata so that all miners are aligned.
        model_config, model_metadata = await self.register_loop()

        if not await self._setup_local_model(
            model_config=model_config,
            model_metadata=model_metadata,
            model_weights=current_model_weights,
            optimizer_state=current_model_optimizer_state,
            layer=self.state_manager.layer,
            device=miner_settings.DEVICE,
        ):
            raise Exception("Error setting up local model")

        logger.success("✅ Successfully setup local model")

        try:
            await self.download_and_set_weights_and_optimizer_state(
                device=miner_settings.DEVICE,
            )
        except Exception as e:
            logger.exception(f"Error downloading and setting weights and optimizer state: {e}")
            raise

    async def merge_partitions(
        self, weight_path_per_layer: list[SubmittedWeightsAndOptimizerPresigned], partitions: list[MinerPartition]
    ) -> list[MinerPartition]:
        """Merge the models from the other miners.

        Args:
            weight_path_per_layer (list[SubmittedWeightsPresigned]): The paths to the other miners' partitions
            partition_ids (list[int]): The partition indices to merge

        Returns:
            list[Partition]: The merged partitions
        """
        final_partitions: list[MinerPartition] = []

        filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]] = await filter_bad_metadata(
            partitions=partitions, submitted_weights_and_optimizers=weight_path_per_layer
        )
        number_of_valid_partitions = self.state_manager.num_metadata_chunks

        for partition in partitions:
            if self.state_manager.num_metadata_chunks is not None:
                if partition.chunk_number >= number_of_valid_partitions:
                    logger.warning(
                        f"Skipping partition {partition.chunk_number} because it is invalid as it doesn't exist in the metadata chunks"
                    )
                    continue
            else:
                logger.error(
                    "Somehow, the number of metadata chunks is None, and it shouldn't be...Continuing without the check!"
                )

            try:
                logger.debug(f"Layer {self.state_manager.layer} | merging partition {partition.chunk_number}")

                weight_average = None
                optimizer_state_average = None
                weight_counter = 0
                optimizer_state_counter = 0

                results: list[tuple[torch.Tensor, torch.Tensor]] = await asyncio.gather(
                    *[
                        download_partition(
                            weight_metadata=metadata[partition.chunk_number]["weights"],
                            optimizer_metadata=metadata[partition.chunk_number]["optimizer_state"],
                        )
                        for _, metadata in filtered_metadata.items()
                    ]
                )
                for metadata, (weights, optimizer_state) in zip(filtered_metadata.values(), results):
                    try:
                        if weights is None or optimizer_state is None:
                            logger.warning(f"No weights or optimizer state downloaded. Partitions: {partitions}")
                            raise Exception(f"No weights or optimizer state downloaded. Partitions: {partitions}")

                        # TODO: We will be changing the way that weights and optimizer states are merged.
                        weights_metadata: ChunkMetadata = metadata[partition.chunk_number]["weights"]
                        optimizer_state_metadata: ChunkMetadata = metadata[partition.chunk_number]["optimizer_state"]

                        if weight_average is None:
                            weight_average = weights.to(torch.float32) * weights_metadata.weighting_factor
                            optimizer_state_average = (
                                optimizer_state.to(torch.float32) * optimizer_state_metadata.weighting_factor
                            )

                        else:
                            # create a running sum of weights weighted by the weighting factor
                            weight_average += weights.to(torch.float32) * weights_metadata.weighting_factor
                            optimizer_state_average += (
                                optimizer_state.to(torch.float32) * optimizer_state_metadata.weighting_factor
                            )

                        weight_counter += weights_metadata.weighting_factor
                        optimizer_state_counter += optimizer_state_metadata.weighting_factor

                    except Exception as e:
                        logger.exception(f"Error downloading chunk {partition.chunk_number} from {metadata}: {e}")

                if weight_average is None:
                    raise Exception(f"No weights downloaded. Partitions: {partitions}")

                # Average the weights
                weight_average /= weight_counter
                weight_average = weight_average.to(torch.bfloat16)
                optimizer_state_average /= optimizer_state_counter
                optimizer_state_average = optimizer_state_average.to(torch.bfloat16)

                weight_upload_response: CompleteFileUploadResponse = await upload_tensor(
                    tensor=weight_average.detach().cpu(),
                    file_type="weights",
                    hotkey=self.wallet.hotkey,
                )

                optimizer_state_upload_response: CompleteFileUploadResponse = await upload_tensor(
                    tensor=optimizer_state_average.detach().cpu(),
                    file_type="optimizer_state",
                    hotkey=self.wallet.hotkey,
                )

                partition.weight_path = weight_upload_response.object_path
                partition.optimizer_state_path = optimizer_state_upload_response.object_path
                partition.weight_metadata_path = extract_filename_from_url(weights_metadata.metadata_path)
                partition.optimizer_state_metadata_path = extract_filename_from_url(
                    optimizer_state_metadata.metadata_path
                )

                final_partitions.append(partition)
            except Exception as e:
                logger.exception(f"Failed to get partition {partition.chunk_number}: {e}")

        valid_partitions = []
        for partition in final_partitions:
            if partition.is_valid():
                valid_partitions.append(partition)
            else:
                logger.warning(
                    f"Skipping partition {partition.chunk_number} because it is invalid; partition: {partition}"
                )

        if not valid_partitions:
            raise Exception("No valid partitions to submit")

        await self.miner_api_client.submit_merged_partitions(merged_partitions=valid_partitions)
