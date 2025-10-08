import asyncio
import copy
from loguru import logger
import json
import sys
from miner.utils.partition_merging import download_previous_optimizer_state_for_partition_batch, merge_partition_batch
from miner.utils.partition_merging import get_partition_batch
from miner.utils.partition_merging import download_pseudograds_for_partition_batch
from miner.utils.partition_merging import upload_partition_batch
from subnet.utils.partition_utils import save_model_weights_and_optimizer_state
import torch
import aiohttp
from bittensor import Wallet
from subnet.common_api_client import CommonAPIClient
from miner.health_server import HealthServerMixin
from miner.utils.partition_merging import (
    filter_bad_metadata,
    get_weight_partition_info,
)
from miner import settings as miner_settings
from miner.state_manager import StateManager
from miner.utils.utils import (
    create_metadata,
    upload_file,
    upload_tensor,
    wait_for_state,
)
from miner.utils.run_utils import identify_best_run
from common import settings as common_settings
from common.models.api_models import (
    MinerRegistrationResponse,
    RegisterMinerRequest,
    SubmittedWeightsAndOptimizerPresigned,
    WeightUpdate,
)
from common.models.miner_models import ChunkMetadata
from common.utils.exceptions import (
    APIException,
    LayerStateException,
    RateLimitException,
    MinerNotRegisteredException,
    NanInfException,
    NanInfWarning,
    SpecVersionException,
    SubmittedWeightsError,
    WeightPartitionException,
)
from common.utils.partitions import MinerPartition
from common.utils.shared_states import LayerPhase
from common.models.run_flags import RUN_FLAGS  # noqa: F401
from subnet.base.base_neuron import BaseNeuron
from subnet.miner_api_client import MinerAPIClient
from subnet.model.utils import _clean_gpu_memory, log_gpu_memory_usage
from subnet.test_client import TestAPIClient
from subnet.utils.partition_utils import (
    MergingPartition,
    load_model_weights,
    load_model_weights_and_optimizer_state,
)
from subnet.utils.vector_utils import check_for_nans_and_infs
from miner.training import TrainingPhase
from subnet.model import gpu_device


class Miner(BaseNeuron, HealthServerMixin):
    def __init__(self, wallet_name: str | None = None, wallet_hotkey: str | None = None, wallet: Wallet | None = None):
        super().__init__()
        self.init_neuron(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, wallet=wallet)
        self.state_manager: StateManager = StateManager(wallet=self.wallet)
        self.weights_submitted: bool = False
        self.partitions_submitted: bool = False
        self.miner_api_client: MinerAPIClient = MinerAPIClient(hotkey=self.wallet.hotkey)
        self.need_to_pull_weights = True
        self.training_phase: TrainingPhase = TrainingPhase(
            miner_api_client=self.miner_api_client,
            state_manager=self.state_manager,
            model_manager=self.model_manager,
        )

    async def run(self):
        await self.reset_miner_state()
        logger.info(f"🚀 Starting miner {self.hotkey[:8]} on layer {self.layer} | Timeout: {miner_settings.TIMEOUT}s")

        # You will only enter the while loop if we are in the training state.
        await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client, raise_bad_sync=False)

        while True:
            try:
                with logger.contextualize(
                    hotkey=self.hotkey[:8],
                    run_id=self.state_manager.run_id,
                    layer=self.state_manager.layer,
                ):
                    if not await CommonAPIClient.check_orchestrator_health(hotkey=self.wallet.hotkey):
                        logger.info(
                            f"🔄 Orchestrator health check failed for miner {self.wallet.hotkey.ss58_address[:8]}"
                        )
                        # A small delay before continuing might be beneficial.
                        await asyncio.sleep(5)
                        continue

                    # Final memory check after loading
                    allocated_memory = gpu_device.allocated_memory() / 1024**3  # GB
                    logger.debug(f"💾 GPU memory: {allocated_memory:.2f}GB")

                    logger.info(
                        f"🔄 Miner {self.hotkey[:8]} in Layer {self.state_manager.layer} is in state: {self.miner_api_client.layer_state}"
                    )

                    if self.miner_api_client.layer_state == LayerPhase.TRAINING:
                        if self.need_to_pull_weights:
                            try:
                                await self.download_and_set_global_weights(
                                    device=miner_settings.DEVICE,
                                    client=self.miner_api_client,
                                    download_local_optimizer_state=False,
                                )
                            except Exception as e:
                                logger.exception(f"Error downloading and setting weights: {e}")
                            finally:
                                # Always persist a snapshot at epoch start so submit_weights has previous weights
                                save_model_weights_and_optimizer_state(
                                    model_weights=torch.nn.utils.parameters_to_vector(
                                        self.model_manager.model.parameters()
                                    ),
                                    optimizer_state_dict=self.model_manager.optimizer.state_dict(),
                                    hotkey=self.hotkey,
                                    run_id=self.state_manager.run_id,
                                    layer_idx=self.state_manager.layer,
                                )
                                logger.info(
                                    f"Saved current model weights and optimizer state for miner {self.hotkey[:8]}"
                                )

                        # Need to ensure that we don't pull weights again in this loop
                        self.need_to_pull_weights = False
                        self.weights_submitted = False
                        self.partitions_submitted = False
                        await self.training_phase.run()

                    if self.miner_api_client.layer_state == LayerPhase.WEIGHTS_UPLOADING:
                        self.need_to_pull_weights = True
                        logger.info(
                            f"\n\n\n\n\n\n\n\n 🔄 Miner in layer {self.state_manager.layer} submitting weights state!\n\n\n\n\n\n\n\n"
                        )
                        if self.weights_submitted:
                            logger.debug(f"Weights already submitted for miner {self.hotkey[:8]}, skipping")
                        else:
                            await self.submit_weights()
                            self.weights_submitted = True
                        logger.info("🔄 Miner submitted weights, switching to merging partitions")
                        await wait_for_state(
                            state=LayerPhase.MERGING_PARTITIONS, miner_api_client=self.miner_api_client
                        )
                        continue

                    if self.miner_api_client.layer_state == LayerPhase.MERGING_PARTITIONS:
                        self.need_to_pull_weights = True
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

                            logger.info(f"🔄 Miner starting merging partitions: {[p.chunk_number for p in partitions]}")
                            await self.merge_partitions(
                                weight_path_per_layer=weight_path_per_layer,
                                partitions=partitions,
                            )
                            logger.info("🔄 Miner finished merged partitions")

                            self.partitions_submitted = True
                            await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)

                        else:
                            logger.info(f"🔄 Miner {self.hotkey[:8]} already submitted partitions, skipping...")
                            await wait_for_state(state=LayerPhase.TRAINING, miner_api_client=self.miner_api_client)

                        # Epoch counter for local optimization step
                        self.model_manager.epoch_counter += 1
                        continue

                await asyncio.sleep(1.1)

            except LayerStateException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} layer state change...: {e}")
                continue
            except MinerNotRegisteredException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} miner not registered error: {e}")
                await self.reset_miner_state()
                continue
            except APIException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} API exception: {e}")
                continue
            except RateLimitException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Rate limit exception: {e}")
                continue
            except aiohttp.ClientResponseError as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Client response error: {e}")
                continue
            except (asyncio.TimeoutError, TimeoutError) as e:
                logger.warning(f"🔄 Miner {self.hotkey[:8]} Timeout error: {e}")
                continue
            except SubmittedWeightsError as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Submitted weights error: {e}")
                continue
            except WeightPartitionException as e:
                logger.info(f"🔄 Miner {self.hotkey[:8]} Partition exception: {e}")
                continue
            except NanInfWarning as e:
                logger.info(f"⚠️ Miner {self.hotkey[:8]} NaN/Inf warning: {e}")
                continue
            except NanInfException as e:
                logger.error(f"❌ Miner {self.hotkey[:8]} NaN/Inf exception: {e}")
                raise
            except Exception:
                raise

    async def register_loop(self) -> tuple[dict, dict]:
        """
        Register the miner with the orchestrator, acquiring a layer during the process.
        If the miner is not registered, it will try to register every 60 seconds
        """
        while True:
            try:
                if not common_settings.BITTENSOR:
                    await TestAPIClient.register_to_metagraph(hotkey=self.wallet.hotkey)

                logger.info(f"🔄 Attempting to fetch run info for miner {self.hotkey[:8]}...")
                run_info_list = await self.miner_api_client.fetch_run_info_request()
                if not run_info_list:
                    raise Exception("Fatal Error: Could not fetch run info")

                best_run = identify_best_run(run_info_list=run_info_list)
                logger.info(f"✅ Best run for miner {self.hotkey[:8]} is {best_run.run_id}")

                logger.info(
                    f"🔄 Attempting to register miner {self.hotkey[:8]} on run {best_run.run_id} with orchestrator..."
                )
                response: MinerRegistrationResponse = await self.miner_api_client.register_miner_request(
                    register_miner_request=RegisterMinerRequest(run_id=best_run.run_id)
                )

                assigned_layer = int(response.layer)
                current_epoch = int(response.current_epoch)

                if response.layer is None:
                    raise Exception(
                        f"Miner {self.hotkey[:8]} registered with no layer assigned, this should not happen"
                    )

                # TODO: clean these up
                self.layer = assigned_layer
                self.state_manager.layer = assigned_layer
                self.state_manager.training_epoch_when_registered = current_epoch
                self.state_manager.run_id = response.run_id
                self.run_id = response.run_id
                self.model_manager.epoch_on_registration = current_epoch

                global RUN_FLAGS
                RUN_FLAGS = response.run_flags

                logger.success(
                    f"✅ Miner {self.hotkey[:8]} registered successfully in layer {self.state_manager.layer} on training epoch {current_epoch}"
                )
                logger.debug(f"Run flags for miner {self.hotkey[:8]}: {RUN_FLAGS}")
                return response.model_cfg.model_dump(), response.model_metadata.model_dump()

            except SpecVersionException as e:
                logger.error(f"Spec version mismatch: {e}")
                raise

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
        if self.training_phase.backwards_since_reset == 0:
            logger.warning(f"Backwards since reset for miner {self.hotkey[:8]} is 0, skipping")
            return

        current_weights = (
            torch.nn.utils.parameters_to_vector(parameters=self.model_manager.model.parameters()).detach().to("cpu")
        )
        previous_weights = load_model_weights(
            hotkey=self.hotkey, run_id=self.state_manager.run_id, layer_idx=self.state_manager.layer
        )

        # For diloco we want to upload the pseudo gradients to the orchestrator
        if previous_weights is None:
            raise Exception(f"Previous weights are None for miner {self.hotkey[:8]}")

        pseudo_gradients = previous_weights.to(torch.float32) - current_weights.to(torch.float32)
        pseudo_gradients = await self.model_manager.clip_pseudo_gradients(pseudo_gradients)
        pseudo_gradients = pseudo_gradients.to(torch.bfloat16)

        # Log some stats about the pseudo gradients
        logger.info(
            f"Pseudo gradients for miner {self.hotkey[:8]} have mean {pseudo_gradients.mean():.6f} and std {pseudo_gradients.std():.6f}"
        )
        logger.info(
            f"Previous weights for miner {self.hotkey[:8]} have mean {previous_weights.mean():.6f} and std {previous_weights.std():.6f}"
        )
        logger.info(
            f"New weights for miner {self.hotkey[:8]} have mean {current_weights.mean():.6f} and std {current_weights.std():.6f}"
        )
        logger.info(f"Pseudo gradients shape: {pseudo_gradients.shape}")

        try:
            self.model_manager.optimizer.zero_grad()
            await self.training_phase.optimization_reset()

            try:
                await self.miner_api_client.notify_orchestrator_of_state_call()
            except Exception as e:
                logger.warning(f"Error notifying orchestrator of state call: {e}")

            check_for_nans_and_infs(
                tensor=pseudo_gradients,
                name=f"pseudo gradients for miner {self.hotkey[:8]}",
                exception_type=NanInfException,
            )

            metadata: dict = await create_metadata(tensor=pseudo_gradients, num_sections=self.num_partitions)

            # Convert tensor to bytes, handling bfloat16 compatibility
            path = await upload_tensor(
                tensor=pseudo_gradients,
                file_type="weights",
                hotkey=self.wallet.hotkey,
                miner_api_client=self.miner_api_client,
            )

            # Upload metadata as activation type since orchestrator doesn't have a metadata type
            metadata_path = await upload_file(
                miner_api_client=self.miner_api_client,
                data=json.dumps(metadata).encode(),
                file_type="weights_metadata",
                hotkey=self.wallet.hotkey,
            )

            response: dict = await self.miner_api_client.submit_weights(
                weight_update=WeightUpdate(weights_path=path.object_path, weights_metadata_path=metadata_path),
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

        except LayerStateException as e:
            logger.warning(f"Layer state exception: {e}")

        except Exception as e:
            logger.exception(f"❌ Critical error in run_miner: {e}")
            await asyncio.sleep(5)

        finally:
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
        logger.info("🔄 Resetting miner entire state!")
        self.need_to_pull_weights = True

        old_run_id = self.state_manager.run_id
        old_layer = self.state_manager.layer
        await self.training_phase.reset()
        # We provide the model config and metadata so that all miners are aligned.
        model_config, model_metadata = await self.register_loop()

        # if we continue on the same run and layer, save off what we've done so far and load weights
        current_model_weights: torch.Tensor = None
        current_model_optimizer_state: dict = None

        if old_run_id == self.state_manager.run_id and old_layer == self.state_manager.layer:
            if self.model_manager.model is not None and self.model_manager.optimizer is not None:
                current_model_weights = torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters())
                current_model_optimizer_state = self.model_manager.optimizer.state_dict()

            else:
                current_model_weights, current_model_optimizer_state = load_model_weights_and_optimizer_state(
                    hotkey=self.hotkey,
                    run_id=self.state_manager.run_id,
                    layer_idx=self.state_manager.layer,
                )

        self.model_manager.reset()

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

    async def get_old_partition_for_partition_batch(
        self, batch_partitions: list[MergingPartition]
    ) -> list[MergingPartition]:
        previous_partitions = await self.miner_api_client.get_previous_partitions(
            partition_indices=[partition.new_partition.chunk_number for partition in batch_partitions]
        )
        for partition in batch_partitions:
            previous_partition = [
                p for p in previous_partitions if p.chunk_number == partition.new_partition.chunk_number
            ]
            if not previous_partition:
                logger.warning(f"No previous partition found for partition {partition.new_partition.chunk_number}")
                partition.old_partition = None
            else:
                partition.old_partition = previous_partition[0]
        logger.debug(f"{len(batch_partitions)} batch partitions got old partition")
        return batch_partitions

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
        filtered_metadata: dict[str, dict[int, dict[str, ChunkMetadata]]] = await filter_bad_metadata(
            partitions=partitions, submitted_weights_and_optimizers=weight_path_per_layer
        )
        # Grab a batch of partitions to download the weights for
        for batch in range(min(miner_settings.N_PARTITION_BATCHES, len(partitions))):
            logger.debug(f"Merging batch {batch} of {min(miner_settings.N_PARTITION_BATCHES, len(partitions))}")

            # Grab a batch of partitions to merge (no downloading yet)
            batch_partitions: list[MergingPartition] = await get_partition_batch(
                batch_index=batch, partitions=partitions
            )
            logger.debug(f"{len(batch_partitions)} batch partitions grabbed")

            # Download the weights for the batch (fills partitions.weights with a list of all pseudograds from all the other miners)
            merging_partitions: list[MergingPartition] = await download_pseudograds_for_partition_batch(
                batch_partitions=batch_partitions, filtered_metadata=filtered_metadata
            )
            logger.debug(f"{len(merging_partitions)} batch partitions downloaded successfully")

            # Gets the old partition for the batch (which point us to the previous optimizer state)
            merging_partitions = await self.get_old_partition_for_partition_batch(merging_partitions)
            logger.debug(f"{len(merging_partitions)} batch partitions got old partition")

            # Download the previous optimizer state for the batch (fills partitions.old_optimizer_state with the previous optimizer state)
            merging_partitions = await download_previous_optimizer_state_for_partition_batch(merging_partitions)
            logger.debug(f"{len(merging_partitions)} batch partitions downloaded previous optimizer state")

            # Determine if we have enough memory in the GPU to merge the partitions on GPU or CPU
            device = miner_settings.DEVICE
            if device != "cpu":
                gpu_device.synchronize()
                gpu_device.empty_cache()
                avail_memory = gpu_device.available_memory()
                # TODO: @cassova: correct this calculation - 100x is just to push it to cpu for now
                need_to_merge_on_gpu = (
                    100
                    * torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()).numel()
                    * len(merging_partitions)
                )
                if need_to_merge_on_gpu > avail_memory:
                    logger.warning(
                        "Not enough memory available to merge partitions on GPU"
                        f" - needed {need_to_merge_on_gpu / 1024**3:.2f}GB, available {avail_memory / 1024**3:.2f}GB"
                    )
                    device = "cpu"

            # Load old weights into model
            if device == "cpu":
                old_model = copy.deepcopy(self.model_manager.model).cpu()
            else:
                old_model = copy.deepcopy(self.model_manager.model)
                log_gpu_memory_usage(note="after copying old model")
            torch.nn.utils.vector_to_parameters(
                load_model_weights(
                    hotkey=self.hotkey, run_id=self.state_manager.run_id, layer_idx=self.state_manager.layer
                ),
                old_model.parameters(),
            )

            # Do the actual merging (apply the optimizer state to the weights)
            merged_partitions = await merge_partition_batch(
                partition_batch=merging_partitions,
                filtered_metadata=filtered_metadata,
                old_model=old_model,
                local_optimizer_state=self.model_manager.optimizer,
                weights_length=torch.nn.utils.parameters_to_vector(old_model.parameters()).numel(),
                num_partitions=self.num_partitions,
                device=device,
            )
            logger.debug(f"{len(merged_partitions)} batch partitions merged")
            log_gpu_memory_usage(note=f"after merging partitions on {device}")

            # Upload the merged partitions to the database and return list of MinerPartition
            final_partitions = await upload_partition_batch(
                merged_partitions=merged_partitions,
                hotkey=self.wallet.hotkey,
                miner_api_client=self.miner_api_client,
            )
            logger.debug(f"{len(final_partitions)} batch partitions uploaded")

            # Submit the merged partitions to the database
            await self.miner_api_client.submit_merged_partitions(merged_partitions=final_partitions)
            logger.debug(f"{len(final_partitions)} batch partitions submitted")

            self.model_manager.model = self.model_manager.model.to(miner_settings.DEVICE)

            del old_model
            del merged_partitions  # TODO: @cassova: do a better job of cleaning this up
            del final_partitions  # TODO: @cassova: do a better job of cleaning this up
            log_gpu_memory_usage(note="after merging partitions")
