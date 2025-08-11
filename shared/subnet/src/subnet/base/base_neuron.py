import json
from datetime import datetime
from hashlib import sha256
from typing import Callable, Optional

import bittensor as bt
import torch
from bittensor_wallet import Keypair
from bittensor_wallet.mock import get_mock_wallet
from common import settings as common_settings
from common.models.miner_models import MinerStatus
from common.utils.partitions import MinerPartition, format_chunk_data
from common.utils.s3_utils import download_file
from common.utils.shared_states import LayerPhase
from loguru import logger
from subnet.common_api_client import CommonAPIClient
from subnet.model.model_mixin import ModelManager
from subnet.utils.bt_utils import get_wallet
from subnet.utils.s3_torch import download_activation
from subnet.utils.vector_utils import (
    add_artificial_gradients,
    check_for_nans_and_infs,
    flatten_optimizer_state,
    reconstruct_optimizer_state,
)


class BaseNeuron:
    def __init__(self):
        super().__init__()
        self.wallet: bt.wallet | None = None
        self.layer: Optional[int] = None
        self.status: str = MinerStatus.IDLE.value
        self.registration_time: str = datetime.now().isoformat()
        self.merge_status: str = LayerPhase.TRAINING.value

        self.model_manager = ModelManager()

    def init_neuron(
        self, wallet_name: str = None, wallet_hotkey: str = None, mock: bool = False, wallet: bt.wallet | None = None
    ):
        if common_settings.BITTENSOR:
            if not wallet:
                self.create_wallet(wallet_name=wallet_name, wallet_hotkey=wallet_hotkey, mock=mock)
        else:
            coldkey = Keypair.create_from_seed(seed=sha256(wallet_hotkey.encode()).hexdigest())
            hotkey = Keypair.create_from_seed(seed=sha256(wallet_name.encode()).hexdigest())
            self.wallet = get_mock_wallet(hotkey=hotkey, coldkey=coldkey)

        self.hotkey = self.wallet.hotkey.ss58_address
        logger.info(f"Launching with hotkey: {self.hotkey}")

        return self

    async def _setup_local_model(self, layer: int, device: str) -> bool:
        try:
            logger.info(f"📥 Attempting to load model for layer {layer} for miner {self.hotkey}")
            await self.model_manager.initialize_model_manager(
                layer=layer,
                device=device,
                logger_attributes={
                    "hotkey": self.hotkey,
                },
            )

        except Exception as e:
            logger.error(f"❌ Error loading model for miner {self.hotkey[:8]}: {e}")
            return False
        return True

    def create_wallet(self, wallet_name: str, wallet_hotkey: str, mock: bool):
        self.wallet = get_wallet(
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            mock=mock,
        )
        return self

    async def build_chunk_data(self, partition: MinerPartition):
        weight_metadata_bytes: bytes = await download_file(presigned_url=partition.weight_metadata_path)
        weight_metadata: dict = json.loads(weight_metadata_bytes)
        optimizer_state_metadata_bytes: bytes = await download_file(
            presigned_url=partition.optimizer_state_metadata_path
        )
        optimizer_state_metadata: dict = json.loads(optimizer_state_metadata_bytes)
        partition.weight_data = await format_chunk_data(weight_metadata, partition.chunk_number)
        partition.optimizer_state_data = await format_chunk_data(optimizer_state_metadata, partition.chunk_number)
        return partition

    async def download_and_set_weights_and_optimizer_state(
        self, layer_idx: int, device: str, parser: Callable = None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Downloads the weights for a given layer and device.

        Miners/validators have to download the shards of the final merged layer weights. These weights exist across multiple files and are 1D byte strings

        Args:
            layer_idx (int): The layer index to download weights for
            device (str): The device to download weights for
            parser (Callable, optional): A parser function to parse the response from the API. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor] | None: The weights and optimizer state for the layer
        """
        try:
            response: list[MinerPartition] = await CommonAPIClient.get_merged_partitions(hotkey=self.wallet.hotkey)

            merged_partitions: list[MinerPartition] = await parser(response) if parser else response

            if not merged_partitions:
                logger.warning("No merged partitions found. Initializing with random weights. 🎲")
                return

            # Allocate memory to the full 1d tensor
            new_weights = torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters())
            check_for_nans_and_infs(new_weights, f"current weights for miner {self.hotkey[:8]}")

            # Set random gradients
            add_artificial_gradients(model=self.model_manager.model, device=device)

            # Take a step to populate internal state
            self.model_manager.optimizer.step()
            self.model_manager.optimizer.zero_grad()
            flat_tensor, tensor_shapes, state_dict = flatten_optimizer_state(
                optimizer=self.model_manager.optimizer, device=device
            )
            check_for_nans_and_infs(flat_tensor, f"current optimizer state for miner {self.hotkey[:8]}")
            # Convert to numpy array
            new_optimizer_state = flat_tensor  # .to(torch.float16).detach().cpu().numpy(force=True)

            for partition in merged_partitions:
                partition: MinerPartition

                await self.build_chunk_data(partition)
                try:
                    weight_shard = download_activation(
                        path=partition.weight_path,
                        device=device,
                    )

                    shard_optimizer_state = download_activation(
                        path=partition.optimizer_state_path,
                        device=device,
                    )

                    check_for_nans_and_infs(weight_shard, f"weight shard downloaded for miner {self.hotkey[:8]}")
                    check_for_nans_and_infs(
                        shard_optimizer_state, f"shard optimizer state downloaded for miner {self.hotkey[:8]}"
                    )

                    new_weights[
                        partition.weight_data.chunk_start_idx : partition.weight_data.chunk_end_idx
                    ] = weight_shard

                    new_optimizer_state[
                        partition.optimizer_state_data.chunk_start_idx : partition.optimizer_state_data.chunk_end_idx
                    ] = shard_optimizer_state

                except Exception as e:
                    logger.exception(f"Error downloading partition {partition}: {e}")

            # Check to make sure we didn't download any nans
            check_for_nans_and_infs(new_weights, f"weights downloaded for miner {self.hotkey[:8]}")
            check_for_nans_and_infs(new_optimizer_state, f"optimizer downloaded for miner {self.hotkey[:8]}")

            # assign weights to self.model
            # reshape thecomplete 1D tensor into the appropriate shape
            new_optimizer_state_dict = reconstruct_optimizer_state(
                flat_tensor=new_optimizer_state, tensor_shapes=tensor_shapes, state_dict=state_dict
            )

            # Set the optimizer state.
            self.model_manager.optimizer.load_state_dict(new_optimizer_state_dict)

            # Set's the weights of the model.
            torch.nn.utils.vector_to_parameters(new_weights, self.model_manager.model.parameters())

            logger.success(f"Successfully applied weights to model for layer {layer_idx}")

        except Exception as e:
            logger.exception(f"Error downloading weights: {e}")
            raise
