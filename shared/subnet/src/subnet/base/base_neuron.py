from datetime import datetime
from typing import Optional

import bittensor as bt
from subnet.utils.partition_utils import download_partitions
import torch
from common.models.miner_models import MinerStatus
from common.utils.partitions import MinerPartition
from common.utils.shared_states import LayerPhase
from loguru import logger
from subnet.common_api_client import CommonAPIClient
from subnet.model.model_mixin import ModelManager
from subnet.utils.bt_utils import get_wallet
from subnet.utils.vector_utils import (
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

    def init_neuron(self, wallet_name: str = None, wallet_hotkey: str = None, wallet: bt.wallet | None = None):
        self.wallet = wallet or get_wallet(
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
        )
        self.hotkey = self.wallet.hotkey.ss58_address
        return self

    async def _setup_local_model(
        self,
        model_config: dict,
        model_metadata: dict,
        model_weights: torch.Tensor,
        optimizer_state: dict,
        layer: int,
        device: str,
    ) -> bool:
        try:
            logger.info(f"📥 Attempting to load model for layer {layer} for {self.hotkey[:8]}")
            await self.model_manager.initialize_model_manager(
                model_config=model_config,
                model_metadata=model_metadata,
                model_weights=model_weights,
                optimizer_state=optimizer_state,
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

    async def download_and_set_weights_and_optimizer_state(
        self, device: str
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        # Get copy of current weights and optimizer state
        old_weights = torch.nn.utils.parameters_to_vector(self.model_manager.model.parameters()).clone()
        old_optimizer_state, tensor_shapes, state_dict = flatten_optimizer_state(
            optimizer=self.model_manager.optimizer, device=device
        )

        # Get merged partitions from db
        merged_partitions: list[MinerPartition] = await CommonAPIClient.get_merged_partitions(hotkey=self.wallet.hotkey)

        # Download new weights and optimizer state
        new_weights, new_optimizer_state = await download_partitions(
            merged_partitions=merged_partitions,
            weights=old_weights,
            optimizer_state=old_optimizer_state,
            device=device,
        )
        if new_weights is None or new_optimizer_state is None:
            logger.warning("No new weights or optimizer state downloaded")
            return

        # Set new weights and optimizer state to model
        new_optimizer_state_dict: dict = reconstruct_optimizer_state(
            flat_tensor=new_optimizer_state, tensor_shapes=tensor_shapes, state_dict=state_dict
        )
        await self.model_manager.set_model_weights_and_optimizer_state(
            model_weights=new_weights, optimizer_state=new_optimizer_state_dict
        )
