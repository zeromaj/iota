from typing import Literal
from common.utils.shared_states import LayerPhase
import requests
from common import settings as common_settings
from bittensor.utils import Keypair


class TestAPIClient:
    @classmethod
    async def register_to_metagraph(cls, hotkey: Keypair, role: Literal["miner", "validator"] = "miner"):
        response = requests.post(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/add_entity_to_metagraph",
            params={"hotkey": hotkey.ss58_address, "role": role},
        )
        if response.status_code != 200:
            raise Exception(f"Error registering miner to metagraph: {response.text}")
        return response

    @classmethod
    async def set_layer_state(cls, layer: int, new_state: LayerPhase):
        response = requests.post(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/set_layer_state",
            params={"layer": layer, "new_state": new_state},
        )
        if response.status_code != 200:
            raise Exception(f"Error setting layer state: {response.text}")
        return response

    @classmethod
    async def create_sample(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/create_sample",
        )
        if response.status_code != 200:
            raise Exception(f"Error creating sample: {response.text}")
        return response

    @classmethod
    async def set_miner_layer(cls, layer: int, hotkey: Keypair):
        response = requests.post(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/set_miner_layer",
            params={"layer": layer, "hotkey": hotkey.ss58_address},
        )
        if response.status_code != 200:
            raise Exception(f"Error setting miner layer: {response.text}")
        return response

    # ================================ STATE CHECKING ENDPOINTS ================================

    @classmethod
    async def get_miners(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/get_miners",
        )
        if response.status_code != 200:
            raise Exception(f"Error getting miners: {response.text}")
        return response

    @classmethod
    async def get_activations(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/get_activations",
        )
        if response.status_code != 200:
            raise Exception(f"Error getting activations: {response.text}")
        return response

    @classmethod
    async def get_activation_history(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/get_activation_history",
        )
        if response.status_code != 200:
            raise Exception(f"Error getting activation history: {response.text}")
        return response

    @classmethod
    async def get_layer_state(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/get_layer_state",
        )
        if response.status_code != 200:
            raise Exception(f"Error getting layer state: {response.text}")
        return response

    @classmethod
    async def get_activation_assignment(cls):
        response = requests.get(
            url=f"{common_settings.ORCHESTRATOR_SCHEMA}://{common_settings.ORCHESTRATOR_HOST}:{common_settings.ORCHESTRATOR_PORT}/test_endpoints/get_activation_assignment",
        )
        if response.status_code != 200:
            raise Exception(f"Error getting activation assignment: {response.text}")
        return response
