from orchestrator.api.test_router import _get_download_link
import asyncio
from common.models.api_models import WeightUpdate
from subnet.miner_api_client import MinerAPIClient
from common.utils.shared_states import LayerPhase
from miner.new_miner import Miner
from subnet.test_client import TestAPIClient
from bittensor_wallet.mock import get_mock_wallet
from loguru import logger
from common.utils.s3_utils import download_file

wallet = get_mock_wallet()
miner = Miner(wallet_name="test_miner", wallet_hotkey=wallet.hotkey)


async def test_register_miner_with_metagraph():
    await TestAPIClient.register_to_metagraph(hotkey=wallet.hotkey)


async def test_set_layer_state():
    await TestAPIClient.set_layer_state(layer=0, new_state=LayerPhase.TRAINING)


async def test_set_miner_layer():
    await TestAPIClient.set_miner_layer(layer=0, hotkey=wallet.hotkey)


async def test_register_miner():
    response = await MinerAPIClient.register_miner_request(hotkey=wallet.hotkey)


async def test_get_activation():
    response = await MinerAPIClient.get_activation(hotkey=wallet.hotkey)
    logger.info(f"\n\nACTIVATIONS: \n\n{response.json()}")


async def list_tables():
    response = await TestAPIClient.get_miners()
    logger.info(f"\n\nMINERS: \n\n{response.json()}")
    response = await TestAPIClient.get_activations()
    logger.info(f"\n\nACTIVATIONS: \n\n{response.json()}")


async def test_submit_weights():
    upload_path = await miner.upload_file(data=b"t" * 10, file_type="weights")
    response = await MinerAPIClient.submit_weights_request(
        hotkey=wallet.hotkey,
        weight_update=WeightUpdate(
            weight_path=upload_path,
            weight_metadata_path=upload_path,
            optimizer_state_metadata_path=upload_path,
            optimizer_state_path=upload_path,
        ),
    )
    logger.info(f"\n\nWEIGHTS: \n\n{response.json()}")


async def run_tests():
    file_link = "steff-migration-bucket/009d5a9e10ff64dafbdc5db0a93186149f92bbaee184dd5c0f23ceb79e5798e1.txt"
    link = await _get_download_link(file_link)
    logger.info(f"\n\nLINK: \n\n{link}")
    data = await download_file(link)
    logger.info(f"\n\nDATA: \n\n{data}")
    # await TestAPIClient.register_miner_to_metagraph(hotkey=wallet.hotkey)
    # logger.success("Registered miner to metagraph")
    # await MinerAPIClient.register_miner_request(hotkey=wallet.hotkey)
    # logger.success("Registered miner")
    # await TestAPIClient.set_layer_state(layer=0, new_state=LayerPhase.TRAINING)
    # logger.success("Set layer state to training")
    # await TestAPIClient.set_miner_layer(layer=0, hotkey=wallet.hotkey)
    # logger.success("Set miner layer")
    # await TestAPIClient.set_layer_state(layer=0, new_state=LayerPhase.WEIGHTS_UPLOADING)
    # logger.success("Set layer state to weights uploading")
    # await test_submit_weights()

    # # List final state of tables
    # await list_tables()

    # await MinerAPIClient.get_activation(hotkey=wallet.hotkey)


if __name__ == "__main__":
    asyncio.run(run_tests())
