import asyncio
from loguru import logger
import torch

from common.models.api_models import LossReportRequest
from subnet.miner_api_client import MinerAPIClient
from common.models.api_models import CompleteFileUploadResponse, SubmitActivationRequest
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from miner.utils.utils import upload_tensor


class ActivationPublisher:
    def __init__(self, miner_api_client: MinerAPIClient):
        self._miner_api_client = miner_api_client
        self._publishing_tasks: list[asyncio.Task] = []

    def publish_activation(self, tensor: torch.Tensor, activation_id: str, direction: str):
        """Publish an activation to the orchestrator."""
        task = asyncio.create_task(
            self._publish_activation(tensor=tensor, activation_id=activation_id, direction=direction)
        )
        self._publishing_tasks.append(task)

    def publish_loss(self, loss: float, activation_id: str):
        """Publish a loss to the orchestrator."""
        task = asyncio.create_task(self._publish_loss(loss=loss, activation_id=activation_id))
        self._publishing_tasks.append(task)

    async def _publish_activation(self, tensor: torch.Tensor, activation_id: str, direction: str):
        """Upload an activation to the orchestrator."""
        try:
            upload_response: CompleteFileUploadResponse = await upload_tensor(
                miner_api_client=self._miner_api_client,
                tensor=tensor,
                hotkey=self._miner_api_client.hotkey,
            )
            logger.debug(f"tensor shape before upload:{tensor.shape}")
            await self._miner_api_client.submit_activation_request(
                submit_activation_request=SubmitActivationRequest(
                    activation_id=activation_id,
                    activation_path=upload_response.object_path,
                    direction=direction,
                ),
            )
            logger.success(f"✅ Successfully published activation {activation_id} direction {direction}")

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            logger.warning(f"Anticipated exception has occurred while publishing activations (swallowed): {e}")
            pass
        except Exception as e:
            logger.exception(f"Failed to publish activation to orchestrator: {e}")
            raise

    async def _publish_loss(self, loss: float, activation_id: str):
        """Report a loss to the orchestrator."""
        try:
            await self._miner_api_client.report_loss(
                loss_report=LossReportRequest(activation_id=activation_id, loss=loss),
            )
            logger.success(f"✅ Successfully published loss for activation {activation_id}")

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            logger.warning(f"Anticipated exception has occurred while publishing loss (swallowed): {e}")
            pass
        except Exception as e:
            logger.exception(f"Failed to publish loss to orchestrator: {e}")
            raise

    async def reset(self):
        """Cancel any in-progress publishing tasks."""
        if len(self._publishing_tasks) > 0:
            for task in self._publishing_tasks:
                if not task.done():
                    task.cancel()

            results = await asyncio.gather(*self._publishing_tasks, return_exceptions=True)
            for result in results:
                try:
                    if isinstance(result, Exception):
                        raise result
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Failed to publish message to orchestrator: {e}")
                    pass
            self._publishing_tasks.clear()
