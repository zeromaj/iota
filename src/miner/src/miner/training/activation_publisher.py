import asyncio
from loguru import logger
from common.utils.timer_logger import TimerLogger
import torch

from common.models.api_models import (
    AttestationChallengeResponse,
    CompleteFileUploadResponse,
    LossReportRequest,
    MinerAttestationPayload,
    SubmitActivationRequest,
)
from common.models.run_flags import RUN_FLAGS
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException
from miner.utils.attestation_utils import AttestationUnavailableError, collect_attestation_payload
from miner.utils.utils import upload_tensor
from subnet.miner_api_client import MinerAPIClient


class ActivationPublisher:
    def __init__(self, miner_api_client: MinerAPIClient):
        self._miner_api_client = miner_api_client
        self._publishing_tasks: list[asyncio.Task] = []

    def publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        attestation_challenge_blob: str | None,
        upload_url: list[str] | None,
        activation_path: str | None,
    ):
        """Publish an activation to the orchestrator."""
        task = asyncio.create_task(
            self._publish_activation(
                tensor=tensor,
                activation_id=activation_id,
                direction=direction,
                attestation_challenge_blob=attestation_challenge_blob,
                upload_url=upload_url,
                activation_path=activation_path,
            )
        )
        self._publishing_tasks.append(task)

    def publish_loss(self, loss: float, activation_id: str):
        """Publish a loss to the orchestrator."""
        task = asyncio.create_task(self._publish_loss(loss=loss, activation_id=activation_id))
        self._publishing_tasks.append(task)

    async def _publish_activation(
        self,
        tensor: torch.Tensor,
        activation_id: str,
        direction: str,
        attestation_challenge_blob: str | None,
        upload_url: list[str] | None,
        activation_path: str | None,
    ):
        """Upload an activation to the orchestrator."""
        try:
            async with TimerLogger(
                name="upload_activation",
                metadata={
                    "activation_id": activation_id,
                    "direction": direction,
                },
            ):
                upload_response: CompleteFileUploadResponse = await upload_tensor(
                    miner_api_client=self._miner_api_client,
                    tensor=tensor,
                    hotkey=self._miner_api_client.hotkey,
                    upload_urls=upload_url,
                    object_name=activation_path,
                )
                logger.debug(f"tensor shape before upload:{tensor.shape}")

                attestation_payload: MinerAttestationPayload | None = None
                if RUN_FLAGS.attest.isOn():
                    try:
                        challenge = AttestationChallengeResponse(challenge_blob=attestation_challenge_blob)
                        attestation_payload = await asyncio.to_thread(
                            collect_attestation_payload,
                            challenge,
                        )
                        logger.info(
                            f"Collected attestation payload for activation {activation_id}",
                        )
                    except AttestationUnavailableError as exc:
                        error_code = getattr(exc, "error_code", None)
                        code_suffix = f" (error_code={error_code})" if error_code is not None else ""
                        logger.error(
                            f"Attestation unavailable while submitting activation {activation_id}{code_suffix}: {exc}"
                        )
                    except Exception as exc:
                        logger.exception(
                            f"Error collecting attestation for activation {activation_id}: {exc}",
                        )

            async with TimerLogger(
                name="submit_activation",
                metadata={"activation_id": activation_id, "direction": direction},
            ):
                await self._miner_api_client.submit_activation_request(
                    submit_activation_request=SubmitActivationRequest(
                        activation_id=activation_id,
                        activation_path=upload_response.object_path,
                        direction=direction,
                        attestation=attestation_payload,
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
            async with TimerLogger(
                name="publish_loss",
                metadata={"activation_id": activation_id},
            ):
                await self._miner_api_client.report_loss(
                    loss_report=LossReportRequest(activation_id=activation_id, loss=loss),
                )
                logger.success(f"✅ Successfully published loss for activation {activation_id}")

        except (LayerStateException, MinerNotRegisteredException) as e:
            # Swallow expected exceptions
            logger.warning(f"Anticipated exception has occurred while publishing loss (swallowed): {e}")
            pass
        except Exception as e:
            logger.error(f"Failed to publish loss to orchestrator: {e}")
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
