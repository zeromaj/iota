from typing import Any
from common.models.api_models import (
    RunInfo,
    ActivationResponse,
    CompleteFileUploadResponse,
    FileUploadCompletionRequest,
    FileUploadRequest,
    FileUploadResponse,
    GetTargetsRequest,
    LossReportRequest,
    MinerRegistrationResponse,
    RegisterMinerRequest,
    SubmitActivationRequest,
    SubmittedWeightsAndOptimizerPresigned,
    SyncActivationAssignmentsRequest,
    WeightUpdate,
)
from common.models.error_models import BaseErrorModel, LayerStateError, EntityNotRegisteredError, SpecVersionError
from common.utils.exceptions import LayerStateException, MinerNotRegisteredException, SpecVersionException
from common.utils.partitions import MinerPartition
from common.utils.s3_utils import upload_parts
from common.utils.shared_states import LayerPhase
from loguru import logger
from subnet.common_api_client import CommonAPIClient
from substrateinterface.keypair import Keypair


class MinerAPIClient(CommonAPIClient):
    def __init__(self, hotkey: Keypair | None = None):
        self.hotkey = hotkey
        self.layer_state = LayerPhase.TRAINING

    async def get_targets(self, get_targets_request: GetTargetsRequest) -> str | BaseErrorModel:
        response = await CommonAPIClient.orchestrator_request(
            method="POST", path="/miner/get_targets", hotkey=self.hotkey, body=get_targets_request.model_dump()
        )
        return self.parse_response(response)

    async def fetch_run_info_request(self) -> list[RunInfo]:
        response = await CommonAPIClient.orchestrator_request(
            method="GET", path="/common/get_run_info", hotkey=self.hotkey
        )
        parsed_response = self.parse_response(response)
        return [RunInfo.model_validate(run_info) for run_info in parsed_response]

    async def register_miner_request(
        self, register_miner_request: RegisterMinerRequest
    ) -> MinerRegistrationResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST", path="/miner/register", hotkey=self.hotkey, body=register_miner_request.model_dump()
            )
            parsed_response = self.parse_response(response)
            return MinerRegistrationResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error registering miner: {e}")
            raise

    async def get_layer_state_request(self) -> LayerPhase | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET", path="/miner/layer_state", hotkey=self.hotkey
            )
            parsed_response = self.parse_response(response)
            return LayerPhase(parsed_response)
        except Exception as e:
            logger.error(f"Error getting layer state: {e}")
            raise

    async def get_activation(self) -> ActivationResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET", path="/miner/get_activation", hotkey=self.hotkey
            )
            parsed_response = self.parse_response(response)
            return ActivationResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error getting activation: {e}")
            raise

    async def submit_weights(
        self,
        weight_update: WeightUpdate,
    ) -> dict:
        """Attempts to submit weights to the orchestrator"""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_weights",
                hotkey=self.hotkey,
                body=weight_update.model_dump(),
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting weights: {e}")
            raise

    async def report_loss(self, loss_report: LossReportRequest) -> None:
        """Report loss to orchestrator"""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/report_loss",
                hotkey=self.hotkey,
                body=loss_report.model_dump(),
            )
            self.parse_response(response)
        except Exception as e:
            logger.exception(f"Error reporting loss: {e}")
            raise e

    async def submit_activation_request(self, submit_activation_request: SubmitActivationRequest) -> None:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_activation",
                hotkey=self.hotkey,
                body=submit_activation_request.model_dump(),
            )
            self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting activation: {e}")
            raise

    async def sync_activation_assignments(self, activation_ids: list[str]) -> dict[str, bool]:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/sync_activation_assignments",
                hotkey=self.hotkey,
                body=SyncActivationAssignmentsRequest(activation_ids=activation_ids).model_dump(),
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error checking if activation is active: {e}")
            raise

    async def get_partitions(self) -> list[int] | dict:
        """Get the partition indices for a given hotkey."""
        try:
            response: list[MinerPartition] | dict = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_partitions",
                hotkey=self.hotkey,
            )
            return self.parse_response(response)

        except Exception as e:
            logger.error(f"Error getting weight partition info: {e}")
            raise

    async def get_weight_path_per_layer(self) -> list[SubmittedWeightsAndOptimizerPresigned] | dict:
        """Get the weight path for a given layer."""
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_weight_path_per_layer",
                hotkey=self.hotkey,
            )
            parsed_response = self.parse_response(response)
            paths = [SubmittedWeightsAndOptimizerPresigned.model_validate(weight) for weight in parsed_response]
            return paths

        except Exception as e:
            logger.error(f"Error getting weight path per layer: {e}")
            raise

    async def get_num_splits(self) -> int | dict:
        """Get the number of splits for a given hotkey."""
        try:
            response: int | dict = await CommonAPIClient.orchestrator_request(
                method="GET",
                path="/miner/get_num_splits",
                hotkey=self.hotkey,
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error getting number of splits: {e}")
            raise

    async def get_learning_rate(self) -> float | dict:
        """Get the current learning rate."""
        try:
            response: float = await CommonAPIClient.orchestrator_request(
                method="GET", path="/miner/learning_rate", hotkey=self.hotkey
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error getting learning rate: {e}")
            raise

    async def submit_merged_partitions(self, merged_partitions: list[MinerPartition]) -> dict:
        """Submit merged partitions to the orchestrator."""
        try:
            response: dict = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/submit_merged_partitions",
                hotkey=self.hotkey,
                body=[partition.model_dump() for partition in merged_partitions],
            )
            return self.parse_response(response)
        except Exception as e:
            logger.error(f"Error submitting merged partitions: {e}")
            raise

    async def initiate_file_upload_request(
        self,
        hotkey: Keypair,
        file_upload_request: FileUploadRequest,
    ) -> FileUploadResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/initiate_file_upload",
                hotkey=hotkey,
                body=file_upload_request.model_dump(),
            )
            parsed_response = self.parse_response(response)
            return FileUploadResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error initiating file upload: {e}")
            raise

    @classmethod
    async def upload_multipart_to_s3(cls, urls: list[str], data: bytes, upload_id: str) -> list[dict]:
        parts = await upload_parts(urls=urls, data=data, upload_id=upload_id)
        return parts

    async def complete_file_upload_request(
        self,
        hotkey: Keypair,
        file_upload_completion_request: FileUploadCompletionRequest,
    ) -> CompleteFileUploadResponse | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="POST",
                path="/miner/complete_multipart_upload",
                hotkey=hotkey,
                body=file_upload_completion_request.model_dump(),
            )
            parsed_response = self.parse_response(response)
            return CompleteFileUploadResponse.model_validate(parsed_response)
        except Exception as e:
            logger.error(f"Error completing file upload: {e}")
            raise

    async def get_merged_partitions(self, hotkey: Keypair) -> list[MinerPartition] | dict:
        try:
            response = await CommonAPIClient.orchestrator_request(
                method="GET", path="/common/get_merged_partitions", hotkey=hotkey
            )
            parsed_response = self.parse_response(response)
            return [MinerPartition.model_validate(partition) for partition in parsed_response]
        except Exception as e:
            logger.error(f"Error getting merged partitions: {e}")
            raise

    def parse_response(self, response: Any) -> Any:
        if not isinstance(response, dict):
            return response
        if (error_name := response.get("error_name")) is not None:
            if error_name == LayerStateError.__name__:
                logger.warning(f"Layer state change: {response['error_dict']}")
                error_dict = LayerStateError(**response["error_dict"])
                self.layer_state = error_dict.actual_status
                raise LayerStateException(
                    f"Miner is moving state from {error_dict.expected_status} to {error_dict.actual_status}"
                )
            if error_name == EntityNotRegisteredError.__name__:
                logger.error(f"Miner not registered error: {response['error_dict']}")
                raise MinerNotRegisteredException("Miner not registered")
            if error_name == SpecVersionError.__name__:
                logger.error(f"Spec version mismatch: {response['error_dict']}")
                raise SpecVersionException(
                    expected_version=response["error_dict"]["expected_version"],
                    actual_version=response["error_dict"]["actual_version"],
                )
            else:
                raise Exception(f"Unexpected error from orchestrator. Response: {response}")
        else:
            return response
