from typing import Literal
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator

from common import settings
from common.models.ml_models import ModelConfig, ModelMetadata
from common.models.run_flags import RunFlags


class WeightsUploadResponse(BaseModel):
    urls: list[str]
    upload_id: str


class FileUploadCompletionRequest(BaseModel):
    object_name: str
    upload_id: str
    parts: list[dict]


class CompleteFileUploadResponse(BaseModel):
    object_path: str


class FileUploadResponse(BaseModel):
    object_name: str
    urls: list[str]
    upload_id: str


class FileUploadRequest(BaseModel):
    num_parts: int
    file_type: Literal["weights", "optimizer_state", "activation", "weights_metadata", "local_optimizer_state"]

    @model_validator(mode="after")
    def validate_num_parts(self):
        if self.num_parts < 1:
            raise HTTPException(status_code=400, detail="Number of parts must be at least 1")
        if self.num_parts > settings.MAX_NUM_PARTS:
            raise HTTPException(status_code=400, detail="Number of parts must be less than 1000")

        return self


class GetTargetsRequest(BaseModel):
    activation_id: str | None = None


class SyncActivationAssignmentsRequest(BaseModel):
    activation_ids: list[str]


class WeightUpdate(BaseModel):
    weights_path: str
    weights_metadata_path: str


class MinerRegistrationResponse(BaseModel):
    layer: int | None = None
    current_epoch: int | None = None
    model_cfg: ModelConfig | None = None
    model_metadata: ModelMetadata | None = None
    run_id: str
    run_flags: RunFlags


class ValidatorRegistrationResponse(BaseModel):
    layer: int | None = None
    miner_uid_to_track: int
    miner_hotkey_to_track: str
    model_cfg: ModelConfig | None = None
    model_metadata: ModelMetadata | None = None
    run_id: str
    run_flags: RunFlags


class LossReportRequest(BaseModel):
    activation_id: str
    loss: float


class ActivationResponse(BaseModel):
    activation_id: str | None = None
    direction: Literal["forward", "backward"] | None = None
    presigned_upload_url: str | None = None
    upload_id: str | None = None
    presigned_download_url: str | None = None
    reason: str | None = None
    attestation_challenge_blob: str | None = None


class SubmittedWeightsAndOptimizerPresigned(BaseModel):
    layer: int
    weights_path_presigned: str
    weight_metadata_path_presigned: str
    weight_metadata_path: str
    weighting_factor: int | None = None


#### Validator related models
class ValidationTaskResponse(BaseModel):
    task_type: str
    success: bool
    score: float
    reason: str | None = None
    run_id: str | None = "default"
    epoch: int | None = -1
    layer_idx: int | None = -1


class ValidatorTask(BaseModel):
    function_name: str
    inputs: dict


class TestTaskModel(BaseModel):
    reason: str


class ValidateActivationModel(BaseModel):
    validator_activation_path: str
    miner_activation_path: str
    direction: Literal["forward", "backward"]


class ValidateWeightsAndOptimizerStateModel(BaseModel):
    weights_path: str
    optimizer_state_path: str


class ValidatorResetModel(BaseModel):
    pass


class ValidatorSetBurnRateModel(BaseModel):
    burn_factor: float


class MinerScore(BaseModel):
    """Miner's incentive details"""

    uid: int
    hotkey: str

    # Assigned run_id
    run_id: str

    # The score for the given time window for this run
    # Calculated by: sum(scores within time window) * multipler
    total_score: float

    # Percentage of the incentive_perc assigned to this miner (these total to 1.0 across all miners *in the run*)
    # Calculated by: total_score / all total_scores for the run
    run_weight: float | None = None

    # Overall weight for this hotkey (these total to 1.0 across all miners)
    # Calculated by: weight_in_run * (1 - run's burn rate) * run's incentive_perc
    weight: float | None = None


class RunIncentiveAllocation(BaseModel):
    """Run incentive allocation details"""

    run_id: str

    # Weight of the run that determines percentage of incentive allocated for this run
    incentive_weight: float

    # Percentage of incentive allocated for this run
    # Calculated by: incentive_weight / total_incentive_weight
    incentive_perc: float | None = None

    # How much of the allocated incentive is burned for this run
    burn_factor: float


class SubnetScores(BaseModel):
    """Details about a subnet's scores (weights)"""

    miner_scores: list[MinerScore]
    runs: list[RunIncentiveAllocation]

    # Overall burn factor calculated for the subnet
    # Calculated by: 1 - sum(all miner weights)
    burn_factor: float


class GetActivationRequest(BaseModel):
    n_fwd_activations: int = 1


class MinerAttestationRuntime(BaseModel):
    duration_ms: float
    delay_suspect: bool


class MinerAttestationPayload(BaseModel):
    payload_blob: str
    runtime: MinerAttestationRuntime | None = Field(default=None, exclude=True, repr=False)


class AttestationChallengeResponse(BaseModel):
    challenge_blob: str


class SubmitActivationRequest(BaseModel):
    direction: Literal["forward", "backward"]
    activation_id: str | None = None
    activation_path: str | None = None
    attestation: MinerAttestationPayload | None = None


class RegisterMinerRequest(BaseModel):
    run_id: str
    coldkey: str | None = None  # (optional for miner pool miners)
    attestation: MinerAttestationPayload | None = None


class RunInfo(BaseModel):
    run_id: str
    is_default: bool
    num_miners: int
    whitelisted: bool
    burn_factor: float
    incentive_perc: float
    authorized: bool
    run_flags: RunFlags
