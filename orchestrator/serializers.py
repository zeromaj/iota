from typing import Any, Dict, List, Literal

from pydantic import BaseModel


class SubmittedWeights(BaseModel):
    weights_path: str
    weight_metadata_path: str
    optimizer_state_path: str
    optimizer_state_metadata_path: str
    hotkey: str
    weighting_factor: int | None = None


class MinerStatusUpdate(BaseModel):
    status: Literal["forward", "backward", "initial", "idle"]
    activation_uid: str | None = None
    activation_path: str | None = None


class MinerRegistrationResponse(BaseModel):
    hotkey: str
    layer: int | None = None
    message: str = "Successfully registered"
    version: str | None = None


class LayerAssignmentResponse(BaseModel):
    layer: int
    message: str = "Layer assigned successfully"


class LossReportRequest(BaseModel):
    activation_uid: str
    loss_value: float


class LossReport(BaseModel):
    hotkey: str
    activation_uid: str
    loss_value: float
    timestamp: float


class LossReportResponse(LossReport):
    message: str = "Loss reported successfully"


class MinerLossesResponse(BaseModel):
    losses: List[LossReport]


class AllLossesResponse(BaseModel):
    losses: Dict[str, List[LossReport]]


class OrchestratorStats(BaseModel):
    total_forwards: int
    total_backwards: int
    total_completed: int
    miners: List[Dict[str, Any]]
    tracked_activations: Dict[str, List[Dict[str, Any]]]
    losses: Dict[str, List[LossReport]] = {}


# Gradient Validator Serializers
class GradientValidationResponse(BaseModel):
    is_valid: bool
    score: float
    reason: str
