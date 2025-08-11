from typing import Optional

from common.utils.shared_states import LayerPhase
from pydantic import BaseModel, model_validator


class LayerStateError(BaseModel):
    layer: int
    expected_status: LayerPhase
    actual_status: LayerPhase
    message: Optional[str] = None

    @model_validator(mode="after")
    def make_message(self):
        self.message = (
            f"Layer {self.layer} is not in the expected state {self.expected_status}, actual: {self.actual_status}"
        )
        return self


class MinerNotRegisteredError(BaseModel):
    message: str = "Miner not registered"
    name: Optional[str] = None


class BaseErrorModel(BaseModel):
    error_name: str | None = None
    error_dict: dict | None = None


class SpecVersionError(BaseModel):
    expected_version: int
    actual_version: str
    message: str = "Spec version mismatch"

    @model_validator(mode="after")
    def make_message(self):
        self.message = f"Spec version mismatch. Expected: {self.expected_version}, Received: {self.actual_version}"
        return self
