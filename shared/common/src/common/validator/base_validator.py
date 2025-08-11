from abc import ABC, abstractmethod
from common.models.api_models import (
    ValidateActivationModel,
    ValidationTaskResponse,
)


class BaseValidator(ABC):
    @abstractmethod
    async def validate_activations(self, activations: list[ValidateActivationModel]) -> ValidationTaskResponse:
        raise NotImplementedError

    @abstractmethod
    async def reset_validator(self) -> ValidationTaskResponse:
        raise NotImplementedError

    @abstractmethod
    async def set_burn_factor(self, burn_factor: float) -> ValidationTaskResponse:
        raise NotImplementedError
