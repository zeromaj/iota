import asyncio
import enum

from loguru import logger
from pydantic import BaseModel


class LayerPhase(str, enum.Enum):
    TRAINING = "training"  # Running forward and backward passes
    WEIGHTS_UPLOADING = "weights_uploading"  # Miners uploading their full weight matrices
    MERGING_PARTITIONS = "merging_partitions"  # Miners download, merge them locally, and upload partitions

    def next(self) -> "LayerPhase":
        if self == LayerPhase.TRAINING:
            return LayerPhase.WEIGHTS_UPLOADING
        elif self == LayerPhase.WEIGHTS_UPLOADING:
            return LayerPhase.MERGING_PARTITIONS
        else:
            return LayerPhase.TRAINING

    @classmethod
    def from_str(cls, phase_str: str) -> "LayerPhase":
        """Get the LayerPhase enum value from its string representation."""
        return cls(phase_str)


class MergingPhaseManager(BaseModel):
    stage: LayerPhase = LayerPhase.TRAINING
    _reset_task: asyncio.Task | None = None

    async def advance_phase(self, timeout: float, expected_phase: LayerPhase):
        """Advance to next merging phase and reset after timeout.
        Args:
            timeout: Time in seconds before resetting to NOT_MERGING
            expected_phase: The phase that the orchestrator expects the global state of each layer to be in.
        """
        assert self.stage == expected_phase, f"Expected {expected_phase} but got {self.stage}"
        # Cancel any existing reset task
        if self._reset_task and not self._reset_task.done():
            self._reset_task.cancel()

        # Advance the phase
        self.stage = self.stage.next()
        logger.info(f"Advanced merging phase to: {self.stage}")

        # Schedule reset after timeout
        self._reset_task = asyncio.create_task(self._reset_after_timeout(timeout=timeout, current_phase=self.stage))

    async def _reset_after_timeout(self, timeout: float, current_phase: LayerPhase):
        """Reset phase to NOT_MERGING after timeout."""
        await asyncio.sleep(timeout)  # wait for the timeout first.
        if self.stage == current_phase:
            self.stage = LayerPhase.TRAINING
            logger.info("Reset merging phase to NOT_MERGING after timeout")
