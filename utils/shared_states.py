from pydantic import BaseModel
import enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class MergingPhase(str, enum.Enum):
    IS_TRAINING = "is_training"  # Running forward and backward passes
    WEIGHTS_UPLOADING = "weights_uploading"  # Miners uploading their full weight matrices
    MINERS_MERGING_PARTITIONS = (
        "miners_merging_partitions"  # Miners download, merge them locally, and upload partitions
    )

    def next(self) -> "MergingPhase":
        if self == MergingPhase.IS_TRAINING:
            return MergingPhase.WEIGHTS_UPLOADING
        elif self == MergingPhase.WEIGHTS_UPLOADING:
            return MergingPhase.MINERS_MERGING_PARTITIONS
        else:
            return MergingPhase.IS_TRAINING


class MergingPhaseManager(BaseModel):
    stage: MergingPhase = MergingPhase.IS_TRAINING
    _reset_task: asyncio.Task | None = None

    async def advance_phase(self, timeout: float, expected_phase: MergingPhase):
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
        logger.debug(f"Advanced merging phase to: {self.stage}")

        # Schedule reset after timeout
        self._reset_task = asyncio.create_task(self._reset_after_timeout(timeout=timeout, current_phase=self.stage))

    async def _reset_after_timeout(self, timeout: float, current_phase: MergingPhase):
        """Reset phase to NOT_MERGING after timeout."""
        if self.stage == current_phase:
            await asyncio.sleep(timeout)
            self.stage = MergingPhase.IS_TRAINING
            logger.debug("Reset merging phase to NOT_MERGING after timeout")
