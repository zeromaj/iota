import os
import csv
import time
import shutil
import asyncio
from enum import Enum
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from storage.serializers import ActivationResponse
from loguru import logger
import settings
from utils.s3_interactions import file_exists


# Constants
LOG_FILE = os.getenv("LOG_FILE", "activation_log.csv")
MAX_LOG_ROWS = 500_000


def cleanup_activation_cache():
    """Clean up the activation cache directory and log file."""
    try:
        if os.path.exists(settings.ACTIVATION_DIR):
            shutil.rmtree(settings.ACTIVATION_DIR)
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
    except Exception as e:
        logger.exception(f"Error cleaning up activation cache: {e}")


def _log_operation(
    operation: str,
    activation_uid: str,
    layer: int,
    direction: str,
    response: list[str] = None,
    miner_hotkey: str = None,
):
    # Read existing rows
    rows = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            rows = list(reader)

    # Add new row
    rows.append(
        [
            time.time(),
            operation,
            activation_uid,
            layer,
            direction,
            response,
            miner_hotkey,
        ]
    )

    # Keep only last MAX_LOG_ROWS
    rows = rows[-MAX_LOG_ROWS:]

    # Write back to file
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "operation",
                "activation_uid",
                "layer",
                "direction",
                "response",
            ]
        )
        writer.writerows(rows)


class ActivationState(str, Enum):
    """Enumeration of possible activation states."""

    AVAILABLE = "available"
    PROCESSING = "processing"
    ARCHIVED = "archived"
    REVERTED = "reverted"


class Activation(BaseModel):
    """Persistent object to track the state and history of an activation."""

    activation_uid: str
    direction: Literal["forward", "backward", "initial"]
    path: str
    state: ActivationState = ActivationState.AVAILABLE
    layer: int
    history: list[dict] = Field(default_factory=list)
    processing_start_time: float | None = None
    processing_task: asyncio.Task | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)
    miner_hotkey: str | None = None  # uid of the last miner who processed this activation

    def __init__(self, **data):
        super().__init__(**data)

    def add_to_history(
        self,
        state: ActivationState,
        layer: int,
        path: str,
        direction: str,
        operation: str,
        miner_hotkey: str,
    ):
        """Add a new entry to the activation's history."""
        self.history.append(
            {
                "timestamp": time.time(),
                "state": state,
                "layer": layer,
                "path": path,
                "direction": direction,
                "operation": operation,
                "miner_hotkey": miner_hotkey,
            }
        )

    def get_path(self, layer: int = None, direction: str = None) -> str | None:
        """
        Get the path for a specific layer and direction from history.

        Args:
            layer: Target layer to find. If None, uses the current layer.
            direction: Target direction to find. If None, uses the current direction.

        Returns:
            The path if found, None otherwise.
        """
        # Default to current values if not specified
        target_layer = layer if layer is not None else self.layer
        target_direction = direction if direction is not None else self.direction

        # First check if current state matches the request
        if self.layer == target_layer and self.direction == target_direction:
            if self.path is None:
                logger.error(
                    f"GET PATH [ACTIVATION {self.activation_uid}]: NO CURRENTPATH FOR ACTIVATION {self.activation_uid} DIRECTION {self.direction} LAYER {self.layer}"
                )
                return None
            return self.path

        # Search history from newest to oldest
        for entry in reversed(self.history):
            if entry["layer"] == target_layer and entry["direction"] == target_direction:
                if entry["path"] is None:
                    logger.error(
                        f"GET PATH [ACTIVATION {self.activation_uid}]: NO HISTORIC PATH FOR ACTIVATION {self.activation_uid} DIRECTION {self.direction} LAYER {self.layer}"
                    )
                    return None
                return entry["path"]

        # Nothing found
        logger.error(
            f"No path found for activation {self.activation_uid} with direction {self.direction} in layer {target_layer}. Activation is in layer {self.layer}, direction {self.direction}, state {self.state}.\n\nHistory: {self.history}"
        )
        raise ValueError(
            f"No (historic) path found for activation {self.activation_uid} with direction {self.direction} in layer {target_layer}. Activation: {self}"
        )

    def mark_processing(self, timeout_seconds: int = 60, miner_hotkey: str = None):
        """Mark the activation as 'processing' which means that it was downloaded by a miner and is being processed by it.
        After timeout_seconds, the activation is considered 'stale' and will be reverted to its previous layer so it can be
        processed by another miner.
        """
        previous_layer = self.layer
        previous_path = self.path

        self.state = ActivationState.PROCESSING
        self.processing_start_time = time.time()

        # Add to history
        self.add_to_history(
            state=self.state,
            layer=self.layer,
            path=self.path,
            direction=self.direction,
            operation="processing",
            miner_hotkey=miner_hotkey,
        )

        # Create a task to revert if processing takes too long
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()

        async def revert_after_timeout():
            await asyncio.sleep(timeout_seconds)
            if self.state == ActivationState.PROCESSING:
                logger.warning(
                    f"Processing timeout for activation {self.activation_uid} with direction {self.direction}"
                )
                self.state = ActivationState.REVERTED
                self.layer = previous_layer
                self.path = previous_path
                self.processing_start_time = None
                _log_operation(
                    "revert",
                    self.activation_uid,
                    self.layer,
                    self.direction,
                    miner_hotkey=miner_hotkey,
                )

                # Add reversion to history
                self.add_to_history(
                    state=self.state,
                    layer=self.layer,
                    path=self.path,
                    direction=self.direction,
                    operation="revert",
                    miner_hotkey=miner_hotkey,
                )

        self.processing_task = asyncio.create_task(revert_after_timeout())
        return self.processing_task


class ActivationStore(BaseModel):
    # Forward/backward activations
    activations: dict[str, Activation] = Field(default_factory=dict)
    # Initial activations (separate to avoid overwriting)
    initial_activations: dict[str, Activation] = Field(default_factory=dict)

    def __init__(self, **data):
        # Clean up any existing cache and logs
        super().__init__(**data)
        cleanup_activation_cache()
        # Initialize new log file
        self._init_log_file()
        _log_operation("initialized_file", None, None, None, None, None)

    def _init_log_file(self):
        os.makedirs(settings.ACTIVATION_DIR, exist_ok=True)
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "operation",
                    "uid",
                    "layer",
                    "direction",
                    "response",
                    "miner_hotkey",
                ]
            )

    def get_activation_path(
        self, activation_uid: str, direction: Literal["forward", "backward", "initial"], layer: int
    ):
        """Get an activation from the activation store."""
        activation_dict = self._get_activation_dict(direction)
        if activation_uid not in activation_dict:
            return None
        return activation_dict[activation_uid].get_path(layer=layer, direction=direction)

    def _get_activation_dict(self, direction: Literal["forward", "backward", "initial"]):
        """Get the appropriate activation dictionary based on direction."""
        if direction == "initial":
            return self.initial_activations
        else:
            return self.activations

    async def upload_activation_to_activation_store(
        self,
        activation_uid: str,
        layer: int,
        direction: Literal["forward", "backward", "initial"],
        activation_path: str,
        miner_hotkey: str,
    ):
        try:
            if not file_exists(activation_path):
                logger.warning(
                    f"Activation {activation_uid} not found in {activation_path}, miner {miner_hotkey} uploaded and invalid activation"
                )
                return None

            logger.debug(
                f"UPLOADING ACTIVATION |  {activation_uid} | DIRECTION: {direction} | LAYER: {layer} | PATH: {activation_path}"
            )
            if layer < 0 or layer >= settings.N_LAYERS:
                raise ValueError(f"Layer {layer} is not valid")

            # Check if this activation exists. If not, assert it gets uploaded by a layer 0 miner
            if activation_uid not in self.activations and activation_uid not in self.initial_activations:
                if (direction == "forward" or direction == "initial") and layer == 0:
                    logger.debug("New activation added by layer 0 miner")
                else:
                    raise ValueError(
                        f"Miner tried to upload activation {activation_uid} with direction {direction} in layer {layer} but it doesn't exist"
                    )

            # Calculate target layer based on direction
            target_layer = layer
            if direction == "forward":
                target_layer = layer + 1
            elif direction == "backward":
                target_layer = layer - 1
            elif direction == "initial":
                target_layer = settings.N_LAYERS - 1

            logger.debug(
                f"Activation {activation_uid}, direction {direction}, path {activation_path}, origin layer {layer} has target layer {target_layer}"
            )

            # Get the appropriate activation dictionary, either activations or initial_activations
            activation_dict = self._get_activation_dict(direction)

            # Check if this activation already exists
            if activation_uid not in activation_dict:
                # Create new activation
                activation = Activation(
                    activation_uid=activation_uid,
                    direction=direction,
                    path=activation_path,
                    layer=target_layer,
                    state=ActivationState.AVAILABLE,
                    miner_hotkey=miner_hotkey,
                )
            else:
                if layer != 0:
                    if activation_uid not in activation_dict:
                        logger.error(f"Activation {activation_uid} not found in activation store")
                        return None
                activation = activation_dict[activation_uid]

                # TODO: THINK ABOUT THIS
                if activation.state != ActivationState.PROCESSING:
                    logger.error(
                        f"Activation {activation_uid} with direction {direction} already exists: {activation} in state {activation.state}, not processing. Discarding this upload."
                    )
                    return None

                # Create a new activation with the same UID
                activation = Activation(
                    activation_uid=activation_uid,
                    direction=direction,
                    path=activation_path,
                    layer=target_layer,
                    state=ActivationState.AVAILABLE,
                    history=activation.history,
                    miner_hotkey=miner_hotkey,
                )
            activation.add_to_history(
                state=activation.state,
                layer=activation.layer,
                path=activation.path,
                direction=activation.direction,
                operation="uploaded",
                miner_hotkey=miner_hotkey,
            )

            activation_dict[activation_uid] = activation

            logger.debug(f"RECEIVED ACTIVATION {activation}")
            _log_operation(
                "upload",
                activation_uid,
                target_layer,
                direction,
                miner_hotkey=miner_hotkey,
            )
            return activation_path

        except Exception as e:
            logger.error(f"Error uploading activation: {e}")

    async def download_activation_from_activation_store(
        self,
        activation_uid: str,
        direction: Literal["forward", "backward", "initial"],
        delete: bool,
        layer: int | None = None,
        fetch_historic: bool = False,
        miner_hotkey: str = None,
    ) -> str | None:
        try:
            # Find the activation with this uid and direction
            logger.debug(
                "DOWNLOADING ACTIVATION | UID: {} | DIRECTION: {} | DELETE: {} | LAYER: {}",
                activation_uid,
                direction,
                delete,
                layer,
            )

            # Get the appropriate activation dictionary
            activation_dict = self._get_activation_dict(direction)

            if activation_uid not in activation_dict:
                logger.error(f"No activation found for uid {activation_uid} with direction {direction}")
                return None

            activation = activation_dict[activation_uid]
            # Validators will always have fetch_historic set to True. Therefore,
            # if fetch_historic is False we need to confirm it's the most recent layer
            # and mark the activation as being processed by a miner
            if not fetch_historic:
                if layer is not None and layer != activation.layer:
                    #
                    logger.error(
                        f"Activation for uid {activation_uid} with direction {direction} is in layer {activation.layer}, not {layer}"
                    )
                    return None
                if activation.state == ActivationState.PROCESSING:
                    logger.warning(
                        f"""Activation for uid {activation_uid} with direction {direction} is already being processed by a miner.
                        This is likely because two miners listed the same activation and are now both trying to process it. Currently, this activation
                        is being processed by miner {activation.miner_hotkey} and the request is coming from miner {miner_hotkey}."""
                    )
                    return None
                activation.mark_processing(timeout_seconds=60)

            # Try to get historic path from the specified layer
            historic_path = activation.get_path(layer=layer, direction=direction)
            _log_operation(
                "download-historic" if fetch_historic else "download",
                activation_uid,
                layer,
                direction,
                miner_hotkey=miner_hotkey,
            )
            return historic_path

        except Exception as e:
            logger.error(f"Error downloading activation: {e}")

    async def list_activations(
        self,
        layer: int,
        direction: Literal["forward", "backward", "initial"],
        include_pending: bool = False,
        miner_hotkey: str = None,
    ) -> list[Activation]:
        try:
            # Get activations in the specified layer with the specified direction
            result: list[Activation] = []

            # Get the appropriate activation dictionary
            activation_dict = self._get_activation_dict(direction)

            for uid, activation in activation_dict.items():
                # Skip if layer doesn't match
                if activation.layer != layer or activation.direction != direction:
                    continue

                # Add based on state
                if activation.state == ActivationState.AVAILABLE or activation.state == ActivationState.REVERTED:
                    result.append(activation)
                if include_pending:
                    if activation.state == ActivationState.PROCESSING or activation.state == ActivationState.ARCHIVED:
                        result.append(activation)

            _log_operation(
                "list_include_pending" if include_pending else "list",
                None,
                layer,
                direction,
                response=[activation.activation_uid for activation in result],
                miner_hotkey=miner_hotkey,
            )
            return result
        except Exception as e:
            logger.error(f"Error listing activations: {e}")
            return []  # Return empty list instead of None when there's an error

    async def get_miner_activation(self, layer: int, cached_activations: list[str], hotkey: str):
        """Get a random activation for a miner.

        Args:
            hotkey: The hotkey of the miner
        """
        # First try to get a backward activation from the miner's cache
        backwards_activations = await self.list_activations(
            layer=layer,
            direction="backward",
            include_pending=False,
            miner_hotkey=hotkey,
        )
        for available_activation in backwards_activations:
            if available_activation.activation_uid in cached_activations:
                logger.debug(
                    f"RETURNING BACKWARD ACTIVATION {available_activation} FOR MINER {hotkey} ON LAYER {layer}"
                )
                return ActivationResponse(
                    activation_uid=available_activation.activation_uid,
                    direction="backward",
                    activation_path=available_activation.path,
                )
        # This is just for logging, remove later
        logger.debug(f"NO BACKWARD ACTIVATION FOUND FOR MINER {hotkey} ON LAYER {layer}")
        forwards_activations = await self.list_activations(
            layer=layer,
            direction="forward",
            include_pending=False,
            miner_hotkey=hotkey,
        )
        if not forwards_activations:
            logger.debug(f"NO FORWARDS ACTIVATION FOUND FOR MINER {hotkey} ON LAYER {layer}")
            return ActivationResponse(activation_uid=None, direction=None, activation_path=None)

        if layer < settings.N_LAYERS - 1:
            assert (
                forwards_activations[0].activation_uid is not None and forwards_activations[0].path is not None
            ), f"Activation is required for layer {layer}, activation: {forwards_activations[0]}"
            logger.debug(f"RETURNING FORWARD ACTIVATION {forwards_activations[0]} FOR MINER {hotkey} ON LAYER {layer}")
            response = ActivationResponse(
                activation_uid=forwards_activations[0].activation_uid,
                direction="forward",
                activation_path=forwards_activations[0].path,
            )
            logger.debug(f"RETURNING FORWARD ACTIVATION {response} FOR MINER {hotkey} ON LAYER {layer}")
            return response

        for forward_activation in forwards_activations:
            if forward_activation.activation_uid in self.initial_activations.keys():
                initial_activation = self.initial_activations[forward_activation.activation_uid]
                assert (
                    initial_activation.activation_uid is not None and initial_activation.path is not None
                ), f"Initial activation is required for layer {layer}, activation: {initial_activation}"
                response = ActivationResponse(
                    activation_uid=forward_activation.activation_uid,
                    direction="forward",
                    activation_path=forward_activation.path,
                    initial_activation=initial_activation.activation_uid,
                    initial_activation_path=initial_activation.path,
                )
                return response

        logger.error(
            f"No activation found for miner {hotkey} on layer {layer}. Forward activations: {forwards_activations}. Backwards activations: {backwards_activations}. Cached by miner: {cached_activations}"
        )

    async def get_activations_stats(self):
        stats = {}
        for layer in range(-1, settings.N_LAYERS):
            stats[layer] = {
                "forward": 0,
                "backward": 0,
                "initial": 0,
                "processing": 0,
                "pending_deletion": 0,
                "archived": 0,
            }

        # Count regular activations by layer, direction and state
        for activation in self.activations.values():
            layer = activation.layer
            direction = activation.direction
            state = activation.state

            # Skip if layer is out of range
            if layer < -1 or layer >= settings.N_LAYERS:
                continue

            # Increment direction counter
            stats[layer][direction] += 1

            # Increment state counter
            stats[layer][state.value] += 1

        # Count initial activations separately
        for activation in self.initial_activations.values():
            layer = activation.layer
            state = activation.state

            # Skip if layer is out of range
            if layer < -1 or layer >= settings.N_LAYERS:
                continue

            # Increment direction counter (always "initial")
            stats[layer]["initial"] += 1

            # Increment state counter
            stats[layer][state.value] += 1

        return stats

    async def is_activation_active(self, layer: int, activation_uid: str) -> bool:
        """Check if an activation is still needed by checking if it exists as a backward activation in any higher layer."""
        try:
            normal_activation_active = False
            initial_activation_active = False

            # Check in regular activations
            if activation_uid in self.activations:
                activation = self.activations[activation_uid]

                # If the activation is in a higher layer and not archived or reverted
                if (
                    activation.layer >= layer
                    and activation.state != ActivationState.ARCHIVED
                    and activation.state != ActivationState.REVERTED
                ):
                    normal_activation_active = True

            # Check in initial activations
            if activation_uid in self.initial_activations:
                activation = self.initial_activations[activation_uid]

                # Initial activations in the target layer or higher are considered active
                if (
                    activation.layer >= layer
                    and activation.state != ActivationState.ARCHIVED
                    and activation.state != ActivationState.REVERTED
                ):
                    initial_activation_active = True

            return normal_activation_active and initial_activation_active
        except Exception as e:
            logger.error(f"Error checking if activation is active: {e}")
            return False

    async def does_activation_exist(self, activation_uid: str) -> bool:
        """Check if an activation exists in the activation store."""
        return activation_uid in self.activations or activation_uid in self.initial_activations

    async def reset_layer(self, layer: int):
        _log_operation(
            "reset_layer",
            None,
            layer,
            None,
            None,
            None,
        )
        for activation in self.activations.values():
            if activation.layer == layer:
                activation.state = ActivationState.ARCHIVED
        for activation in self.initial_activations.values():
            if activation.layer == layer:
                activation.state = ActivationState.ARCHIVED

    def __del__(self):
        # Cleanup any remaining activation files
        if os.path.exists(settings.ACTIVATION_DIR):
            for file in Path(settings.ACTIVATION_DIR).glob("*.pt"):
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Failed to delete activation file {file}: {e}")
