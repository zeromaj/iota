import time
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Any, Optional
from settings import MAX_ACTIVATION_CACHE_SIZE


class MinerData(BaseModel):
    """Holds all data for a miner"""

    hotkey: str
    layer: int | None = None
    status: Literal["forward", "backward", "idle"] = "idle"
    backwards_since_reset: int = 0
    miner_scores: List[float] = []
    responsible_validator: str | None = None
    last_s3_upload: float = Field(default_factory=time.time)

    # Throughput tracking
    processed_activations: int = 0
    registration_time: int = Field(default_factory=lambda: int(time.time()))
    last_throughput_report: float = Field(default_factory=time.time)

    # Cache tracking - maps activation_uid to timestamp when cached
    cached_activations: Dict[str, float] = Field(default_factory=dict)

    # Merge session tracking
    merge_session_status: str = "idle"
    current_merge_session_id: Optional[str] = None
    last_weight_upload_start: Optional[float] = None
    last_weight_upload_complete: Optional[float] = None
    last_partition_merge_start: Optional[float] = None
    last_partition_merge_complete: Optional[float] = None

    # Performance tracking
    last_activity_timestamp: float = Field(default_factory=time.time)
    merge_participation_count: int = 0

    @property
    def out_of_cache(self) -> bool:
        """Check if the miner is out of cache."""
        return len(self.cached_activations) >= MAX_ACTIVATION_CACHE_SIZE

    def update_merge_status(self, status: str, session_id: Optional[str] = None) -> None:
        """Update the miner's merge session status."""
        self.merge_session_status = status
        if session_id:
            self.current_merge_session_id = session_id
        self.last_activity_timestamp = time.time()

    def start_weight_upload(self, session_id: str) -> None:
        """Mark the start of weight upload process."""
        self.last_weight_upload_start = time.time()
        self.update_merge_status("uploading_weights", session_id)

    def complete_weight_upload(self) -> None:
        """Mark the completion of weight upload process."""
        self.last_weight_upload_complete = time.time()
        self.update_merge_status("weights_uploaded")

    def start_partition_merge(self) -> None:
        """Mark the start of partition merging process."""
        self.last_partition_merge_start = time.time()
        self.update_merge_status("merging_partitions")

    def complete_partition_merge(self) -> None:
        """Mark the completion of partition merging process."""
        self.last_partition_merge_complete = time.time()
        self.merge_participation_count += 1
        self.update_merge_status("merge_complete")

    def fail_merge_session(self, reason: str = "unknown") -> None:
        """Mark the merge session as failed for this miner."""
        self.update_merge_status("merge_failed")

    def reset_merge_status(self) -> None:
        """Reset merge session status to idle."""
        self.merge_session_status = "idle"
        self.current_merge_session_id = None
        self.last_activity_timestamp = time.time()

    def get_weight_upload_duration(self) -> Optional[float]:
        """Get the duration of the last weight upload if both start and complete times exist."""
        if self.last_weight_upload_start and self.last_weight_upload_complete:
            return self.last_weight_upload_complete - self.last_weight_upload_start
        return None

    def get_partition_merge_duration(self) -> Optional[float]:
        """Get the duration of the last partition merge if both start and complete times exist."""
        if self.last_partition_merge_start and self.last_partition_merge_complete:
            return self.last_partition_merge_complete - self.last_partition_merge_start
        return None


class MinerRegistry(BaseModel):
    """Class for handling miner scores, credibilities, and other attributes."""

    registry: Dict[str, MinerData] = Field(default_factory=dict)

    def __init__(self, miner_hotkeys: List[str], **data):
        super().__init__(**data)
        for miner_hotkey in miner_hotkeys:
            self.add_miner_to_registry(miner_hotkey)

    def add_miner_to_registry(self, miner_hotkey: str, layer: int | None = None) -> None:
        """Adds a miner to the registry with default values.

        Args:
            miner_hotkey: The hotkey of the miner
            layer: Optional layer assignment for the miner
        """
        miner_data = MinerData(hotkey=miner_hotkey, layer=layer)
        self.registry[miner_hotkey] = miner_data

    def get_miner_data(self, miner_hotkey: str) -> MinerData | None:
        """Get miner data for a specific miner."""
        miner: MinerData | None = self.registry.get(miner_hotkey)
        if miner is None:
            raise ValueError(f"Miner {miner_hotkey} not found in registry, possible miners: {self.registry.keys()}")
        return miner

    def update_miner_data(self, miner_hotkey: str, miner_data: MinerData) -> None:
        """Update miner data for a specific miner."""
        self.registry[miner_hotkey] = miner_data

    def get_miners_in_layer(self, layer: int) -> list[MinerData]:
        """Get all miners in a specific layer."""
        return [miner for miner in self.registry.values() if miner.layer == layer]

    def get_all_miner_data(self) -> Dict[str, MinerData]:
        """Get all miner data."""
        return self.registry

    def remove_miner_from_registry(self, miner_hotkey: str) -> None:
        """Remove miner data for a specific miner."""
        self.registry.pop(miner_hotkey)

    def set_miner_attribute(self, miner_hotkey: str, attribute: str, value: Any) -> None:
        """Set an attribute for a specific miner."""
        # check if the miner_hotkey exists
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        # Get the miner data
        miner_data = self.registry[miner_hotkey]

        # Check if the attribute exists in the MinerData model
        if not hasattr(miner_data, attribute):
            raise ValueError(
                f"Attribute {attribute} not found in registry for miner {miner_hotkey}, possible attributes: {miner_data.__dict__.keys()}"
            )

        # Set the attribute using setattr
        setattr(miner_data, attribute, value)
        # Update the registry with the modified data
        self.registry[miner_hotkey] = miner_data

    def add_to_miner_cache(self, miner_hotkey: str, activation_uid: str) -> None:
        """Add an activation to a miner's cache tracking.

        This should be called when a miner downloads a forward activation
        (which gets cached for later backward pass).

        Args:
            miner_hotkey: The hotkey of the miner
            activation_uid: The activation UID that was cached
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        current_time = time.time()

        # Add to cache
        miner_data.cached_activations[activation_uid] = current_time

        self.registry[miner_hotkey] = miner_data

    def remove_from_miner_cache(self, miner_hotkey: str, activation_uid: str) -> None:
        """Remove an activation from a miner's cache tracking.

        This should be called when a miner completes a backward pass
        (which removes the activation from their cache).

        Args:
            miner_hotkey: The hotkey of the miner
            activation_uid: The activation UID that was removed from cache
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.cached_activations.pop(activation_uid, None)
        self.registry[miner_hotkey] = miner_data

    def get_miner_cached_activations(self, miner_hotkey: str) -> Dict[str, float]:
        """Get the cached activations for a specific miner.

        Args:
            miner_hotkey: The hotkey of the miner

        Returns:
            Dict mapping activation_uid to timestamp when cached
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        return self.registry[miner_hotkey].cached_activations.copy()

    def is_activation_cached_by_miner(self, miner_hotkey: str, activation_uid: str) -> bool:
        """Check if a specific activation is cached by a miner.

        Args:
            miner_hotkey: The hotkey of the miner
            activation_uid: The activation UID to check

        Returns:
            True if the activation is cached by the miner, False otherwise
        """
        if miner_hotkey not in self.registry:
            return False

        return activation_uid in self.registry[miner_hotkey].cached_activations

    def clear_miner_cache(self, miner_hotkey: str) -> None:
        """Clear all cached activations for a miner.

        This should be called during weight merging phases when miners
        clear their caches.

        Args:
            miner_hotkey: The hotkey of the miner
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        self.registry[miner_hotkey].cached_activations = {}

    def get_miners_with_cached_activation(self, activation_uid: str) -> List[str]:
        """Get all miners that have a specific activation cached.

        Args:
            activation_uid: The activation UID to search for

        Returns:
            List of miner hotkeys that have this activation cached
        """
        miners_with_activation = []
        for miner_hotkey, miner_data in self.registry.items():
            if activation_uid in miner_data.cached_activations:
                miners_with_activation.append(miner_hotkey)
        return miners_with_activation

    def update_s3_upload_time(self, miner_hotkey: str) -> None:
        """Update the last S3 upload time for a miner.

        Args:
            miner_hotkey: The hotkey of the miner that uploaded to S3
        """
        self.set_miner_attribute(miner_hotkey=miner_hotkey, attribute="last_s3_upload", value=time.time())

    def update_miner_merge_status(self, miner_hotkey: str, status: str, session_id: Optional[str] = None) -> None:
        """Update miner's merge session status.

        Args:
            miner_hotkey: The hotkey of the miner
            status: The new merge status
            session_id: Optional session ID for tracking
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.update_merge_status(status, session_id)
        self.registry[miner_hotkey] = miner_data

    def start_miner_weight_upload(self, miner_hotkey: str, session_id: str) -> None:
        """Mark that a miner has started uploading weights.

        Args:
            miner_hotkey: The hotkey of the miner
            session_id: The merge session ID
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.start_weight_upload(session_id)
        self.registry[miner_hotkey] = miner_data

    def complete_miner_weight_upload(self, miner_hotkey: str) -> None:
        """Mark that a miner has completed uploading weights.

        Args:
            miner_hotkey: The hotkey of the miner
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.complete_weight_upload()
        self.registry[miner_hotkey] = miner_data

    def start_miner_partition_merge(self, miner_hotkey: str) -> None:
        """Mark that a miner has started merging partitions.

        Args:
            miner_hotkey: The hotkey of the miner
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.start_partition_merge()
        self.registry[miner_hotkey] = miner_data

    def complete_miner_partition_merge(self, miner_hotkey: str) -> None:
        """Mark that a miner has completed merging partitions.

        Args:
            miner_hotkey: The hotkey of the miner
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.complete_partition_merge()
        self.registry[miner_hotkey] = miner_data

    def fail_miner_merge_session(self, miner_hotkey: str, reason: str = "unknown") -> None:
        """Mark that a miner's merge session has failed.

        Args:
            miner_hotkey: The hotkey of the miner
            reason: Reason for failure
        """
        if miner_hotkey not in self.registry:
            raise ValueError(f"Miner {miner_hotkey} not found in registry")

        miner_data = self.registry[miner_hotkey]
        miner_data.fail_merge_session(reason)
        self.registry[miner_hotkey] = miner_data

    def reset_all_miners_merge_status(self) -> None:
        """Reset merge status for all miners to idle."""
        for miner_hotkey in self.registry:
            miner_data = self.registry[miner_hotkey]
            miner_data.reset_merge_status()
            self.registry[miner_hotkey] = miner_data

    def get_miners_by_merge_status(self, layer: Optional[int] = None) -> Dict[str, List[str]]:
        """Get miners grouped by their current merge status.

        Args:
            layer: Optional layer filter

        Returns:
            Dict mapping status to list of miner hotkeys
        """
        status_groups = {}

        for miner_hotkey, miner_data in self.registry.items():
            if layer is not None and miner_data.layer != layer:
                continue

            status = miner_data.merge_session_status
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(miner_hotkey)

        return status_groups

    def get_miners_grid_data(self) -> Dict[str, Any]:
        """Get comprehensive miner data for grid visualization.

        Returns:
            Dict with miners organized by layer with detailed status information
        """
        current_time = time.time()
        layers_data = {}

        for miner_hotkey, miner_data in self.registry.items():
            layer = miner_data.layer
            if layer is None:
                continue

            if layer not in layers_data:
                layers_data[layer] = []

            # Calculate time since last activity
            time_since_activity = current_time - miner_data.last_activity_timestamp

            # Determine if miner is offline (no activity for more than 5 minutes)
            is_offline = time_since_activity > 300

            # Calculate progress for weight upload or partition merge
            progress = 0.0
            if miner_data.merge_session_status == "uploading_weights":
                # Estimate progress based on time elapsed (rough estimate)
                if miner_data.last_weight_upload_start:
                    elapsed = current_time - miner_data.last_weight_upload_start
                    progress = min(elapsed / 60.0, 0.9)  # Assume 60s for weight upload, max 90%
            elif miner_data.merge_session_status == "merging_partitions":
                if miner_data.last_partition_merge_start:
                    elapsed = current_time - miner_data.last_partition_merge_start
                    progress = min(elapsed / 120.0, 0.9)  # Assume 120s for partition merge, max 90%

            miner_info = {
                "hotkey": miner_hotkey,
                "display_hotkey": f"{miner_hotkey[:6]}...{miner_hotkey[-6:]}",
                "status": "offline" if is_offline else miner_data.merge_session_status,
                "layer": layer,
                "last_activity": miner_data.last_activity_timestamp,
                "merge_session_id": miner_data.current_merge_session_id,
                "progress": progress,
                "performance_metrics": {
                    "merge_participation_count": miner_data.merge_participation_count,
                    "weight_upload_duration": miner_data.get_weight_upload_duration(),
                    "partition_merge_duration": miner_data.get_partition_merge_duration(),
                    "time_since_activity": time_since_activity,
                },
            }

            layers_data[layer].append(miner_info)

        return {"layers": layers_data, "timestamp": current_time}
