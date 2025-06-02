import time
import settings
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

from loguru import logger


class ActivationLifecycleState(Enum):
    """States that an activation can be in during its lifecycle."""

    REQUESTED = "requested"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ActivationTimestamps(BaseModel):
    """Simplified timestamps for activation processing."""

    requested: Optional[float] = None  # When miner requests activation
    status_updated: Optional[float] = None  # When miner reports status completion


class ActivationDurations(BaseModel):
    """Simplified calculated durations for activation processing."""

    total_processing_time: Optional[float] = None  # Time from request to status update

    def calculate_from_timestamps(self, timestamps: ActivationTimestamps):
        """Calculate total processing duration from timestamps."""
        if timestamps.requested and timestamps.status_updated:
            self.total_processing_time = timestamps.status_updated - timestamps.requested


class ActivationQueueMetrics(BaseModel):
    """Metrics about activation queues and backlogs."""

    layer: int
    available_forward_activations: int = 0
    available_backward_activations: int = 0
    processing_activations: int = 0
    queue_depth_per_direction: Dict[str, int] = Field(default_factory=dict)
    average_wait_time: float = 0.0
    oldest_activation_age: float = 0.0


class MinerPerformanceMetrics(BaseModel):
    """Performance metrics for individual miners."""

    miner_hotkey: str
    layer: int
    cache_hit_rate: float = 0.0
    average_processing_time: float = 0.0
    throughput_activations_per_minute: float = 0.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    validation_success_rate: float = 0.0
    total_activations_processed: int = 0


class SystemHealthMetrics(BaseModel):
    """System-wide health and performance metrics."""

    total_active_activations: int = 0
    activations_per_layer: Dict[int, int] = Field(default_factory=dict)
    average_system_latency: float = 0.0
    bottleneck_layers: List[int] = Field(default_factory=list)
    system_throughput: float = 0.0
    storage_utilization: float = 0.0


class ActivationMetricsEvent(BaseModel):
    """Event for activation lifecycle tracking."""

    activation_uid: str
    event_type: str
    timestamp: float
    layer: int
    miner_hotkey: str
    state: ActivationLifecycleState
    additional_data: Optional[Dict[str, Any]] = None


class ActivationMetrics(BaseModel):
    """Simplified metrics for a single activation processing cycle."""

    activation_uid: str
    miner_hotkey: str
    layer: int
    direction: Literal["forward", "backward", "initial"]
    current_state: ActivationLifecycleState = ActivationLifecycleState.REQUESTED
    timestamps: ActivationTimestamps = Field(default_factory=ActivationTimestamps)
    durations: ActivationDurations = Field(default_factory=ActivationDurations)
    events: List[ActivationMetricsEvent] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed: bool = False
    failed: bool = False
    error_message: Optional[str] = None

    def _record_event(
        self, event_type: str, state: ActivationLifecycleState, additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record an event in the activation lifecycle."""
        event = ActivationMetricsEvent(
            activation_uid=self.activation_uid,
            event_type=event_type,
            timestamp=time.time(),
            layer=self.layer,
            miner_hotkey=self.miner_hotkey,
            state=state,
            additional_data=additional_data,
        )
        self.events.append(event)
        self.current_state = state

    def mark_requested(self):
        """Mark when activation was requested by miner."""
        self.timestamps.requested = time.time()
        self._record_event("requested", ActivationLifecycleState.REQUESTED)

    def mark_status_updated(self, success: bool = True, error_message: Optional[str] = None):
        """Mark when miner reported status completion."""
        self.timestamps.status_updated = time.time()
        self.durations.calculate_from_timestamps(self.timestamps)

        if success:
            self._record_event("status_updated", ActivationLifecycleState.COMPLETED)
            self.completed = True
        else:
            self._record_event("status_updated", ActivationLifecycleState.FAILED, {"error": error_message})
            self.completed = True
            self.failed = True
            self.error_message = error_message

    def mark_timeout(self):
        """Mark when activation timed out."""
        self.timestamps.status_updated = time.time()
        self.durations.calculate_from_timestamps(self.timestamps)
        self.completed = True
        self.failed = True
        self.error_message = "timeout"
        self._record_event("timeout", ActivationLifecycleState.TIMEOUT)


class ActivationMetricsCollector(BaseModel):
    """Simplified collector for managing activation timing metrics."""

    active_metrics: Dict[str, ActivationMetrics] = Field(default_factory=dict)
    completed_metrics: List[ActivationMetrics] = Field(default_factory=list)
    queue_metrics: Dict[int, ActivationQueueMetrics] = Field(default_factory=dict)
    miner_performance: Dict[str, MinerPerformanceMetrics] = Field(default_factory=dict)
    system_health: SystemHealthMetrics = Field(default_factory=SystemHealthMetrics)
    max_completed_history: int = Field(default=1000)

    def create_activation_metric(
        self, activation_uid: str, miner_hotkey: str, layer: int, direction: Literal["forward", "backward", "initial"]
    ) -> ActivationMetrics:
        """Create a new activation metric entry."""
        metric = ActivationMetrics(
            activation_uid=activation_uid, miner_hotkey=miner_hotkey, layer=layer, direction=direction
        )
        self.active_metrics[activation_uid] = metric
        return metric

    def get_or_create_metric(
        self, activation_uid: str, miner_hotkey: str, layer: int, direction: Literal["forward", "backward", "initial"]
    ) -> ActivationMetrics:
        """Get existing metric or create new one."""
        if activation_uid in self.active_metrics:
            return self.active_metrics[activation_uid]
        return self.create_activation_metric(activation_uid, miner_hotkey, layer, direction)

    def record_activation_requested(self, activation_uid: str, miner_hotkey: str, layer: int, direction: str):
        """Record when activation was requested."""
        # Cast direction to the expected literal type
        valid_directions = {"forward", "backward", "initial"}
        if direction not in valid_directions:
            logger.warning(f"Invalid direction '{direction}', defaulting to 'forward'")
            direction = "forward"

        direction_typed: Literal["forward", "backward", "initial"] = direction  # type: ignore
        metric = self.get_or_create_metric(activation_uid, miner_hotkey, layer, direction_typed)
        metric.mark_requested()
        logger.debug(f"Recorded activation request: {activation_uid} by {miner_hotkey[:8]}...")

    def record_status_updated(self, activation_uid: str, success: bool = True, error_message: Optional[str] = None):
        """Record when status was updated and move to completed metrics."""
        if activation_uid in self.active_metrics:
            metric = self.active_metrics[activation_uid]
            metric.mark_status_updated(success, error_message)

            # Move to completed metrics
            self.completed_metrics.append(metric)
            del self.active_metrics[activation_uid]

            # Update miner performance metrics
            self._update_miner_performance(metric)

            # Maintain history limit
            if len(self.completed_metrics) > self.max_completed_history:
                self.completed_metrics = self.completed_metrics[-self.max_completed_history :]

            if metric.durations.total_processing_time:
                logger.debug(
                    f"Completed activation metrics: {activation_uid}, total time: {metric.durations.total_processing_time:.2f}s"
                )
            return metric
        return None

    def record_timeout(self, activation_uid: str):
        """Record when activation timed out."""
        if activation_uid in self.active_metrics:
            metric = self.active_metrics[activation_uid]
            metric.mark_timeout()

            # Move to completed metrics
            self.completed_metrics.append(metric)
            del self.active_metrics[activation_uid]

            # Update miner performance metrics
            self._update_miner_performance(metric)

            logger.warning(f"Activation timed out: {activation_uid}")
            return metric
        return None

    def _update_miner_performance(self, metric: ActivationMetrics):
        """Update miner performance metrics based on completed activation."""
        hotkey = metric.miner_hotkey
        if hotkey not in self.miner_performance:
            self.miner_performance[hotkey] = MinerPerformanceMetrics(miner_hotkey=hotkey, layer=metric.layer)

        perf = self.miner_performance[hotkey]
        perf.total_activations_processed += 1

        # Update error and timeout rates
        if metric.failed:
            if metric.error_message == "timeout":
                # Calculate timeout rate
                timeouts = sum(
                    1 for m in self.completed_metrics if m.miner_hotkey == hotkey and m.error_message == "timeout"
                )
                perf.timeout_rate = timeouts / perf.total_activations_processed
            else:
                # Calculate error rate
                errors = sum(
                    1
                    for m in self.completed_metrics
                    if m.miner_hotkey == hotkey and m.failed and m.error_message != "timeout"
                )
                perf.error_rate = errors / perf.total_activations_processed

        # Update processing time
        if metric.durations.total_processing_time:
            # Running average of processing time
            total_processing_time = (
                perf.average_processing_time * (perf.total_activations_processed - 1)
                + metric.durations.total_processing_time
            )
            perf.average_processing_time = total_processing_time / perf.total_activations_processed

    def update_queue_metrics(self, layer: int, queue_data: Dict[str, Any]):
        """Update queue metrics for a layer."""
        if layer not in self.queue_metrics:
            self.queue_metrics[layer] = ActivationQueueMetrics(layer=layer)

        queue_metric = self.queue_metrics[layer]
        queue_metric.available_forward_activations = queue_data.get("forward_count", 0)
        queue_metric.available_backward_activations = queue_data.get("backward_count", 0)
        queue_metric.processing_activations = queue_data.get("processing_count", 0)
        queue_metric.queue_depth_per_direction = queue_data.get("queue_depths", {})
        queue_metric.average_wait_time = queue_data.get("avg_wait_time", 0.0)
        queue_metric.oldest_activation_age = queue_data.get("oldest_age", 0.0)

    def update_system_health(self):
        """Update system-wide health metrics."""
        self.system_health.total_active_activations = len(self.active_metrics)

        # Calculate activations per layer
        layer_counts = defaultdict(int)
        for metric in self.active_metrics.values():
            layer_counts[metric.layer] += 1
        self.system_health.activations_per_layer = dict(layer_counts)

        # Calculate average system latency
        recent_metrics = [m for m in self.completed_metrics[-100:] if m.durations.total_processing_time]
        if recent_metrics:
            self.system_health.average_system_latency = sum(
                m.durations.total_processing_time
                for m in recent_metrics
                if m.durations.total_processing_time is not None
            ) / len(recent_metrics)

        # Identify bottleneck layers (layers with high queue depths)
        bottlenecks = []
        for layer, queue_metric in self.queue_metrics.items():
            total_queue = (
                queue_metric.available_forward_activations
                + queue_metric.available_backward_activations
                + queue_metric.processing_activations
            )
            if total_queue > 10:  # Threshold for bottleneck
                bottlenecks.append(layer)
        self.system_health.bottleneck_layers = bottlenecks

        # Calculate system throughput (activations per minute)
        current_time = time.time()
        recent_completions = [m for m in self.completed_metrics if current_time - m.created_at <= 60]  # Last minute
        self.system_health.system_throughput = len(recent_completions)

    def get_miner_average_times(
        self, miner_hotkey: str, time_window_seconds: Optional[float] = None
    ) -> Dict[str, float]:
        """Get average processing times for a specific miner."""
        current_time = time.time()
        relevant_metrics = []

        for metric in self.completed_metrics:
            if metric.miner_hotkey == miner_hotkey:
                if time_window_seconds is None or (current_time - metric.created_at) <= time_window_seconds:
                    relevant_metrics.append(metric)

        if not relevant_metrics:
            return {}

        # Calculate averages
        def safe_avg(values):
            return sum(values) / len(values) if values else 0

        processing_times = [
            m.durations.total_processing_time for m in relevant_metrics if m.durations.total_processing_time
        ]

        return {
            "avg_processing_time": safe_avg(processing_times),
            "sample_count": len(relevant_metrics),
            "success_rate": sum(1 for m in relevant_metrics if not m.failed) / len(relevant_metrics),
        }

    def get_layer_statistics(self, layer: int, time_window_seconds: Optional[float] = None) -> Dict[str, Any]:
        """Get processing statistics for a specific layer."""
        current_time = time.time()
        relevant_metrics = []

        for metric in self.completed_metrics:
            if metric.layer == layer:
                if time_window_seconds is None or (current_time - metric.created_at) <= time_window_seconds:
                    relevant_metrics.append(metric)

        if not relevant_metrics:
            return {}

        processing_times = [
            m.durations.total_processing_time for m in relevant_metrics if m.durations.total_processing_time
        ]

        return {
            "count": len(relevant_metrics),
            "avg_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0,
            "min_processing_time": min(processing_times) if processing_times else 0,
            "max_processing_time": max(processing_times) if processing_times else 0,
            "throughput_per_minute": len(relevant_metrics) / (time_window_seconds / 60) if time_window_seconds else 0,
            "success_rate": sum(1 for m in relevant_metrics if not m.failed) / len(relevant_metrics),
            "active_count": len([m for m in self.active_metrics.values() if m.layer == layer]),
        }

    def cleanup_stale_metrics(self, max_age_seconds: float = 3600):
        """Clean up metrics that have been active too long (likely orphaned)."""
        current_time = time.time()
        stale_uids = []

        for uid, metric in self.active_metrics.items():
            if (current_time - metric.created_at) > max_age_seconds:
                stale_uids.append(uid)

        for uid in stale_uids:
            logger.warning(f"Cleaning up stale activation metric: {uid}")
            # Mark as timeout before removing
            self.record_timeout(uid)

        return len(stale_uids)


class TimeSeriesMetricsCollector(BaseModel):
    """Collects time-series performance metrics for historical analysis."""

    historical_data: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)
    max_history_hours: int = Field(default=24)
    collection_interval_seconds: int = Field(default=60)  # Collect every minute
    last_collection_time: float = Field(default_factory=time.time)

    def collect_metrics_snapshot(self, orchestrator):
        """Collect a snapshot of current system metrics."""
        current_time = time.time()

        # Skip if not enough time has passed
        if current_time - self.last_collection_time < self.collection_interval_seconds:
            return

        # Collect system-wide metrics
        system_snapshot = {
            "timestamp": current_time,
            "total_active_activations": len(orchestrator.activation_metrics_collector.active_metrics),
            "system_throughput": orchestrator.activation_metrics_collector.system_health.system_throughput,
            "average_latency": orchestrator.activation_metrics_collector.system_health.average_system_latency,
            "bottleneck_layers": orchestrator.activation_metrics_collector.system_health.bottleneck_layers.copy(),
        }

        # Calculate success rate from recent completions
        recent_completions = [
            m
            for m in orchestrator.activation_metrics_collector.completed_metrics[-100:]
            if current_time - m.created_at <= 300
        ]  # Last 5 minutes
        total_recent = len(recent_completions)
        successful_recent = sum(1 for m in recent_completions if not m.failed)
        system_snapshot["success_rate"] = successful_recent / total_recent if total_recent > 0 else 0

        # Store system metrics
        if "system" not in self.historical_data:
            self.historical_data["system"] = []
        self.historical_data["system"].append(system_snapshot)

        # Collect layer-specific metrics
        for layer in range(orchestrator.N_LAYERS):
            layer_key = f"layer_{layer}"
            if layer_key not in self.historical_data:
                self.historical_data[layer_key] = []

            # Get layer statistics
            layer_stats = orchestrator.activation_metrics_collector.get_layer_statistics(layer, 300)  # 5 min window
            queue_metrics = orchestrator.activation_metrics_collector.queue_metrics.get(layer, {})

            layer_snapshot = {
                "timestamp": current_time,
                "layer": layer,
                "active_count": layer_stats.get("active_count", 0),
                "throughput": layer_stats.get("throughput_per_minute", 0),
                "success_rate": layer_stats.get("success_rate", 0),
                "avg_processing_time": layer_stats.get("avg_processing_time", 0),
                "queue_depth": (
                    (
                        queue_metrics.available_forward_activations
                        + queue_metrics.available_backward_activations
                        + queue_metrics.processing_activations
                    )
                    if hasattr(queue_metrics, "available_forward_activations")
                    else 0
                ),
                "avg_wait_time": queue_metrics.average_wait_time if hasattr(queue_metrics, "average_wait_time") else 0,
            }
            self.historical_data[layer_key].append(layer_snapshot)

        # Collect miner-specific metrics (sample top performers and problematic miners)
        miner_data = {}
        for hotkey in orchestrator.miner_registry.get_all_miner_data().keys():
            miner_stats = orchestrator.activation_metrics_collector.get_miner_average_times(hotkey, 300)
            if miner_stats.get("sample_count", 0) > 0:
                miner_key = f"miner_{hotkey}"
                if miner_key not in self.historical_data:
                    self.historical_data[miner_key] = []

                miner_snapshot = {
                    "timestamp": current_time,
                    "hotkey": hotkey,
                    "layer": orchestrator.miner_registry.get_miner_data(hotkey).layer,
                    "success_rate": miner_stats.get("success_rate", 0),
                    "avg_processing_time": miner_stats.get("avg_processing_time", 0),
                    "sample_count": miner_stats.get("sample_count", 0),
                }
                self.historical_data[miner_key].append(miner_snapshot)

        # Clean up old data
        self._cleanup_old_data(current_time)
        self.last_collection_time = current_time

    def _cleanup_old_data(self, current_time: float):
        """Remove data older than max_history_hours."""
        cutoff_time = current_time - (self.max_history_hours * 3600)

        for key in self.historical_data:
            self.historical_data[key] = [
                entry for entry in self.historical_data[key] if entry["timestamp"] > cutoff_time
            ]

    def get_time_series_data(
        self,
        metric_type: str,
        time_window_hours: int,
        granularity_minutes: int,
        layer: Optional[int] = None,
        miner_hotkey: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get time series data for a specific metric."""
        current_time = time.time()
        start_time = current_time - (time_window_hours * 3600)

        # Determine data source
        if miner_hotkey:
            data_key = f"miner_{miner_hotkey}"
        elif layer is not None:
            data_key = f"layer_{layer}"
        else:
            data_key = "system"

        if data_key not in self.historical_data:
            return {"timestamps": [], "values": [], "metric_type": metric_type}

        # Filter data by time window
        filtered_data = [entry for entry in self.historical_data[data_key] if entry["timestamp"] >= start_time]

        # Aggregate by granularity
        bucket_size = granularity_minutes * 60
        buckets = {}

        for entry in filtered_data:
            bucket_key = int(entry["timestamp"] // bucket_size) * bucket_size
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(entry)

        # Calculate aggregated values
        timestamps = []
        values = []

        for bucket_time in sorted(buckets.keys()):
            bucket_data = buckets[bucket_time]
            if not bucket_data:
                continue

            timestamps.append(bucket_time)

            # Calculate aggregated value based on metric type
            if metric_type in ["throughput", "avg_processing_time", "avg_wait_time", "queue_depth"]:
                # Average for these metrics
                values.append(sum(entry.get(metric_type, 0) for entry in bucket_data) / len(bucket_data))
            elif metric_type in ["success_rate"]:
                # Weighted average for success rate
                total_samples = sum(entry.get("sample_count", 1) for entry in bucket_data)
                if total_samples > 0:
                    weighted_sum = sum(
                        entry.get(metric_type, 0) * entry.get("sample_count", 1) for entry in bucket_data
                    )
                    values.append(weighted_sum / total_samples)
                else:
                    values.append(0)
            elif metric_type in ["total_active_activations", "active_count"]:
                # Maximum for count metrics (peak usage)
                values.append(max(entry.get(metric_type, 0) for entry in bucket_data))
            else:
                # Default to average
                values.append(sum(entry.get(metric_type, 0) for entry in bucket_data) / len(bucket_data))

        return {
            "timestamps": timestamps,
            "values": values,
            "metric_type": metric_type,
            "data_source": data_key,
            "granularity_minutes": granularity_minutes,
            "time_window_hours": time_window_hours,
        }

    def get_layer_heatmap_data(self, time_window_hours: int, time_bucket_minutes: int) -> Dict[str, Any]:
        """Generate heatmap data for layer performance over time."""
        current_time = time.time()
        start_time = current_time - (time_window_hours * 3600)
        bucket_size = time_bucket_minutes * 60

        # Initialize data structures
        layer_data = {}
        time_buckets = set()

        # Collect data for each layer
        for layer in range(settings.N_LAYERS):
            layer_key = f"layer_{layer}"
            if layer_key not in self.historical_data:
                continue

            layer_data[layer] = {}

            # Filter and bucket the data
            filtered_data = [entry for entry in self.historical_data[layer_key] if entry["timestamp"] >= start_time]

            buckets = {}
            for entry in filtered_data:
                bucket_key = int(entry["timestamp"] // bucket_size) * bucket_size
                time_buckets.add(bucket_key)
                if bucket_key not in buckets:
                    buckets[bucket_key] = []
                buckets[bucket_key].append(entry)

            # Calculate performance score for each time bucket
            for bucket_time, bucket_entries in buckets.items():
                if not bucket_entries:
                    continue

                # Calculate composite performance score (0-100)
                avg_success_rate = sum(entry.get("success_rate", 0) for entry in bucket_entries) / len(bucket_entries)
                avg_throughput = sum(entry.get("throughput", 0) for entry in bucket_entries) / len(bucket_entries)
                avg_processing_time = sum(entry.get("avg_processing_time", 0) for entry in bucket_entries) / len(
                    bucket_entries
                )

                # Normalize and combine metrics (higher is better)
                success_score = avg_success_rate * 100  # Already 0-1
                throughput_score = min(avg_throughput * 10, 100)  # Scale throughput
                time_score = max(100 - (avg_processing_time * 100), 0)  # Lower time is better

                performance_score = success_score * 0.5 + throughput_score * 0.3 + time_score * 0.2
                layer_data[layer][bucket_time] = {
                    "performance_score": performance_score,
                    "success_rate": avg_success_rate,
                    "throughput": avg_throughput,
                    "processing_time": avg_processing_time,
                    "active_count": max(entry.get("active_count", 0) for entry in bucket_entries),
                }

        # Convert to matrix format for heatmap
        sorted_times = sorted(time_buckets)
        layers = list(range(settings.N_LAYERS))

        performance_matrix = []
        success_matrix = []
        throughput_matrix = []

        for layer in layers:
            performance_row = []
            success_row = []
            throughput_row = []

            for time_bucket in sorted_times:
                if layer in layer_data and time_bucket in layer_data[layer]:
                    data = layer_data[layer][time_bucket]
                    performance_row.append(data["performance_score"])
                    success_row.append(data["success_rate"] * 100)
                    throughput_row.append(data["throughput"])
                else:
                    performance_row.append(0)
                    success_row.append(0)
                    throughput_row.append(0)

            performance_matrix.append(performance_row)
            success_matrix.append(success_row)
            throughput_matrix.append(throughput_row)

        return {
            "performance_matrix": performance_matrix,
            "success_matrix": success_matrix,
            "throughput_matrix": throughput_matrix,
            "time_labels": [datetime.fromtimestamp(t).strftime("%H:%M") for t in sorted_times],
            "layer_labels": [f"Layer {i}" for i in layers],
            "time_buckets": sorted_times,
            "bucket_size_minutes": time_bucket_minutes,
        }


class WeightMergingEvent(BaseModel):
    """Event for weight merging lifecycle tracking."""

    event_type: str  # "session_started", "weight_received", "partitions_assigned", etc.
    timestamp: float
    layer: int
    session_id: str
    miner_hotkey: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class WeightMergingSession(BaseModel):
    """Tracking for a complete weight merging session."""

    session_id: str
    layer: int
    started_at: float
    completed_at: Optional[float] = None
    target_miners: List[str] = Field(default_factory=list)  # Expected participants
    weights_received: Dict[str, float] = Field(default_factory=dict)  # hotkey -> timestamp
    partitions_completed: Dict[str, float] = Field(default_factory=dict)  # hotkey -> timestamp
    status: str = "started"  # started, weights_collecting, partitions_merging, completed, failed
    events: List[WeightMergingEvent] = Field(default_factory=list)

    def _record_event(
        self, event_type: str, miner_hotkey: Optional[str] = None, additional_data: Optional[Dict[str, Any]] = None
    ):
        """Record an event in the merging session."""
        event = WeightMergingEvent(
            event_type=event_type,
            timestamp=time.time(),
            layer=self.layer,
            session_id=self.session_id,
            miner_hotkey=miner_hotkey,
            additional_data=additional_data,
        )
        self.events.append(event)

    def record_weight_received(self, miner_hotkey: str):
        """Record when a miner uploads weights."""
        self.weights_received[miner_hotkey] = time.time()
        self._record_event("weight_received", miner_hotkey)

    def record_partition_completed(self, miner_hotkey: str):
        """Record when a miner completes partition merging."""
        self.partitions_completed[miner_hotkey] = time.time()
        self._record_event("partition_completed", miner_hotkey)

    def update_status(self, new_status: str, additional_data: Optional[Dict[str, Any]] = None):
        """Update session status and record event."""
        old_status = self.status
        self.status = new_status
        self._record_event(
            f"status_changed_to_{new_status}", additional_data={"old_status": old_status, **(additional_data or {})}
        )

        if new_status == "completed":
            self.completed_at = time.time()

    def get_session_duration(self) -> Optional[float]:
        """Get total session duration if completed."""
        if self.completed_at:
            return self.completed_at - self.started_at
        return None

    def get_weights_collection_duration(self) -> Optional[float]:
        """Get duration of weights collection phase."""
        if not self.weights_received:
            return None
        last_weight_time = max(self.weights_received.values())
        return last_weight_time - self.started_at

    def get_partition_merging_duration(self) -> Optional[float]:
        """Get duration of partition merging phase."""
        if not self.weights_received or not self.partitions_completed:
            return None
        first_partition_start = min(self.weights_received.values())
        last_partition_completion = max(self.partitions_completed.values())
        return last_partition_completion - first_partition_start

    def get_participation_rate(self) -> float:
        """Get participation rate (0.0 to 1.0)."""
        if not self.target_miners:
            return 0.0
        participated = len(set(self.weights_received.keys()).union(set(self.partitions_completed.keys())))
        return participated / len(self.target_miners)


class WeightMergingMetricsCollector(BaseModel):
    """Collector for managing weight merging metrics."""

    active_sessions: Dict[str, WeightMergingSession] = Field(default_factory=dict)
    completed_sessions: List[WeightMergingSession] = Field(default_factory=list)
    max_history: int = Field(default=50)

    def start_merge_session(self, layer: int, target_miners: List[str]) -> str:
        """Start a new merge session and return session ID."""
        session_id = f"merge_{layer}_{int(time.time())}_{len(self.active_sessions)}"

        session = WeightMergingSession(
            session_id=session_id, layer=layer, started_at=time.time(), target_miners=target_miners.copy()
        )
        session._record_event(
            "session_started",
            additional_data={"target_miners_count": len(target_miners), "target_miners": target_miners},
        )

        self.active_sessions[session_id] = session
        logger.debug(f"Started merge session {session_id} for layer {layer} with {len(target_miners)} target miners")
        return session_id

    def record_weight_upload(self, layer: int, miner_hotkey: str):
        """Record when a miner uploads weights."""
        # Find active session for this layer
        session = self._get_active_session_for_layer(layer)
        if session:
            session.record_weight_received(miner_hotkey)
            logger.debug(f"Recorded weight upload from {miner_hotkey[:8]}... in session {session.session_id}")
        else:
            logger.warning(
                f"No active merge session found for layer {layer} when recording weight upload from {miner_hotkey[:8]}..."
            )

    def record_partition_completion(self, layer: int, miner_hotkey: str):
        """Record when a miner completes partition merging."""
        session = self._get_active_session_for_layer(layer)
        if session:
            session.record_partition_completed(miner_hotkey)
            logger.debug(f"Recorded partition completion from {miner_hotkey[:8]}... in session {session.session_id}")
        else:
            logger.warning(
                f"No active merge session found for layer {layer} when recording partition completion from {miner_hotkey[:8]}..."
            )

    def update_session_status(self, layer: int, new_status: str, additional_data: Optional[Dict[str, Any]] = None):
        """Update the status of active session for a layer."""
        session = self._get_active_session_for_layer(layer)
        if session:
            session.update_status(new_status, additional_data)
            logger.debug(f"Updated session {session.session_id} status to {new_status}")

            # Move to completed if session is done
            if new_status in ["completed", "failed"]:
                self._complete_session(session.session_id)
        else:
            logger.warning(f"No active merge session found for layer {layer} when updating status to {new_status}")

    def _get_active_session_for_layer(self, layer: int) -> Optional[WeightMergingSession]:
        """Get the active merge session for a specific layer."""
        for session in self.active_sessions.values():
            if session.layer == layer:
                return session
        return None

    def _complete_session(self, session_id: str):
        """Move session from active to completed."""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            self.completed_sessions.append(session)
            del self.active_sessions[session_id]

            # Maintain history limit
            if len(self.completed_sessions) > self.max_history:
                self.completed_sessions = self.completed_sessions[-self.max_history :]

            duration = session.get_session_duration()
            logger.info(
                f"Completed merge session {session_id} for layer {session.layer} in {duration:.2f}s"
                if duration
                else f"Completed merge session {session_id}"
            )

    def get_merge_statistics(
        self, layer: Optional[int] = None, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get merge statistics for analysis."""
        current_time = time.time()

        # Filter sessions by layer and time window
        relevant_sessions = []
        for session in self.completed_sessions:
            if layer is not None and session.layer != layer:
                continue
            if time_window_seconds is not None and (current_time - session.started_at) > time_window_seconds:
                continue
            relevant_sessions.append(session)

        if not relevant_sessions:
            return {}

        # Calculate statistics
        total_sessions = len(relevant_sessions)
        successful_sessions = sum(1 for s in relevant_sessions if s.status == "completed")

        durations = [s.get_session_duration() for s in relevant_sessions if s.get_session_duration()]
        weights_durations = [
            s.get_weights_collection_duration() for s in relevant_sessions if s.get_weights_collection_duration()
        ]
        partition_durations = [
            s.get_partition_merging_duration() for s in relevant_sessions if s.get_partition_merging_duration()
        ]
        participation_rates = [s.get_participation_rate() for s in relevant_sessions]

        return {
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": successful_sessions / total_sessions if total_sessions > 0 else 0,
            "avg_session_duration": sum(durations) / len(durations) if durations else 0,
            "avg_weights_collection_duration": (
                sum(weights_durations) / len(weights_durations) if weights_durations else 0
            ),
            "avg_partition_merging_duration": (
                sum(partition_durations) / len(partition_durations) if partition_durations else 0
            ),
            "avg_participation_rate": sum(participation_rates) / len(participation_rates) if participation_rates else 0,
            "sample_count": total_sessions,
        }

    def get_miner_merge_performance(
        self, miner_hotkey: str, time_window_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get merge performance metrics for a specific miner."""
        current_time = time.time()

        # Find sessions where this miner participated
        participated_sessions = []
        for session in self.completed_sessions:
            if time_window_seconds is not None and (current_time - session.started_at) > time_window_seconds:
                continue
            if miner_hotkey in session.weights_received or miner_hotkey in session.partitions_completed:
                participated_sessions.append(session)

        if not participated_sessions:
            return {"participation_count": 0}

        # Calculate miner-specific metrics
        weight_upload_times = []
        partition_completion_times = []
        participated_in_weights = 0
        participated_in_partitions = 0

        for session in participated_sessions:
            if miner_hotkey in session.weights_received:
                participated_in_weights += 1
                weight_time = session.weights_received[miner_hotkey] - session.started_at
                weight_upload_times.append(weight_time)

            if miner_hotkey in session.partitions_completed:
                participated_in_partitions += 1
                # Time from when weights collection finished to partition completion
                if session.weights_received:
                    last_weight_time = max(session.weights_received.values())
                    partition_time = session.partitions_completed[miner_hotkey] - last_weight_time
                    partition_completion_times.append(partition_time)

        return {
            "participation_count": len(participated_sessions),
            "weight_participation_rate": participated_in_weights / len(participated_sessions),
            "partition_participation_rate": participated_in_partitions / len(participated_sessions),
            "avg_weight_upload_time": sum(weight_upload_times) / len(weight_upload_times) if weight_upload_times else 0,
            "avg_partition_completion_time": (
                sum(partition_completion_times) / len(partition_completion_times) if partition_completion_times else 0
            ),
            "sample_count": len(participated_sessions),
        }

    def cleanup_stale_sessions(self, max_age_seconds: float = 7200):  # 2 hours
        """Clean up sessions that have been active too long."""
        current_time = time.time()
        stale_session_ids = []

        for session_id, session in self.active_sessions.items():
            if (current_time - session.started_at) > max_age_seconds:
                stale_session_ids.append(session_id)

        for session_id in stale_session_ids:
            logger.warning(f"Cleaning up stale merge session: {session_id}")
            session = self.active_sessions[session_id]
            session.update_status("failed", {"reason": "timeout"})
            self._complete_session(session_id)

        return len(stale_session_ids)

    def update_miner_merge_status(self, miner_hotkey: str, status: str, session_id: str):
        """Update individual miner's merge status within a session.

        Args:
            miner_hotkey: The hotkey of the miner
            status: The new status for the miner
            session_id: The session ID this miner belongs to
        """
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session._record_event(
                f"miner_status_update_{status}", miner_hotkey=miner_hotkey, additional_data={"new_status": status}
            )
            logger.debug(f"Updated miner {miner_hotkey[:8]}... status to {status} in session {session_id}")
        else:
            logger.warning(
                f"Session {session_id} not found when updating miner {miner_hotkey[:8]}... status to {status}"
            )

    def get_miners_by_status(self, layer: Optional[int] = None) -> Dict[str, List[str]]:
        """Get miners grouped by their current merge status.

        Args:
            layer: Optional layer filter

        Returns:
            Dict mapping status to list of miner hotkeys
        """
        status_groups = {}

        # Get status from active sessions
        for session in self.active_sessions.values():
            if layer is not None and session.layer != layer:
                continue

            # Determine miner statuses based on session state and participation
            for miner_hotkey in session.target_miners:
                if miner_hotkey in session.partitions_completed:
                    status = "merge_complete"
                elif miner_hotkey in session.weights_received:
                    if session.status == "partitions_merging":
                        status = "merging_partitions"
                    else:
                        status = "weights_uploaded"
                elif session.status == "weights_collecting" or session.status == "started":
                    status = "waiting_for_merge"
                else:
                    status = "idle"

                if status not in status_groups:
                    status_groups[status] = []
                status_groups[status].append(miner_hotkey)

        return status_groups

    def get_session_for_miner(self, miner_hotkey: str) -> Optional[WeightMergingSession]:
        """Get the active session that a miner is participating in.

        Args:
            miner_hotkey: The hotkey of the miner

        Returns:
            The active session the miner is in, or None
        """
        for session in self.active_sessions.values():
            if miner_hotkey in session.target_miners:
                return session
        return None

    def get_miner_current_status(self, miner_hotkey: str) -> str:
        """Get the current merge status of a specific miner.

        Args:
            miner_hotkey: The hotkey of the miner

        Returns:
            Current status string
        """
        session = self.get_session_for_miner(miner_hotkey)
        if not session:
            return "idle"

        # Determine status based on session state and miner participation
        if miner_hotkey in session.partitions_completed:
            return "merge_complete"
        elif miner_hotkey in session.weights_received:
            if session.status == "partitions_merging":
                return "merging_partitions"
            else:
                return "weights_uploaded"
        elif session.status in ["weights_collecting", "started"]:
            return "waiting_for_merge"
        else:
            return "idle"

    def get_miner_progress(self, miner_hotkey: str) -> float:
        """Get the progress of a miner's current merge operation.

        Args:
            miner_hotkey: The hotkey of the miner

        Returns:
            Progress as a float between 0.0 and 1.0
        """
        session = self.get_session_for_miner(miner_hotkey)
        if not session:
            return 0.0

        current_time = time.time()

        # If miner has completed partitions, they're done
        if miner_hotkey in session.partitions_completed:
            return 1.0

        # If miner has uploaded weights and we're in partition merging phase
        if miner_hotkey in session.weights_received and session.status == "partitions_merging":
            # Estimate progress based on time since they uploaded weights
            weight_upload_time = session.weights_received[miner_hotkey]
            elapsed_since_upload = current_time - weight_upload_time
            # Assume partition merging takes about 2 minutes, so progress is based on that
            progress = min(elapsed_since_upload / 120.0, 0.9)  # Max 90% until actually complete
            return 0.5 + (progress * 0.5)  # Start at 50% after weight upload

        # If miner is in weight uploading phase
        if session.status in ["weights_collecting", "started"]:
            # Check if they've already uploaded
            if miner_hotkey in session.weights_received:
                return 0.5  # 50% complete after weight upload
            else:
                # Estimate based on session duration (assume weight upload phase lasts ~5 minutes)
                session_duration = current_time - session.started_at
                progress = min(session_duration / 300.0, 0.4)  # Max 40% until weight upload
                return progress

        return 0.0
