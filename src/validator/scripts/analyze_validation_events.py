#!/usr/bin/env python3
"""
Analysis script for validation events logged by the gradient validator.
Usage: python analyze_validation_events.py [validation_events_file.jsonl]
"""

import json
import argparse
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Union
import statistics

try:
    from common.models.storage_models import ValidationEvent

    PYDANTIC_AVAILABLE = True
except ImportError:
    print("Warning: Could not import ValidationEvent from storage.serializers. Using dict-based analysis.")
    PYDANTIC_AVAILABLE = False


def load_validation_events(filepath: str, use_pydantic: bool = True) -> List[Union[ValidationEvent, Dict[str, Any]]]:
    """Load validation events from a JSONL file."""
    events = []
    try:
        with open(filepath, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        event_dict = json.loads(line)
                        if use_pydantic and PYDANTIC_AVAILABLE:
                            # Try to create Pydantic model for type safety
                            try:
                                event = ValidationEvent(**event_dict)
                                events.append(event)
                            except Exception as pydantic_error:
                                print(
                                    f"Warning: Could not create ValidationEvent from line {line_num}: {pydantic_error}"
                                )
                                events.append(event_dict)
                        else:
                            events.append(event_dict)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON on line {line_num}: {e}")
                        continue
        print(f"Loaded {len(events)} validation events from {filepath}")
        return events
    except Exception as e:
        print(f"Error loading events from {filepath}: {e}")
        return []


def get_event_attr(event: Union[ValidationEvent, Dict[str, Any]], attr: str, default: Any = None) -> Any:
    """Get attribute from either Pydantic model or dict."""
    if PYDANTIC_AVAILABLE and isinstance(event, ValidationEvent):
        return getattr(event, attr, default)
    else:
        return event.get(attr, default)


def analyze_events(events: List[Union[ValidationEvent, Dict[str, Any]]]) -> None:
    """Analyze and print statistics about validation events."""
    if not events:
        print("No events to analyze")
        return

    print("\n" + "=" * 80)
    print("VALIDATION EVENTS ANALYSIS")
    print("=" * 80)

    # Basic statistics
    total_events = len(events)
    successful_events = sum(1 for event in events if get_event_attr(event, "success", False))
    failed_events = total_events - successful_events
    success_rate = (successful_events / total_events) * 100 if total_events > 0 else 0

    print("\nOVERALL STATISTICS:")
    print(f"  Total events: {total_events}")
    print(f"  Successful: {successful_events} ({success_rate:.1f}%)")
    print(f"  Failed: {failed_events} ({100 - success_rate:.1f}%)")

    # Event types breakdown
    event_types = Counter(get_event_attr(event, "event_type", "unknown") for event in events)
    print("\nEVENT TYPES:")
    for event_type, count in event_types.items():
        success_count = sum(
            1
            for event in events
            if get_event_attr(event, "event_type") == event_type and get_event_attr(event, "success", False)
        )
        success_rate = (success_count / count) * 100 if count > 0 else 0
        print(f"  {event_type}: {count} events ({success_count} successful, {success_rate:.1f}%)")

    # Miner breakdown
    miners = set(
        get_event_attr(event, "miner_hotkey") for event in events if get_event_attr(event, "miner_hotkey") is not None
    )
    if miners:
        print("\nMINER STATISTICS:")
        for miner_hotkey in sorted(miners):
            miner_events = [e for e in events if get_event_attr(e, "miner_hotkey") == miner_hotkey]
            miner_success = sum(1 for e in miner_events if get_event_attr(e, "success", False))
            miner_success_rate = (miner_success / len(miner_events)) * 100 if miner_events else 0
            print(
                f"  Miner {miner_hotkey[:8] if miner_hotkey else 'Unknown'}...: "
                f"{len(miner_events)} events, {miner_success_rate:.1f}% success rate"
            )

    # Direction breakdown for activation events
    activation_events = [e for e in events if get_event_attr(e, "event_type") == "activation_validation"]
    if activation_events:
        directions = Counter(get_event_attr(event, "direction", "unknown") for event in activation_events)
        print("\nACTIVATION VALIDATION BY DIRECTION:")
        for direction, count in directions.items():
            direction_success = sum(
                1
                for event in activation_events
                if get_event_attr(event, "direction") == direction and get_event_attr(event, "success", False)
            )
            success_rate = (direction_success / count) * 100 if count > 0 else 0
            print(f"  {direction}: {count} events ({direction_success} successful, {success_rate:.1f}%)")

    # Layer breakdown for activation events
    if activation_events:
        layers = Counter(
            get_event_attr(event, "layer") for event in activation_events if get_event_attr(event, "layer") is not None
        )
        print("\nACTIVATION VALIDATION BY LAYER:")
        for layer, count in sorted(layers.items()):
            layer_success = sum(
                1
                for event in activation_events
                if get_event_attr(event, "layer") == layer and get_event_attr(event, "success", False)
            )
            success_rate = (layer_success / count) * 100 if count > 0 else 0
            print(f"  Layer {layer}: {count} events ({layer_success} successful, {success_rate:.1f}%)")

    # Failure reasons
    failed_events_list = [e for e in events if not get_event_attr(e, "success", False)]
    if failed_events_list:
        failure_reasons = Counter(get_event_attr(event, "reason", "unknown") for event in failed_events_list)
        print("\nFAILURE REASONS:")
        for reason, count in failure_reasons.most_common():
            percentage = (count / failed_events) * 100 if failed_events > 0 else 0
            print(f"  {reason}: {count} ({percentage:.1f}% of failures)")

    # Score statistics
    scores = [get_event_attr(event, "score") for event in events if get_event_attr(event, "score") is not None]
    if scores:
        print("\nSCORE STATISTICS:")
        print(f"  Mean score: {statistics.mean(scores):.4f}")
        print(f"  Median score: {statistics.median(scores):.4f}")
        print(f"  Min score: {min(scores):.4f}")
        print(f"  Max score: {max(scores):.4f}")
        if len(scores) > 1:
            print(f"  Standard deviation: {statistics.stdev(scores):.4f}")

    # Time range
    timestamps = [
        get_event_attr(event, "timestamp") for event in events if get_event_attr(event, "timestamp") is not None
    ]
    if timestamps:
        min_time = min(timestamps)
        max_time = max(timestamps)
        duration = max_time - min_time
        print("\nTIME RANGE:")
        print(f"  Start: {datetime.fromtimestamp(min_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End: {datetime.fromtimestamp(max_time).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")


def main():
    parser = argparse.ArgumentParser(description="Analyze validation events from gradient validator")
    parser.add_argument("file", nargs="?", help="Path to validation events JSONL file")
    parser.add_argument(
        "--directory",
        "-d",
        default="validation_events",
        help="Directory to search for validation event files (default: validation_events)",
    )
    parser.add_argument("--no-pydantic", action="store_true", help="Disable Pydantic model validation (use raw dicts)")
    args = parser.parse_args()

    use_pydantic = not args.no_pydantic

    if args.file:
        # Analyze specific file
        events = load_validation_events(args.file, use_pydantic=use_pydantic)
        analyze_events(events)
    else:
        # Find and analyze all files in directory
        if not os.path.exists(args.directory):
            print(f"Directory {args.directory} does not exist")
            return

        files = [f for f in os.listdir(args.directory) if f.endswith(".jsonl")]
        if not files:
            print(f"No .jsonl files found in {args.directory}")
            return

        print(f"Found {len(files)} validation event files:")
        for i, filename in enumerate(sorted(files), 1):
            print(f"  {i}. {filename}")

        # Analyze each file separately
        for filename in sorted(files):
            filepath = os.path.join(args.directory, filename)
            print(f"\n{'=' * 80}")
            print(f"ANALYZING FILE: {filename}")
            print(f"{'=' * 80}")
            events = load_validation_events(filepath, use_pydantic=use_pydantic)
            analyze_events(events)

        # Analyze all files combined
        all_events = []
        for filename in files:
            filepath = os.path.join(args.directory, filename)
            all_events.extend(load_validation_events(filepath, use_pydantic=use_pydantic))

        if len(files) > 1:
            print(f"\n{'=' * 80}")
            print(f"COMBINED ANALYSIS OF ALL {len(files)} FILES")
            print(f"{'=' * 80}")
            analyze_events(all_events)


if __name__ == "__main__":
    main()
