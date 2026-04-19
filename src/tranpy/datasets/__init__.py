"""
TranPy Datasets Module

Provides sklearn-style dataset loaders for power system stability data.
"""

from .loaders import (
    load_newengland,
    load_ieee9bus,
    load_dataset,
    list_available_datasets,
    get_data_path
)
from .base import StabilityDataset
from .temporal_loaders import (
    load_temporal_dataset,
    load_newengland_events,
    load_ieee9bus_events,
    load_event_trajectory,
    list_available_events,
    extract_stability_label,
    get_generator_columns,
    get_bus_columns,
    build_drive_index,
    TemporalEventData,
    TemporalDataset
)

__all__ = [
    # Snapshot loaders (existing)
    'load_newengland',
    'load_ieee9bus',
    'load_dataset',
    'list_available_datasets',
    'get_data_path',
    'StabilityDataset',
    # Temporal loaders (new)
    'load_temporal_dataset',
    'load_newengland_events',
    'load_ieee9bus_events',
    'load_event_trajectory',
    'list_available_events',
    'extract_stability_label',
    'get_generator_columns',
    'get_bus_columns',
    'build_drive_index',
    'TemporalEventData',
    'TemporalDataset'
]
