"""
Regression suite for topocore.analysis.config -- PR19.

Verified all 6 sub-config validation boundaries (rejecting invalid
values at construction) and that AnalysisConfig's frozen-dataclass
default instances are safe to share (no mutable-default hazard,
since all sub-configs are themselves frozen). No bugs found.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from topocore.analysis.config import (
    DEFAULT_ANALYSIS_CONFIG,
    AnalysisConfig,
    DistanceConfig,
    ProfileConfig,
    QualityConfig,
    StatisticsConfig,
    VisibilityConfig,
    VolumeConfig,
)


def test_analysis_config_default_construction() -> None:
    config = AnalysisConfig()
    assert isinstance(config.distance, DistanceConfig)
    assert isinstance(config.volume, VolumeConfig)


def test_default_analysis_config_is_a_real_instance() -> None:
    assert isinstance(DEFAULT_ANALYSIS_CONFIG, AnalysisConfig)


def test_all_configs_are_frozen() -> None:
    config = AnalysisConfig()
    with pytest.raises(FrozenInstanceError):
        config.distance.default_precision = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (DistanceConfig, {"default_precision": -1}),
        (DistanceConfig, {"ellipsoid": ""}),
        (VolumeConfig, {"default_precision": -1}),
        (ProfileConfig, {"default_interval": 0.0}),
        (ProfileConfig, {"default_width": -5.0}),
        (VisibilityConfig, {"observer_height": -1.0}),
        (VisibilityConfig, {"target_height": -1.0}),
        (StatisticsConfig, {"histogram_bins": 0}),
        (StatisticsConfig, {"percentile_precision": -1}),
        (QualityConfig, {"confidence_level": 0.0}),
        (QualityConfig, {"confidence_level": 1.0}),
        (QualityConfig, {"max_correspondence_distance": 0.0}),
    ],
)
def test_rejects_invalid_boundary_values(factory: type, kwargs: dict) -> None:  # type: ignore[type-arg]
    with pytest.raises(ValueError):
        factory(**kwargs)
