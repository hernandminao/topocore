"""
Unit tests for topocore.pointcloud.statistics.
"""

from __future__ import annotations

import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.statistics import (
    compute_min_max,
    compute_statistics,
)


def test_compute_min_max() -> None:
    """Minimum and maximum should be computed correctly."""

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.Z],
    )

    chunk[PointAttribute.Z][:] = [
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]

    result = compute_min_max(
        chunk,
        PointAttribute.Z,
    )

    assert result.minimum == 10.0
    assert result.maximum == 50.0


def test_compute_statistics() -> None:
    """Statistics should match known values."""

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.Z],
    )

    chunk[PointAttribute.Z][:] = [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]

    stats = compute_statistics(
        chunk,
        PointAttribute.Z,
    )

    assert stats.minimum == 1.0
    assert stats.maximum == 5.0
    assert stats.mean == 3.0
    assert stats.count == 5

    assert stats.variance == pytest.approx(
        2.0,
    )

    assert stats.standard_deviation == pytest.approx(
        2.0**0.5,
    )


def test_single_value_statistics() -> None:
    """Statistics for a single value should be valid."""

    chunk = Chunk(
        size=1,
        attributes=[PointAttribute.Z],
    )

    chunk[PointAttribute.Z][0] = 100.0

    stats = compute_statistics(
        chunk,
        PointAttribute.Z,
    )

    assert stats.minimum == 100.0
    assert stats.maximum == 100.0
    assert stats.mean == 100.0
    assert stats.variance == 0.0
    assert stats.standard_deviation == 0.0
    assert stats.count == 1


def test_empty_attribute() -> None:
    """Empty attributes should raise ValueError."""

    chunk = Chunk(
        size=0,
        attributes=[PointAttribute.Z],
    )

    with pytest.raises(ValueError):
        compute_statistics(
            chunk,
            PointAttribute.Z,
        )


def test_missing_attribute() -> None:
    """Missing attributes should raise KeyError."""

    chunk = Chunk()

    with pytest.raises(KeyError):
        compute_statistics(
            chunk,
            PointAttribute.Z,
        )


def test_negative_values() -> None:
    """Negative values should be handled correctly."""

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.Z],
    )

    chunk[PointAttribute.Z][:] = [
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    ]

    stats = compute_statistics(
        chunk,
        PointAttribute.Z,
    )

    assert stats.minimum == -10.0
    assert stats.maximum == 10.0
    assert stats.mean == 0.0


def test_compute_min_max_empty_attribute() -> None:
    """compute_min_max should reject empty attributes."""

    chunk = Chunk(
        size=0,
        attributes=[PointAttribute.Z],
    )

    with pytest.raises(ValueError):
        compute_min_max(
            chunk,
            PointAttribute.Z,
        )


def test_compute_min_max_missing_attribute() -> None:
    """compute_min_max should raise KeyError for missing attributes."""

    chunk = Chunk()

    with pytest.raises(KeyError):
        compute_min_max(
            chunk,
            PointAttribute.Z,
        )
