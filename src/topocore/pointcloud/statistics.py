"""
topocore.pointcloud.statistics
==============================

Statistical utilities for point cloud attributes.

This module provides immutable statistical summaries for individual
point cloud attributes.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


@dataclass(frozen=True, slots=True)
class MinMax:
    """
    Minimum and maximum values of a point attribute.
    """

    minimum: float

    maximum: float


@dataclass(frozen=True, slots=True)
class AttributeStatistics:
    """
    Immutable statistical summary of a point attribute.
    """

    minimum: float

    maximum: float

    mean: float

    variance: float

    standard_deviation: float

    count: int


def compute_min_max(
    chunk: Chunk,
    attribute: PointAttribute,
) -> MinMax:
    """
    Compute the minimum and maximum values of an attribute.

    Parameters
    ----------
    chunk
        Source chunk.

    attribute
        Attribute to analyze.

    Returns
    -------
    MinMax
        Minimum and maximum values.

    Raises
    ------
    ValueError
        If the attribute is empty.
    """

    values = chunk[attribute]

    if values.size == 0:
        raise ValueError(
            "Cannot compute min/max for an empty attribute."
        )

    data = values.astype(
        np.float64,
        copy=False,
    ).reshape(-1)

    return MinMax(
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
    )


def compute_statistics(
    chunk: Chunk,
    attribute: PointAttribute,
) -> AttributeStatistics:
    """
    Compute descriptive statistics for a point attribute.

    Parameters
    ----------
    chunk
        Source chunk.

    attribute
        Attribute to analyze.

    Returns
    -------
    AttributeStatistics
        Statistical summary.

    Raises
    ------
    ValueError
        If the attribute is empty.
    """

    values = chunk[attribute]

    if values.size == 0:
        raise ValueError(
            "Cannot compute statistics for an empty attribute."
        )

    data = values.astype(
        np.float64,
        copy=False,
    ).reshape(-1)

    return AttributeStatistics(
        minimum=float(np.min(data)),
        maximum=float(np.max(data)),
        mean=float(np.mean(data)),
        variance=float(np.var(data)),
        standard_deviation=float(np.std(data)),
        count=int(data.size),
    )


__all__ = [
    "AttributeStatistics",
    "MinMax",
    "compute_min_max",
    "compute_statistics",
]