"""
Unit tests for topocore.pointcloud.chunk.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk


def test_create_empty_chunk() -> None:
    """An empty chunk should contain no points."""

    chunk = Chunk()

    assert chunk.size == 0
    assert len(chunk) == 0
    assert chunk.is_empty
    assert chunk.attributes == frozenset()


def test_create_chunk_with_attributes() -> None:
    """Attributes should be allocated during construction."""

    chunk = Chunk(
        size=10,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
        ],
    )

    assert chunk.size == 10

    assert PointAttribute.X in chunk
    assert PointAttribute.Y in chunk
    assert PointAttribute.Z in chunk


def test_get_attribute() -> None:
    """Allocated attributes should be accessible."""

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.X],
    )

    assert chunk[PointAttribute.X].shape == (5,)


def test_add_attribute() -> None:
    """Attributes can be added."""

    chunk = Chunk(size=3)

    chunk.add_attribute(PointAttribute.X)

    assert PointAttribute.X in chunk


def test_remove_attribute() -> None:
    """Attributes can be removed."""

    chunk = Chunk(
        size=3,
        attributes=[PointAttribute.X],
    )

    chunk.remove_attribute(PointAttribute.X)

    assert PointAttribute.X not in chunk


def test_resize() -> None:
    """Resize should update the number of stored points."""

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.X],
    )

    chunk.resize(8)

    assert chunk.size == 8


def test_clone() -> None:
    """Clone should perform a deep copy."""

    chunk = Chunk(
        size=3,
        attributes=[PointAttribute.X],
    )

    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0]

    cloned = chunk.clone()

    cloned[PointAttribute.X][0] = 100.0

    assert chunk[PointAttribute.X][0] == 1.0
    assert cloned[PointAttribute.X][0] == 100.0


def test_clear() -> None:
    """Clear should remove all data."""

    chunk = Chunk(
        size=10,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
        ],
    )

    chunk.clear()

    assert chunk.size == 0
    assert chunk.attributes == frozenset()
    assert chunk.is_empty


def test_missing_attribute() -> None:
    """Missing attributes should raise KeyError."""

    chunk = Chunk()

    with pytest.raises(KeyError):
        _ = chunk[PointAttribute.X]


def test_repr() -> None:
    """repr() should contain useful information."""

    chunk = Chunk(
        size=25,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
        ],
    )

    text = repr(chunk)

    assert "Chunk" in text
    assert "size=25" in text