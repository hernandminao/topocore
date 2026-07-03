"""
Unit tests for topocore.pointcloud.numpy_memory.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.numpy_memory import NumpyMemory


def test_create_empty_memory() -> None:
    """An empty memory should contain no points or attributes."""

    memory = NumpyMemory()

    assert memory.size == 0
    assert len(memory) == 0
    assert memory.attributes == frozenset()


def test_allocate_attributes() -> None:
    """Requested attributes should be allocated."""

    memory = NumpyMemory(
        size=10,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
        ],
    )

    assert memory.size == 10

    assert PointAttribute.X in memory
    assert PointAttribute.Y in memory
    assert PointAttribute.Z in memory


def test_attribute_shapes() -> None:
    """Allocated arrays must have the expected shape."""

    memory = NumpyMemory(
        size=5,
        attributes=[
            PointAttribute.X,
            PointAttribute.INTENSITY,
        ],
    )

    assert memory[PointAttribute.X].shape == (5,)
    assert memory[PointAttribute.INTENSITY].shape == (5,)


def test_add_attribute() -> None:
    """Attributes can be added after construction."""

    memory = NumpyMemory(size=8)

    memory.add_attribute(PointAttribute.X)

    assert PointAttribute.X in memory
    assert memory[PointAttribute.X].shape == (8,)


def test_remove_attribute() -> None:
    """Removing an attribute should make it unavailable."""

    memory = NumpyMemory(
        size=5,
        attributes=[PointAttribute.X],
    )

    memory.remove_attribute(PointAttribute.X)

    assert PointAttribute.X not in memory


def test_resize_grow() -> None:
    """Growing memory should preserve existing values."""

    memory = NumpyMemory(
        size=3,
        attributes=[PointAttribute.X],
    )

    memory[PointAttribute.X][:] = [1.0, 2.0, 3.0]

    memory.resize(6)

    assert memory.size == 6

    np.testing.assert_array_equal(
        memory[PointAttribute.X][:3],
        np.array([1.0, 2.0, 3.0]),
    )


def test_resize_shrink() -> None:
    """Shrinking memory should truncate values."""

    memory = NumpyMemory(
        size=5,
        attributes=[PointAttribute.X],
    )

    memory[PointAttribute.X][:] = [1, 2, 3, 4, 5]

    memory.resize(2)

    np.testing.assert_array_equal(
        memory[PointAttribute.X],
        np.array([1, 2]),
    )


def test_clone() -> None:
    """Clone should create a deep copy."""

    memory = NumpyMemory(
        size=3,
        attributes=[PointAttribute.X],
    )

    memory[PointAttribute.X][:] = [1, 2, 3]

    clone = memory.clone()

    clone[PointAttribute.X][0] = 100

    assert memory[PointAttribute.X][0] == 1
    assert clone[PointAttribute.X][0] == 100


def test_clear() -> None:
    """Clear should remove every attribute."""

    memory = NumpyMemory(
        size=5,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
        ],
    )

    memory.clear()

    assert memory.size == 0
    assert memory.attributes == frozenset()


def test_negative_size() -> None:
    """Negative sizes are not allowed."""

    with pytest.raises(ValueError):
        NumpyMemory(size=-1)


def test_invalid_attribute_access() -> None:
    """Accessing a missing attribute should raise KeyError."""

    memory = NumpyMemory()

    with pytest.raises(KeyError):
        _ = memory[PointAttribute.X]