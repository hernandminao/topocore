"""
Unit tests for topocore.pointcloud.pointcloud.
"""

from __future__ import annotations

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud


def test_create_empty_pointcloud() -> None:
    """An empty point cloud should contain no chunks."""

    cloud = PointCloud()

    assert cloud.chunk_count == 0
    assert cloud.point_count == 0
    assert cloud.is_empty
    assert len(cloud) == 0
    assert cloud.attributes == frozenset()


def test_add_chunk() -> None:
    """Chunks should be added correctly."""

    cloud = PointCloud()

    chunk = Chunk(
        size=100,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
        ],
    )

    cloud.add_chunk(chunk)

    assert cloud.chunk_count == 1
    assert cloud.point_count == 100
    assert not cloud.is_empty


def test_multiple_chunks() -> None:
    """Point count should be the sum of all chunks."""

    cloud = PointCloud()

    cloud.add_chunk(
        Chunk(
            size=100,
            attributes=[PointAttribute.X],
        )
    )

    cloud.add_chunk(
        Chunk(
            size=250,
            attributes=[PointAttribute.X],
        )
    )

    cloud.add_chunk(
        Chunk(
            size=50,
            attributes=[PointAttribute.X],
        )
    )

    assert cloud.chunk_count == 3
    assert cloud.point_count == 400


def test_attributes_union() -> None:
    """The attribute set should be the union of all chunks."""

    cloud = PointCloud()

    cloud.add_chunk(
        Chunk(
            size=10,
            attributes=[
                PointAttribute.X,
                PointAttribute.Y,
            ],
        )
    )

    cloud.add_chunk(
        Chunk(
            size=10,
            attributes=[
                PointAttribute.Z,
                PointAttribute.INTENSITY,
            ],
        )
    )

    assert cloud.attributes == frozenset(
        {
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.INTENSITY,
        }
    )


def test_get_chunk() -> None:
    """Chunks should be accessible by index."""

    cloud = PointCloud()

    chunk = Chunk(size=25)

    cloud.add_chunk(chunk)

    assert cloud[0] is chunk


def test_remove_chunk() -> None:
    """Removing a chunk should update the cloud."""

    cloud = PointCloud()

    first = Chunk(size=10)
    second = Chunk(size=20)

    cloud.add_chunk(first)
    cloud.add_chunk(second)

    removed = cloud.remove_chunk(0)

    assert removed is first
    assert cloud.chunk_count == 1
    assert cloud.point_count == 20


def test_iteration() -> None:
    """Iteration should return chunks in insertion order."""

    cloud = PointCloud()

    first = Chunk(size=1)
    second = Chunk(size=2)

    cloud.add_chunk(first)
    cloud.add_chunk(second)

    chunks = list(cloud)

    assert chunks == [first, second]


def test_clone() -> None:
    """Clone should perform a deep copy."""

    cloud = PointCloud()

    chunk = Chunk(
        size=5,
        attributes=[PointAttribute.X],
    )

    chunk[PointAttribute.X][:] = [
        1,
        2,
        3,
        4,
        5,
    ]

    cloud.add_chunk(chunk)

    cloned = cloud.clone()

    cloned[0][PointAttribute.X][0] = 100

    assert cloud[0][PointAttribute.X][0] == 1
    assert cloned[0][PointAttribute.X][0] == 100


def test_clear() -> None:
    """Clear should remove every chunk."""

    cloud = PointCloud()

    cloud.add_chunk(Chunk(size=10))
    cloud.add_chunk(Chunk(size=20))

    cloud.clear()

    assert cloud.chunk_count == 0
    assert cloud.point_count == 0
    assert cloud.is_empty


def test_repr() -> None:
    """repr() should contain useful information."""

    cloud = PointCloud()

    cloud.add_chunk(
        Chunk(
            size=50,
            attributes=[PointAttribute.X],
        )
    )

    text = repr(cloud)

    assert "PointCloud" in text
    assert "chunks=1" in text
    assert "points=50" in text
