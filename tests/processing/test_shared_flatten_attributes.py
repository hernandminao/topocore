"""
Coverage audit tests for topocore.processing._shared.flatten_attributes()
and its two direct helper dependencies (_validate_cloud_attributes,
_validate_chunk_attributes), targeting only branches confirmed
reachable via legitimate public API usage.

Audit findings (documented here, not force-tested):

_validate_cloud_attributes' own check (missing X/Y/Z) is confirmed
UNREACHABLE from flatten_attributes()'s calling context: Chunk's own
constructor already rejects a chunk missing X/Y/Z (confirmed
directly: `ValueError: Chunk is missing required attribute(s)`), and
_validate_cloud_attributes is called ONLY from flatten_attributes(),
strictly after flatten_attributes()'s own is_empty check already
guarantees at least one chunk exists. Since PointCloud.attributes is
the union of its chunks' own attribute sets, and every constituent
chunk is guaranteed to already contain X/Y/Z, the union can never
lack them either. Not tested here by design.

_validate_flattened_attributes' two checks (empty dict, mismatched
lengths) are ALSO confirmed unreachable specifically from
flatten_attributes()'s own call path (though they may be reachable
from build_cloud()/build_cloud_from_mask(), which take an externally
supplied `flattened` dict as a parameter -- assessed separately when
auditing those functions): `attributes = frozenset(cloud.attributes)`
is never empty for the same Chunk-contract reason above, and Chunk
itself does not support replacing an attribute's array wholesale
(confirmed directly: `TypeError: 'Chunk' object does not support item
assignment`) -- once constructed with a fixed `size`, every attribute
array within a chunk is guaranteed that same length, so concatenating
across chunks (already validated homogeneous by
_validate_chunk_attributes) always yields equal lengths across every
attribute. Not tested here by design.

What IS tested: the empty-cloud check (line 232, trivially reachable
via a freshly constructed PointCloud with no chunks added), the
heterogeneous-chunk check (line 202, confirmed reachable directly:
PointCloud.add_chunk() does not validate attribute-set consistency
across chunks -- confirmed via direct execution before writing this
test, not assumed), and the happy path across both a single chunk and
multiple chunks (confirmed via direct execution that concatenation
preserves chunk order and produces equal lengths across every
attribute).
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing._shared import flatten_attributes
from topocore.processing.exceptions import ProcessingError


def _chunk(size: int, x: list[float], classification: list[int]) -> Chunk:
    chunk = Chunk(
        size=size,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.CLASSIFICATION,
        ],
    )
    chunk[PointAttribute.X][:] = x
    chunk[PointAttribute.Y][:] = [0.0] * size
    chunk[PointAttribute.Z][:] = [0.0] * size
    chunk[PointAttribute.CLASSIFICATION][:] = classification
    return chunk


# ----------------------------------------------------------------------
# Happy path -- single and multiple chunks, verified against real execution.
# ----------------------------------------------------------------------


def test_single_chunk_flattens_correctly() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk(3, [1.0, 2.0, 3.0], [1, 1, 2]))

    result = flatten_attributes(cloud)

    assert set(result.keys()) == {
        PointAttribute.X,
        PointAttribute.Y,
        PointAttribute.Z,
        PointAttribute.CLASSIFICATION,
    }
    np.testing.assert_array_equal(result[PointAttribute.X], [1.0, 2.0, 3.0])


def test_multiple_chunks_concatenate_in_order() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk(3, [1.0, 2.0, 3.0], [1, 1, 2]))
    cloud.add_chunk(_chunk(2, [4.0, 5.0], [3, 3]))

    result = flatten_attributes(cloud)

    np.testing.assert_array_equal(result[PointAttribute.X], [1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_equal(result[PointAttribute.CLASSIFICATION], [1, 1, 2, 3, 3])


def test_all_attributes_have_equal_length_after_flattening() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk(3, [1.0, 2.0, 3.0], [1, 1, 2]))
    cloud.add_chunk(_chunk(2, [4.0, 5.0], [3, 3]))

    result = flatten_attributes(cloud)

    lengths = {len(values) for values in result.values()}
    assert lengths == {5}


# ----------------------------------------------------------------------
# Empty cloud -- trivially reachable.
# ----------------------------------------------------------------------


def test_empty_cloud_raises_processing_error() -> None:
    with pytest.raises(ProcessingError, match="empty"):
        flatten_attributes(PointCloud())


# ----------------------------------------------------------------------
# Heterogeneous chunks -- confirmed reachable: add_chunk() does not
# validate attribute-set consistency across chunks.
# ----------------------------------------------------------------------


def test_heterogeneous_chunks_raise_processing_error() -> None:
    cloud = PointCloud()
    cloud.add_chunk(_chunk(3, [1.0, 2.0, 3.0], [1, 1, 2]))  # has CLASSIFICATION

    xyz_only = Chunk(size=2, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    xyz_only[PointAttribute.X][:] = [4.0, 5.0]
    xyz_only[PointAttribute.Y][:] = [0.0, 0.0]
    xyz_only[PointAttribute.Z][:] = [0.0, 0.0]
    cloud.add_chunk(xyz_only)  # missing CLASSIFICATION -- heterogeneous with the first chunk

    with pytest.raises(ProcessingError, match="same attribute set"):
        flatten_attributes(cloud)
