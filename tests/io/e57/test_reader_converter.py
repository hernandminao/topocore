"""
Regression suite for topocore.io.e57.reader.E57Reader and
.converter.E57Converter -- PR19.

Verified against real E57 files written with pye57 itself (not
mocks). Includes two real bugs found and fixed in this session:

1. E57Reader called pye57's read_scan() with its own bare defaults
   (intensity=False, colors=False), silently dropping BOTH intensity
   and RGB color from any E57 file that genuinely contains them,
   despite E57Converter already being fully prepared to extract
   both. Confirmed directly with a real E57 file containing
   intensity and colorRed/colorGreen/colorBlue: the resulting Chunk
   had only X/Y/Z.

2. Even after requesting intensity, E57's raw intensity field (a
   common [0,1]-normalized float, per real-world E57 files -- pye57
   itself applies no normalization, confirmed by reading its own
   source) was being assigned directly into PointAttribute.INTENSITY's
   unified uint16 column, silently truncating every value to 0.
   Fixed with data-driven min-max normalization (scales the ACTUAL
   observed range in each scan to fill [0, 65535], not assuming any
   fixed a-priori range) -- confirmed directly: intensity
   [0.1, 0.5, 0.9] correctly became [0, 32767, 65535].

Also confirmed (no bug): pye57's own read_scan() defaults to
transform=True, meaning per-scan pose (rotation + translation) IS
applied by default, converting local scan coordinates into the
global E57 file coordinate system correctly, with no explicit
action needed from TopoCore's own code.
"""

from __future__ import annotations

import numpy as np
import pye57  # type: ignore[import-untyped]
import pytest

from topocore.io.e57.reader import E57Reader
from topocore.pointcloud.attributes import PointAttribute


def _write_e57(path: str, **fields: np.ndarray) -> None:
    e57 = pye57.E57(path, mode="w")
    n = len(next(iter(fields.values())))
    data = {
        "cartesianInvalidState": np.zeros(n, dtype=np.uint8),
        **fields,
    }
    e57.write_scan_raw(data)
    e57.close()


# ----------------------------------------------------------------------
# Bug 1: intensity/color silently dropped.
# ----------------------------------------------------------------------


def test_intensity_and_color_are_read_not_dropped(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    The exact reproduction: before the fix, a real E57 file with
    genuine intensity and RGB color data yielded a Chunk with only
    X/Y/Z.
    """
    path = str(tmp_path / "full.e57")
    _write_e57(
        path,
        cartesianX=np.array([1.0, 2.0, 3.0]),
        cartesianY=np.array([1.0, 2.0, 3.0]),
        cartesianZ=np.array([1.0, 2.0, 3.0]),
        intensity=np.array([0.1, 0.5, 0.9]),
        colorRed=np.array([255, 0, 128], dtype=np.uint8),
        colorGreen=np.array([0, 255, 64], dtype=np.uint8),
        colorBlue=np.array([0, 0, 255], dtype=np.uint8),
    )

    with E57Reader(path, chunk_size=1000) as reader:
        chunk = next(iter(reader.read()))

    assert chunk.has_attribute(PointAttribute.INTENSITY)
    assert chunk.has_attribute(PointAttribute.COLOR)
    np.testing.assert_array_equal(chunk[PointAttribute.COLOR], [[255, 0, 0], [0, 255, 0], [128, 64, 255]])


def test_xyz_only_file_still_works_without_intensity_or_color(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "xyz_only.e57")
    _write_e57(
        path,
        cartesianX=np.array([1.0]),
        cartesianY=np.array([1.0]),
        cartesianZ=np.array([1.0]),
    )

    with E57Reader(path, chunk_size=1000) as reader:
        chunk = next(iter(reader.read()))

    assert not chunk.has_attribute(PointAttribute.INTENSITY)
    assert not chunk.has_attribute(PointAttribute.COLOR)


# ----------------------------------------------------------------------
# Bug 2: intensity truncated to 0 by direct float->uint16 assignment.
# ----------------------------------------------------------------------


def test_normalized_intensity_scaled_to_full_uint16_range(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    The exact reproduction: before the fix, [0.1, 0.5, 0.9] became
    [0, 0, 0] (truncated to the uint16 column's integer dtype).
    """
    path = str(tmp_path / "intensity.e57")
    _write_e57(
        path,
        cartesianX=np.array([1.0, 2.0, 3.0]),
        cartesianY=np.array([1.0, 2.0, 3.0]),
        cartesianZ=np.array([1.0, 2.0, 3.0]),
        intensity=np.array([0.1, 0.5, 0.9]),
    )

    with E57Reader(path, chunk_size=1000) as reader:
        chunk = next(iter(reader.read()))

    intensity = chunk[PointAttribute.INTENSITY]
    assert intensity[0] < intensity[1] < intensity[2]  # order preserved
    assert intensity[0] == 0  # minimum maps to 0
    assert intensity[2] == 65535  # maximum maps to full uint16 range


def test_degenerate_constant_intensity_does_not_divide_by_zero(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "flat_intensity.e57")
    _write_e57(
        path,
        cartesianX=np.array([1.0, 2.0]),
        cartesianY=np.array([1.0, 2.0]),
        cartesianZ=np.array([1.0, 2.0]),
        intensity=np.array([0.5, 0.5]),
    )

    with E57Reader(path, chunk_size=1000) as reader:
        chunk = next(iter(reader.read()))  # must not raise

    np.testing.assert_array_equal(chunk[PointAttribute.INTENSITY], [32767, 32767])


# ----------------------------------------------------------------------
# Pose transform (no bug found -- pye57's own default is transform=True).
# ----------------------------------------------------------------------


def test_pose_transform_applied_by_default(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    Confirms local scan coordinates end up correctly transformed
    into the global E57 coordinate system, matching pye57's own
    transform=True default (no override needed on TopoCore's side).
    """
    path = str(tmp_path / "posed.e57")
    e57 = pye57.E57(path, mode="w")

    # A translation of (100, 0, 0) applied to the scan pose.
    rotation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion (w,x,y,z)
    translation = np.array([100.0, 0.0, 0.0])

    data = {
        "cartesianX": np.array([1.0, 2.0]),
        "cartesianY": np.array([0.0, 0.0]),
        "cartesianZ": np.array([0.0, 0.0]),
        "cartesianInvalidState": np.zeros(2, dtype=np.uint8),
    }
    e57.write_scan_raw(data, rotation=rotation, translation=translation)
    e57.close()

    with E57Reader(path, chunk_size=1000) as reader:
        chunk = next(iter(reader.read()))

    # Local x=1,2 with a +100 translation should read back as ~101, ~102.
    np.testing.assert_allclose(chunk[PointAttribute.X], [101.0, 102.0], atol=1e-6)


# ----------------------------------------------------------------------
# Validation.
# ----------------------------------------------------------------------


def test_rejects_nonpositive_chunk_size() -> None:
    with pytest.raises(ValueError):
        E57Reader("dummy.e57", chunk_size=0)
