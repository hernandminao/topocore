"""
Regression suite for topocore.io.las.converter.LASConverter -- PR19.

Verified with real laspy-written LAS files (not mocks): LAS stores
X/Y/Z as scaled integers internally, and laspy exposes both a "raw"
unscaled property and a "scaled" (real-world) one -- a classic,
easy-to-get-wrong LAS I/O pitfall. Confirmed directly that
LASConverter correctly reads the SCALED coordinates, with realistic
UTM-scale values and non-trivial scale/offset. Also verified
intensity, classification, and RGB color (combined from three
separate LAS channels into one PointAttribute.COLOR). No bugs found
in this file.
"""

from __future__ import annotations

import laspy  # type: ignore[import-untyped]
import numpy as np

from topocore.io.las.converter import LASConverter
from topocore.pointcloud.attributes import PointAttribute


def _write_las(path: str, scale, offset, **fields) -> None:  # type: ignore[no-untyped-def]
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = list(scale)
    header.offsets = list(offset)
    las = laspy.LasData(header)
    for name, values in fields.items():
        setattr(las, name, values)
    las.write(path)


def test_scaled_coordinates_not_raw_integers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    The decisive check: LAS stores coordinates as scaled integers.
    Confirms the converter reads the real-world scaled value, not
    the raw, unscaled integer.
    """
    path = str(tmp_path / "scaled.las")
    known_x = np.array([500000.123, 500001.456, 500002.789])
    known_y = np.array([4000000.111, 4000000.222, 4000000.333])
    known_z = np.array([100.5, 101.25, 99.75])

    _write_las(
        path,
        (0.001, 0.001, 0.001),
        (500000.0, 4000000.0, 100.0),
        x=known_x,
        y=known_y,
        z=known_z,
    )

    with laspy.open(path) as f:
        points = f.read().points

    chunk = LASConverter.from_las_points(points)

    np.testing.assert_allclose(chunk[PointAttribute.X], known_x, atol=1e-3)
    np.testing.assert_allclose(chunk[PointAttribute.Y], known_y, atol=1e-3)
    np.testing.assert_allclose(chunk[PointAttribute.Z], known_z, atol=1e-3)


def test_intensity_and_classification(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "attrs.las")
    _write_las(
        path,
        (0.01, 0.01, 0.01),
        (0.0, 0.0, 0.0),
        x=np.array([1.0, 2.0]),
        y=np.array([1.0, 2.0]),
        z=np.array([1.0, 2.0]),
        intensity=np.array([1000, 2000], dtype=np.uint16),
        classification=np.array([2, 5], dtype=np.uint8),
    )

    with laspy.open(path) as f:
        points = f.read().points

    chunk = LASConverter.from_las_points(points)

    np.testing.assert_array_equal(chunk[PointAttribute.INTENSITY], [1000, 2000])
    np.testing.assert_array_equal(chunk[PointAttribute.CLASSIFICATION], [2, 5])


def test_rgb_color_combined_from_three_channels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = str(tmp_path / "color.las")
    _write_las(
        path,
        (0.01, 0.01, 0.01),
        (0.0, 0.0, 0.0),
        x=np.array([1.0, 2.0]),
        y=np.array([1.0, 2.0]),
        z=np.array([1.0, 2.0]),
        red=np.array([255, 0], dtype=np.uint16),
        green=np.array([128, 255], dtype=np.uint16),
        blue=np.array([0, 64], dtype=np.uint16),
    )

    with laspy.open(path) as f:
        points = f.read().points

    chunk = LASConverter.from_las_points(points)
    np.testing.assert_array_equal(chunk[PointAttribute.COLOR], [[255, 128, 0], [0, 255, 64]])


def test_color_omitted_when_point_format_lacks_rgb(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    point_format=3 structurally includes red/green/blue fields per
    the ASPRS spec, regardless of whether meaningful values were
    ever set -- point_format=0 genuinely lacks them, the correct way
    to test color's absence.
    """
    path = str(tmp_path / "nocolor.las")
    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = [0.01, 0.01, 0.01]
    header.offsets = [0.0, 0.0, 0.0]
    las = laspy.LasData(header)
    las.x = np.array([1.0])
    las.y = np.array([1.0])
    las.z = np.array([1.0])
    las.write(path)

    with laspy.open(path) as f:
        points = f.read().points

    chunk = LASConverter.from_las_points(points)
    assert not chunk.has_attribute(PointAttribute.COLOR)
