"""
Regression suite for topocore.analysis.volume.tin_volume,
.average_end_area, and .prismoidal -- PR19.

Each verified INDIVIDUALLY against known geometric/numeric cases,
not assumed to share the NaN-handling bug already found and fixed
in cut_fill.py/grid_volume.py -- confirmed none of them do (a
TIN cannot even be constructed with NaN vertices in this codebase,
and the two cross-section-based methods have no grid/NaN concept at
all).

Includes a real, non-numeric finding documented (not fixed) in this
session: PrismoidalVolume is currently mathematically equivalent to
AverageEndAreaVolume, since it approximates the middle-section area
as the average of the two endpoints rather than accepting a genuinely
measured middle section -- defeating the purpose of Simpson's rule.
Deliberately deferred as a separate design/API decision; see the
module's own docstring for the full explanation.
"""

from __future__ import annotations

import pytest

from topocore.analysis.exceptions import VolumeError
from topocore.analysis.volume.average_end_area import AverageEndAreaVolume
from topocore.analysis.volume.prismoidal import PrismoidalVolume
from topocore.analysis.volume.tin_volume import TINVolume
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN

# ----------------------------------------------------------------------
# TINVolume
# ----------------------------------------------------------------------


def test_tin_volume_flat_triangle_above_datum() -> None:
    # Right triangle, legs 4 and 3 (area=6), flat at z=5, datum=0.
    points = (Point3D(0, 0, 5.0), Point3D(4, 0, 5.0), Point3D(0, 3, 5.0))
    tin = TIN.from_points(points)

    result = TINVolume(datum=0.0).compute(tin)

    assert result.cut_volume == pytest.approx(30.0)
    assert result.fill_volume == pytest.approx(0.0)


def test_tin_volume_flat_triangle_below_datum() -> None:
    points = (Point3D(0, 0, -5.0), Point3D(4, 0, -5.0), Point3D(0, 3, -5.0))
    tin = TIN.from_points(points)

    result = TINVolume(datum=0.0).compute(tin)

    assert result.cut_volume == pytest.approx(0.0)
    assert result.fill_volume == pytest.approx(30.0)


def test_tin_cannot_be_constructed_with_nan_vertex() -> None:
    """
    Confirms TINVolume's own NaN vertex check is defensive, not a
    reachable gap -- TIN.from_points() already rejects NaN at
    construction, matching a pattern already seen elsewhere in this
    session (e.g. Chunk always requiring Z).
    """
    with pytest.raises(Exception):  # noqa: B017 - MathError from the geometry layer
        TIN.from_points((Point3D(0, 0, float("nan")), Point3D(1, 0, 1.0), Point3D(0, 1, 2.0)))


def test_tin_volume_rejects_nonfinite_datum() -> None:
    with pytest.raises(VolumeError):
        TINVolume(datum=float("nan"))


def test_tin_volume_rejects_empty_tin() -> None:
    class _EmptyTIN:
        triangle_count = 0

    with pytest.raises(VolumeError):
        TINVolume().compute(_EmptyTIN())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# AverageEndAreaVolume
# ----------------------------------------------------------------------


def test_average_end_area_constant_sections() -> None:
    volume = AverageEndAreaVolume([(0.0, 20.0), (10.0, 20.0)])
    result = volume.compute()
    assert result.cut_volume == pytest.approx(200.0)
    assert result.fill_volume == pytest.approx(0.0)


def test_average_end_area_trapezoidal_sections() -> None:
    volume = AverageEndAreaVolume([(0.0, 10.0), (20.0, 30.0)])
    result = volume.compute()
    assert result.cut_volume == pytest.approx(400.0)


def test_average_end_area_rejects_out_of_order_stations() -> None:
    with pytest.raises(VolumeError):
        AverageEndAreaVolume([(10.0, 5.0), (5.0, 5.0)])


def test_average_end_area_rejects_negative_area() -> None:
    with pytest.raises(VolumeError):
        AverageEndAreaVolume([(0.0, -1.0), (10.0, 5.0)])


def test_average_end_area_rejects_fewer_than_two_sections() -> None:
    with pytest.raises(VolumeError):
        AverageEndAreaVolume([(0.0, 5.0)])


# ----------------------------------------------------------------------
# PrismoidalVolume -- the documented known limitation.
# ----------------------------------------------------------------------


def test_prismoidal_uses_measured_middle_section() -> None:
    """Use the measured middle section in Simpson's 1/3 rule."""
    sections = [
        (0.0, 10.0),
        (10.0, 30.0),
        (20.0, 10.0),
    ]

    result = PrismoidalVolume(sections).compute()

    expected = 10.0 / 3.0 * (10.0 + 4.0 * 30.0 + 10.0)

    assert result.cut_volume == pytest.approx(expected)
    assert result.fill_volume == pytest.approx(0.0)
    assert result.net_volume == pytest.approx(expected)
    assert result.method == "prismoidal"


def test_prismoidal_rejects_out_of_order_stations() -> None:
    with pytest.raises(VolumeError):
        PrismoidalVolume([(10.0, 5.0), (5.0, 5.0)])


def test_prismoidal_rejects_negative_area() -> None:
    with pytest.raises(VolumeError):
        PrismoidalVolume([(0.0, -1.0), (10.0, 5.0)])


def test_prismoidal_rejects_fewer_than_three_sections() -> None:
    """Reject fewer than three sections."""
    with pytest.raises(VolumeError):
        PrismoidalVolume(
            [
                (0.0, 5.0),
                (10.0, 5.0),
            ]
        )


def test_prismoidal_rejects_even_number_of_sections() -> None:
    """Reject an even number of sections."""
    with pytest.raises(VolumeError):
        PrismoidalVolume(
            [
                (0.0, 5.0),
                (10.0, 10.0),
                (20.0, 5.0),
                (30.0, 10.0),
            ]
        )
