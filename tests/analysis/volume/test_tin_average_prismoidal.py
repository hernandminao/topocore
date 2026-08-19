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


def test_prismoidal_currently_equals_average_end_area() -> None:
    """
    Documents the known, deliberately-deferred limitation: given
    identical (station, area) input, PrismoidalVolume and
    AverageEndAreaVolume produce bit-for-bit identical results,
    since the middle-section area is approximated as the average of
    the two endpoints rather than genuinely measured. This test
    exists so a future fix (accepting real triples) shows up as an
    intentional, visible change here, not a silent behavior shift.
    """
    sections = [(0.0, 10.0), (20.0, 30.0)]

    prismoidal_result = PrismoidalVolume(sections).compute()
    average_result = AverageEndAreaVolume(sections).compute()

    assert prismoidal_result.cut_volume == average_result.cut_volume


def test_prismoidal_rejects_out_of_order_stations() -> None:
    with pytest.raises(VolumeError):
        PrismoidalVolume([(10.0, 5.0), (5.0, 5.0)])


def test_prismoidal_rejects_negative_area() -> None:
    with pytest.raises(VolumeError):
        PrismoidalVolume([(0.0, -1.0), (10.0, 5.0)])


def test_prismoidal_rejects_fewer_than_two_sections() -> None:
    with pytest.raises(VolumeError):
        PrismoidalVolume([(0.0, 5.0)])
