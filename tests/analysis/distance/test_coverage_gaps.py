"""
Targeted coverage suite for topocore.analysis.distance.euclidean,
.geodesic, .horizontal, .slope, and .vertical -- PR20 coverage
phase.

Formalizes real-domain-behavior verification already performed
manually during this session's audit (confirmed no bugs in any of
these 5 files) into permanent pytest regressions -- covering what
the existing PR19 regression files (test_euclidean.py,
test_geodesic.py, test_horizontal_vertical_slope.py) only partially
exercised: dimension property, compute_many's full validation chain
(shape mismatch, wrong ndim, wrong column count, NaN/inf) and its
"instance dimension mode always wins" behavior (a 2D-configured
instance silently ignores Z even when 3D arrays/tuples are passed --
confirmed this is consistent, deliberate behavior across compute_many
AND distance_between_points, not a bug), distance_between_points'
2D/3D/unsupported-dimension branches, non-numeric coordinate
rejection, and every __call__ operator; GeodesicDistance's azimuth(),
inverse() (confirmed real return order is forward_azimuth,
back_azimuth, distance -- verified against the source docstring, not
assumed), polygon_area() (including its own <3-vertices and
mismatched-length guards), calculator property, and initialization
failure (a CRS without an ellipsoid); HorizontalDistance and
VerticalDistance's compute_many/distance_between_points/__call__ and
error paths; and SlopeDistance's compute_many, from_horizontal_and_
vertical, and the slope_gradient-vs-slope_angle asymmetry at
horizontal=0 (slope_gradient raises since percentage grade is
genuinely undefined at zero run; slope_angle correctly returns 90°
via atan2, which is well-defined even at zero horizontal distance --
confirmed this is a deliberate, mathematically correct asymmetry,
not an inconsistency).

No bugs found -- only test coverage was added.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.distance.euclidean import EuclideanDistance
from topocore.analysis.distance.geodesic import GeodesicDistance
from topocore.analysis.distance.horizontal import HorizontalDistance
from topocore.analysis.distance.slope import SlopeDistance
from topocore.analysis.distance.vertical import VerticalDistance
from topocore.analysis.exceptions import DistanceError
from topocore.geodesy.crs import CRS

# ----------------------------------------------------------------------
# EuclideanDistance
# ----------------------------------------------------------------------


def test_euclidean_dimension_property() -> None:
    assert EuclideanDistance(dimension="2d").dimension == "2d"
    assert EuclideanDistance(dimension="3d").dimension == "3d"


def test_euclidean_compute_many_rejects_mismatched_shapes() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="identical shapes"):
        engine.compute_many(np.array([[0.0, 0.0, 0.0]]), np.array([[0.0, 0.0]]))


def test_euclidean_compute_many_rejects_non_2d_array() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="two-dimensional"):
        engine.compute_many(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]))


def test_euclidean_compute_many_rejects_wrong_column_count() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="2 or 3 dimensions"):
        engine.compute_many(np.array([[0.0, 0.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0, 0.0]]))


def test_euclidean_compute_many_rejects_nan() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="NaN or infinite"):
        engine.compute_many(np.array([[0.0, 0.0, np.nan]]), np.array([[1.0, 1.0, 1.0]]))


def test_euclidean_dimension_mode_always_wins_in_compute_many() -> None:
    """A 2D-configured instance ignores Z even when 3D arrays are passed -- deliberate, not a bug."""
    engine = EuclideanDistance(dimension="2d")
    result = engine.compute_many(np.array([[0.0, 0.0, 5.0]]), np.array([[3.0, 4.0, 10.0]]))
    assert result[0] == pytest.approx(5.0)  # Z difference (5) silently ignored


def test_euclidean_distance_between_points_2d() -> None:
    engine = EuclideanDistance(dimension="2d")
    result = engine.distance_between_points((0.0, 0.0), (3.0, 4.0))
    assert result.value == pytest.approx(5.0)


def test_euclidean_distance_between_points_3d() -> None:
    engine = EuclideanDistance(dimension="3d")
    result = engine.distance_between_points((0.0, 0.0, 0.0), (3.0, 4.0, 12.0))
    assert result.value == pytest.approx(13.0)


def test_euclidean_distance_between_points_dimension_mode_wins() -> None:
    """Same instance-mode precedence as compute_many -- confirmed consistent, not a bug."""
    engine = EuclideanDistance(dimension="2d")
    result = engine.distance_between_points((0.0, 0.0, 5.0), (3.0, 4.0, 10.0))
    assert result.value == pytest.approx(5.0)


def test_euclidean_distance_between_points_mismatched_length_raises() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="dimensionality mismatch"):
        engine.distance_between_points((0.0, 0.0), (1.0, 1.0, 1.0))


def test_euclidean_distance_between_points_unsupported_dimension_raises() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError, match="Unsupported point dimension"):
        engine.distance_between_points((0.0,), (1.0,))


def test_euclidean_call_matches_compute() -> None:
    engine = EuclideanDistance(dimension="3d")
    assert engine(0.0, 0.0, 3.0, 4.0, 0.0, 12.0).value == engine.compute(0.0, 0.0, 3.0, 4.0, 0.0, 12.0).value


def test_euclidean_rejects_non_numeric_coordinate() -> None:
    engine = EuclideanDistance(dimension="2d")
    with pytest.raises(DistanceError, match="must be numeric"):
        engine.compute("a", 0.0, 1.0, 1.0)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# GeodesicDistance
# ----------------------------------------------------------------------


@pytest.fixture
def wgs84() -> GeodesicDistance:
    return GeodesicDistance(CRS.from_epsg(4326))


def test_geodesic_azimuth_due_east_at_equator(wgs84: GeodesicDistance) -> None:
    assert wgs84.azimuth(0.0, 0.0, 1.0, 0.0) == pytest.approx(90.0)


def test_geodesic_inverse_return_order_is_az1_az2_distance(
    wgs84: GeodesicDistance,
) -> None:
    """Confirmed against the source docstring: (forward_azimuth, back_azimuth, distance), not (distance, az, az)."""
    az1, az2, distance = wgs84.inverse(0.0, 0.0, 1.0, 0.0)
    assert az1 == pytest.approx(90.0)
    assert az2 == pytest.approx(-90.0)
    assert distance == pytest.approx(111319.49, abs=1.0)


def test_geodesic_polygon_area_positive_for_small_square_near_equator(
    wgs84: GeodesicDistance,
) -> None:
    lons = [0.0, 0.1, 0.1, 0.0]
    lats = [0.0, 0.0, 0.1, 0.1]
    area = wgs84.polygon_area(lons, lats)
    assert area > 0.0


def test_geodesic_polygon_area_rejects_fewer_than_three_vertices(
    wgs84: GeodesicDistance,
) -> None:
    with pytest.raises(DistanceError, match="at least three vertices"):
        wgs84.polygon_area([0.0, 1.0], [0.0, 1.0])


def test_geodesic_polygon_area_rejects_mismatched_lengths(
    wgs84: GeodesicDistance,
) -> None:
    with pytest.raises(DistanceError, match="equal length"):
        wgs84.polygon_area([0.0, 1.0, 2.0], [0.0, 1.0])


def test_geodesic_call_matches_compute(wgs84: GeodesicDistance) -> None:
    assert wgs84(0.0, 0.0, 1.0, 0.0).value == wgs84.compute(0.0, 0.0, 1.0, 0.0).value


def test_geodesic_calculator_property_exposes_underlying_calculator(
    wgs84: GeodesicDistance,
) -> None:
    assert wgs84.calculator is not None


def test_geodesic_rejects_nan_coordinate(wgs84: GeodesicDistance) -> None:
    with pytest.raises(DistanceError, match="finite"):
        wgs84.compute(float("nan"), 0.0, 1.0, 0.0)


def test_geodesic_rejects_out_of_range_latitude(wgs84: GeodesicDistance) -> None:
    with pytest.raises(DistanceError, match="Latitude"):
        wgs84.compute(0.0, 100.0, 1.0, 0.0)


def test_geodesic_initialization_fails_for_crs_without_ellipsoid() -> None:
    class FakeCRS:
        ellipsoid = None

    with pytest.raises(DistanceError, match="Failed to initialize"):
        GeodesicDistance(FakeCRS())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# HorizontalDistance
# ----------------------------------------------------------------------


def test_horizontal_compute_many() -> None:
    result = HorizontalDistance.compute_many(np.array([[0.0, 0.0]]), np.array([[3.0, 4.0]]))
    assert result[0] == pytest.approx(5.0)


def test_horizontal_compute_many_ignores_z_when_present() -> None:
    result = HorizontalDistance.compute_many(np.array([[0.0, 0.0, 100.0]]), np.array([[3.0, 4.0, 999.0]]))
    assert result[0] == pytest.approx(5.0)


def test_horizontal_compute_many_rejects_mismatched_shapes() -> None:
    with pytest.raises(DistanceError, match="identical shapes"):
        HorizontalDistance.compute_many(np.array([[0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]))


def test_horizontal_compute_many_rejects_fewer_than_two_columns() -> None:
    with pytest.raises(DistanceError, match="at least X and Y"):
        HorizontalDistance.compute_many(np.array([[0.0]]), np.array([[1.0]]))


def test_horizontal_compute_many_rejects_nan() -> None:
    with pytest.raises(DistanceError, match="NaN or infinite"):
        HorizontalDistance.compute_many(np.array([[0.0, np.nan]]), np.array([[1.0, 1.0]]))


def test_horizontal_distance_between_points() -> None:
    result = HorizontalDistance.distance_between_points((0.0, 0.0), (3.0, 4.0))
    assert result.value == pytest.approx(5.0)


def test_horizontal_distance_between_points_rejects_wrong_dimension() -> None:
    with pytest.raises(DistanceError, match="requires 2D points"):
        HorizontalDistance.distance_between_points((0.0, 0.0, 0.0), (1.0, 1.0))  # type: ignore[arg-type]


def test_horizontal_call_matches_compute() -> None:
    engine = HorizontalDistance()
    assert engine(0.0, 0.0, 3.0, 4.0).value == HorizontalDistance.compute(0.0, 0.0, 3.0, 4.0).value


# ----------------------------------------------------------------------
# VerticalDistance
# ----------------------------------------------------------------------


def test_vertical_compute_many() -> None:
    result = VerticalDistance.compute_many(np.array([5.0, 1.0]), np.array([8.0, -3.0]))
    np.testing.assert_allclose(result, [3.0, 4.0])


def test_vertical_compute_many_rejects_mismatched_shapes() -> None:
    with pytest.raises(DistanceError, match="identical shapes"):
        VerticalDistance.compute_many(np.array([5.0]), np.array([5.0, 6.0]))


def test_vertical_compute_many_rejects_nan() -> None:
    with pytest.raises(DistanceError, match="NaN or infinite"):
        VerticalDistance.compute_many(np.array([np.nan]), np.array([1.0]))


def test_vertical_call_matches_compute() -> None:
    engine = VerticalDistance()
    assert engine(5.0, 8.0).value == VerticalDistance.compute(5.0, 8.0).value


def test_vertical_rejects_non_finite() -> None:
    with pytest.raises(DistanceError, match="finite"):
        VerticalDistance.compute(float("inf"), 5.0)


# ----------------------------------------------------------------------
# SlopeDistance
# ----------------------------------------------------------------------


def test_slope_compute_many() -> None:
    result = SlopeDistance.compute_many(np.array([[0.0, 0.0, 0.0]]), np.array([[3.0, 4.0, 12.0]]))
    assert result[0] == pytest.approx(13.0)


def test_slope_compute_many_rejects_wrong_shape() -> None:
    with pytest.raises(DistanceError, match="shape \\(N,3\\)"):
        SlopeDistance.compute_many(np.array([[0.0, 0.0]]), np.array([[1.0, 1.0]]))


def test_slope_from_horizontal_and_vertical() -> None:
    result = SlopeDistance.from_horizontal_and_vertical(3.0, 4.0)
    assert result.value == pytest.approx(5.0)


def test_slope_from_horizontal_and_vertical_rejects_negative() -> None:
    with pytest.raises(DistanceError, match="cannot be negative"):
        SlopeDistance.from_horizontal_and_vertical(-1.0, 4.0)


def test_slope_angle_well_defined_at_zero_horizontal() -> None:
    """
    The deliberate asymmetry: slope_gradient raises at horizontal=0
    (percentage grade is genuinely undefined), but slope_angle
    returns 90 degrees via atan2, which handles this case cleanly.
    """
    angle = SlopeDistance.slope_angle(0.0, 0.0, 0.0, 0.0, 0.0, 5.0)
    assert angle == pytest.approx(90.0)


def test_slope_gradient_raises_at_zero_horizontal() -> None:
    with pytest.raises(DistanceError, match="undefined"):
        SlopeDistance.slope_gradient(0.0, 0.0, 0.0, 0.0, 0.0, 5.0)


def test_slope_call_matches_compute() -> None:
    engine = SlopeDistance()
    assert engine(0.0, 0.0, 0.0, 3.0, 4.0, 12.0).value == SlopeDistance.compute(0.0, 0.0, 0.0, 3.0, 4.0, 12.0).value


def test_slope_method_property() -> None:
    assert SlopeDistance().method == "slope"
