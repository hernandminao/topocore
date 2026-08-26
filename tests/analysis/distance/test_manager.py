"""
Regression/coverage suite for topocore.analysis.distance.manager.
DistanceAnalysis -- PR20 coverage phase.

The 6-coordinate Euclidean-dispatch parameter-reordering bug (found
and fixed in PR19) already has its own dedicated regression comment
in the source; this suite adds the remaining real-domain-behavior
gaps: the method/config/geodesic properties, all 5 dispatch branches
in compute() (only EUCLIDEAN had a prior test), the gradient()/
angle() convenience wrappers, and the error paths (missing CRS for
geodesic, geodesic requested but not initialized, unsupported
method, wrong coordinate count). Verified with real geometric cases
(3-4-5, 3-4-12-13) and a real WGS84 CRS for the geodesic path (not
mocked). No bugs found -- only test coverage was added.
"""

from __future__ import annotations

import pytest

from topocore.analysis.distance.manager import DistanceAnalysis, DistanceMethod
from topocore.analysis.exceptions import DistanceError
from topocore.geodesy.crs import CRS

# ----------------------------------------------------------------------
# Properties and defaults.
# ----------------------------------------------------------------------


def test_default_method_is_euclidean() -> None:
    analysis = DistanceAnalysis()
    assert analysis.method == "euclidean"


def test_config_property_returns_distance_config() -> None:
    analysis = DistanceAnalysis()
    assert analysis.config is not None


def test_geodesic_property_is_none_without_crs() -> None:
    analysis = DistanceAnalysis()
    assert analysis.geodesic is None


def test_available_methods_lists_all_five() -> None:
    assert DistanceAnalysis.available_methods() == (
        "euclidean",
        "geodesic",
        "horizontal",
        "vertical",
        "slope",
    )


# ----------------------------------------------------------------------
# compute() -- all 5 dispatch branches with known geometric results.
# ----------------------------------------------------------------------


def test_compute_euclidean_4_coordinates() -> None:
    analysis = DistanceAnalysis()
    result = analysis.compute(0.0, 0.0, 3.0, 4.0, method="euclidean")
    assert result.value == pytest.approx(5.0)


def test_compute_euclidean_6_coordinates_natural_point_order() -> None:
    """The exact PR19 regression case: (x1,y1,z1,x2,y2,z2) natural order, 3-4-12 right triangle."""
    analysis = DistanceAnalysis()
    result = analysis.compute(0.0, 0.0, 0.0, 3.0, 4.0, 12.0, method="euclidean")
    assert result.value == pytest.approx(13.0)


def test_compute_horizontal() -> None:
    analysis = DistanceAnalysis()
    result = analysis.compute(0.0, 0.0, 3.0, 4.0, method="horizontal")
    assert result.value == pytest.approx(5.0)


def test_compute_vertical() -> None:
    analysis = DistanceAnalysis()
    result = analysis.compute(3.0, 7.0, method="vertical")
    assert result.value == pytest.approx(4.0)


def test_compute_slope() -> None:
    analysis = DistanceAnalysis()
    result = analysis.compute(0.0, 0.0, 0.0, 3.0, 4.0, 12.0, method="slope")
    assert result.value == pytest.approx(13.0)


def test_compute_geodesic_via_manager() -> None:
    crs = CRS.from_epsg(4326)
    analysis = DistanceAnalysis(method="geodesic", crs=crs)
    result = analysis.compute(0.0, 0.0, 1.0, 0.0, method="geodesic")
    assert result.value == pytest.approx(111319.49, rel=1e-4)
    assert analysis.geodesic is not None


def test_compute_uses_instance_default_method_when_not_overridden() -> None:
    analysis = DistanceAnalysis(method="horizontal")
    result = analysis.compute(0.0, 0.0, 3.0, 4.0)  # no method kwarg -- uses instance default
    assert result.value == pytest.approx(5.0)


# ----------------------------------------------------------------------
# gradient() / angle() convenience wrappers.
# ----------------------------------------------------------------------


def test_gradient_convenience_method() -> None:
    analysis = DistanceAnalysis()
    assert analysis.gradient(0.0, 0.0, 0.0, 10.0, 0.0, 1.0) == pytest.approx(10.0)


def test_angle_convenience_method() -> None:
    analysis = DistanceAnalysis()
    assert analysis.angle(0.0, 0.0, 0.0, 1.0, 0.0, 1.0) == pytest.approx(45.0)


# ----------------------------------------------------------------------
# __call__
# ----------------------------------------------------------------------


def test_call_matches_compute() -> None:
    analysis = DistanceAnalysis()
    via_call = analysis(0.0, 0.0, 3.0, 4.0, method="euclidean")
    via_compute = analysis.compute(0.0, 0.0, 3.0, 4.0, method="euclidean")
    assert via_call.value == via_compute.value


# ----------------------------------------------------------------------
# Errors.
# ----------------------------------------------------------------------


def test_geodesic_method_without_crs_raises_at_construction() -> None:
    with pytest.raises(DistanceError, match="CRS is required"):
        DistanceAnalysis(method="geodesic")


def test_geodesic_requested_at_call_time_but_not_initialized_raises() -> None:
    analysis = DistanceAnalysis()  # default euclidean, no geodesic calculator
    with pytest.raises(DistanceError, match="not initialized"):
        analysis.compute(0.0, 0.0, 1.0, 0.0, method="geodesic")


def test_unsupported_method_string_raises_at_construction() -> None:
    with pytest.raises(DistanceError, match="Unsupported distance method"):
        DistanceAnalysis(method="bogus")


def test_unsupported_method_string_raises_at_compute_time() -> None:
    analysis = DistanceAnalysis()
    with pytest.raises(DistanceError, match="Unsupported distance method"):
        analysis.compute(0.0, 0.0, 1.0, 1.0, method="bogus")


def test_euclidean_wrong_coordinate_count_raises() -> None:
    analysis = DistanceAnalysis()
    with pytest.raises(DistanceError, match="4 or 6 coordinates"):
        analysis.compute(0.0, 0.0, 0.0, method="euclidean")


def test_distance_method_enum_values() -> None:
    assert DistanceMethod.EUCLIDEAN == "euclidean"
    assert DistanceMethod.GEODESIC == "geodesic"
    assert DistanceMethod.HORIZONTAL == "horizontal"
    assert DistanceMethod.VERTICAL == "vertical"
    assert DistanceMethod.SLOPE == "slope"
