"""
Regression suite for topocore.terrain.hillshade -- PR19.

Every case verified against hand-derived values of the standard
Lambertian illumination formula before writing the test (session
notes): flat terrain at zenith sun gives exactly 255; flat terrain
at 45deg altitude gives exactly 255*sin(45deg); a facet tilted away
from a low sun clamps to exactly 0; a facet whose slope/aspect
exactly matches the sun's zenith/azimuth gives exactly 255
(perpendicular incidence).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from topocore.geometry.point3d import Point3D
from topocore.terrain.exceptions import TerrainValidationError
from topocore.terrain.hillshade import (
    DEFAULT_ALTITUDE,
    DEFAULT_AZIMUTH,
    HillshadeCalculator,
    triangle_hillshade,
)
from topocore.terrain.models import Triangle
from topocore.terrain.tin import TIN


def _flat_triangle() -> Triangle:
    return Triangle(Point3D(0, 0, 5), Point3D(1, 0, 5), Point3D(0, 1, 5))


def _steep_east_triangle() -> Triangle:
    # slope = arctan(5) ~= 78.69 degrees, aspect = East (90).
    return Triangle(Point3D(0, 0, 10), Point3D(1, 0, 5), Point3D(0, 1, 10))


# ----------------------------------------------------------------------
# Analytic reference cases.
# ----------------------------------------------------------------------


def test_flat_terrain_zenith_sun_gives_exact_255() -> None:
    assert triangle_hillshade(_flat_triangle(), azimuth=0.0, altitude=90.0) == pytest.approx(255.0)


def test_flat_terrain_45_degree_altitude_matches_sine_formula() -> None:
    expected = 255.0 * math.sin(math.radians(45.0))
    assert triangle_hillshade(_flat_triangle(), azimuth=315.0, altitude=45.0) == pytest.approx(expected)


def test_facet_facing_away_from_low_sun_clamps_to_zero() -> None:
    """
    Hand-derived (session notes): steep East-facing facet
    (slope~=78.69, aspect=90) under the default NW sun (azimuth=315,
    altitude=45) has a negative cosine of incidence -- clamped to 0,
    not left negative.
    """
    result = triangle_hillshade(_steep_east_triangle(), azimuth=DEFAULT_AZIMUTH, altitude=DEFAULT_ALTITUDE)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_perpendicular_incidence_gives_exact_255() -> None:
    """
    When the sun's azimuth matches the facet's aspect and the sun's
    zenith matches the facet's slope, the sun is exactly
    perpendicular to the facet -- illumination must be exactly 255.
    """
    steep_slope_deg = math.degrees(math.atan(5.0))
    result = triangle_hillshade(_steep_east_triangle(), azimuth=90.0, altitude=90.0 - steep_slope_deg)
    assert result == pytest.approx(255.0, abs=1e-6)


def test_hillshade_never_negative() -> None:
    """
    cos_incidence can be mathematically negative -- output must
    always be clamped to >= 0, never a negative "illumination".
    """
    for azimuth in [0.0, 45.0, 90.0, 180.0, 270.0]:
        result = triangle_hillshade(_steep_east_triangle(), azimuth=azimuth, altitude=10.0)
        assert result >= 0.0


def test_hillshade_never_exceeds_255() -> None:
    for altitude in [0.0, 30.0, 60.0, 90.0]:
        result = triangle_hillshade(_flat_triangle(), azimuth=180.0, altitude=altitude)
        assert result <= 255.0


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("azimuth", [-0.1, 360.1])
def test_rejects_out_of_range_azimuth(azimuth: float) -> None:
    with pytest.raises(TerrainValidationError):
        triangle_hillshade(_flat_triangle(), azimuth=azimuth, altitude=45.0)


@pytest.mark.parametrize("altitude", [-0.1, 90.1])
def test_rejects_out_of_range_altitude(altitude: float) -> None:
    with pytest.raises(TerrainValidationError):
        triangle_hillshade(_flat_triangle(), azimuth=180.0, altitude=altitude)


def test_accepts_inclusive_boundary_values() -> None:
    triangle_hillshade(_flat_triangle(), azimuth=0.0, altitude=0.0)  # must not raise
    triangle_hillshade(_flat_triangle(), azimuth=360.0, altitude=90.0)  # must not raise


# ----------------------------------------------------------------------
# HillshadeCalculator over a real TIN.
# ----------------------------------------------------------------------


@pytest.fixture
def two_triangle_tin() -> TIN:
    vertices = (
        Point3D(0.0, 0.0, 10.0),
        Point3D(1.0, 0.0, 5.0),
        Point3D(0.0, 1.0, 10.0),
        Point3D(1.0, 1.0, 5.0),
    )
    simplices = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return TIN.from_mesh(vertices, simplices)


def test_calculator_rejects_invalid_azimuth_at_construction(
    two_triangle_tin: TIN,
) -> None:
    with pytest.raises(TerrainValidationError):
        HillshadeCalculator(two_triangle_tin, azimuth=400.0)


def test_calculator_uses_default_azimuth_and_altitude(two_triangle_tin: TIN) -> None:
    calculator = HillshadeCalculator(two_triangle_tin)
    assert calculator.azimuth == DEFAULT_AZIMUTH
    assert calculator.altitude == DEFAULT_ALTITUDE


def test_calculator_compute_matches_direct_triangle_calls(
    two_triangle_tin: TIN,
) -> None:
    calculator = HillshadeCalculator(two_triangle_tin, azimuth=180.0, altitude=60.0)
    values = calculator.compute()

    triangle0 = Triangle(
        two_triangle_tin.vertices[0],
        two_triangle_tin.vertices[1],
        two_triangle_tin.vertices[2],
    )
    expected0 = triangle_hillshade(triangle0, azimuth=180.0, altitude=60.0)

    assert values[0] == pytest.approx(expected0)


def test_calculator_callable_matches_compute(two_triangle_tin: TIN) -> None:
    calculator = HillshadeCalculator(two_triangle_tin)
    np.testing.assert_array_equal(calculator(), calculator.compute())
