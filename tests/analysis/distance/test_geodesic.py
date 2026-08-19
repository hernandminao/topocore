"""
Regression suite for topocore.analysis.distance.geodesic.
GeodesicDistance -- PR19. Verified against a known WGS84 equatorial
distance. No bugs found (delegates to the already-audited
topocore.geodesy.geodesic module).
"""

from __future__ import annotations

import pytest

from topocore.analysis.distance.geodesic import GeodesicDistance
from topocore.analysis.exceptions import DistanceError
from topocore.geodesy.crs import CRS


@pytest.fixture
def wgs84_distance() -> GeodesicDistance:
    return GeodesicDistance(crs=CRS.from_epsg(4326))


def test_one_degree_longitude_at_equator(wgs84_distance: GeodesicDistance) -> None:
    result = wgs84_distance.compute(0.0, 0.0, 1.0, 0.0)
    # ~111.32 km, standard WGS84 equatorial degree length.
    assert result.value == pytest.approx(111319.49, abs=1.0)


def test_rejects_out_of_range_longitude(wgs84_distance: GeodesicDistance) -> None:
    with pytest.raises(DistanceError):
        wgs84_distance.compute(200.0, 0.0, 1.0, 0.0)


def test_rejects_out_of_range_latitude(wgs84_distance: GeodesicDistance) -> None:
    with pytest.raises(DistanceError):
        wgs84_distance.compute(0.0, 95.0, 1.0, 0.0)


def test_rejects_nan_coordinate(wgs84_distance: GeodesicDistance) -> None:
    with pytest.raises(DistanceError):
        wgs84_distance.compute(0.0, 0.0, float("nan"), 0.0)


def test_polygon_area_requires_at_least_three_vertices(
    wgs84_distance: GeodesicDistance,
) -> None:
    with pytest.raises(DistanceError):
        wgs84_distance.polygon_area([0.0, 1.0], [0.0, 1.0])
