"""
Regression suite for topocore.analysis.distance.euclidean.
EuclideanDistance -- PR19.

Verified against known 3-4-5 and 3-4-12-13 right triangles in 2D
and 3D. No bugs found in this class itself -- note its parameter
order is (x1, y1, x2, y2, z1, z2), NOT the "point-grouped"
(x1, y1, z1, x2, y2, z2) order used elsewhere in this module; see
test_manager.py for the real bug this asymmetry caused in the
dispatcher.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.analysis.distance.euclidean import EuclideanDistance
from topocore.analysis.exceptions import DistanceError


def test_2d_known_3_4_5_triangle() -> None:
    result = EuclideanDistance(dimension="2d").compute(0, 0, 3, 4)
    assert result.value == pytest.approx(5.0)


def test_3d_known_3_4_12_13_triangle() -> None:
    # Its own parameter order: x1, y1, x2, y2, z1, z2.
    result = EuclideanDistance(dimension="3d").compute(0, 0, 3, 4, 0, 12)
    assert result.value == pytest.approx(13.0)


def test_compute_many_matches_scalar_results() -> None:
    engine = EuclideanDistance(dimension="3d")
    points_a = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    points_b = np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0]])

    result = engine.compute_many(points_a, points_b)
    np.testing.assert_allclose(result, [5.0, 0.0])


def test_2d_ignores_z_component_of_compute_many() -> None:
    engine = EuclideanDistance(dimension="2d")
    points_a = np.array([[0.0, 0.0, 100.0]])
    points_b = np.array([[3.0, 4.0, -500.0]])

    result = engine.compute_many(points_a, points_b)
    np.testing.assert_allclose(result, [5.0])


def test_distance_between_points_2d_tuple() -> None:
    engine = EuclideanDistance(dimension="2d")
    result = engine.distance_between_points((0.0, 0.0), (3.0, 4.0))
    assert result.value == pytest.approx(5.0)


def test_rejects_invalid_dimension() -> None:
    with pytest.raises(DistanceError):
        EuclideanDistance(dimension="4d")  # type: ignore[arg-type]


def test_rejects_nan_coordinate() -> None:
    with pytest.raises(DistanceError):
        EuclideanDistance().compute(0, 0, float("nan"), 4)


def test_compute_many_rejects_shape_mismatch() -> None:
    engine = EuclideanDistance(dimension="3d")
    with pytest.raises(DistanceError):
        engine.compute_many(np.zeros((3, 3)), np.zeros((4, 3)))
