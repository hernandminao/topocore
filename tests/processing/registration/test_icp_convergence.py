"""
Regression suite for the convergence criterion in
topocore.processing.registration.icp.ICPBase.register() -- PR19.

Includes a real bug found and fixed in this session: convergence was
detected ONLY via relative change in RMSE
(abs(prev-error)/max(1e-12, abs(prev))). Once RMSE reaches the
floating-point noise floor (a near-perfect alignment, e.g. RMSE
~1e-14), further iterations only move it by rounding noise
(~1e-16) -- but dividing that noise by an already-tiny previous_error
produces relative-change values (1e-4 to 1e-3 in the reproduction)
that never drop below a typical tolerance (1e-6). Confirmed directly:
a synthetic registration with a known, exactly recoverable
transformation reached RMSE ~2e-14 by iteration 35 of 50, yet
`converged` stayed False through iteration 50 -- burning the full
iteration budget and misreporting convergence status despite the
transformation itself already being correct to machine precision.

Fixed by also converging when the RMSE itself is already below
tolerance, not only when its relative change is small.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.registration.base import Transformation
from topocore.processing.registration.point_to_point import PointToPointICP


def _cloud(points: np.ndarray) -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(
        size=len(points),
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = points[:, 0]
    chunk[PointAttribute.Y][:] = points[:, 1]
    chunk[PointAttribute.Z][:] = points[:, 2]
    cloud.add_chunk(chunk)
    return cloud


@pytest.fixture
def known_transform_scene() -> tuple[PointCloud, PointCloud, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    target_points = rng.uniform(-5, 5, (200, 3))

    theta = np.radians(30)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([2.0, 3.0, 0.5])
    known = Transformation.from_rotation_translation(rotation, translation)

    source_points = known.inverse().apply_points(target_points)

    return _cloud(source_points), _cloud(target_points), rotation, translation


def test_near_perfect_alignment_reports_converged(
    known_transform_scene: tuple[PointCloud, PointCloud, np.ndarray, np.ndarray],
) -> None:
    """
    The exact regression: before the fix, this reported
    converged=False after burning all 50 iterations, despite the
    RMSE reaching the floating-point noise floor.
    """
    source, target, _rotation, _translation = known_transform_scene
    icp = PointToPointICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)

    result = icp.register(source, target)

    assert result.converged is True
    assert result.iterations < 50  # stopped early, not forced to the max budget


def test_transformation_still_correct_after_fix(
    known_transform_scene: tuple[PointCloud, PointCloud, np.ndarray, np.ndarray],
) -> None:
    source, target, rotation, translation = known_transform_scene
    icp = PointToPointICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)

    result = icp.register(source, target)

    np.testing.assert_allclose(result.transformation.rotation, rotation, atol=1e-4)
    np.testing.assert_allclose(result.transformation.translation, translation, atol=1e-4)


def test_rmse_at_convergence_is_near_machine_precision(
    known_transform_scene: tuple[PointCloud, PointCloud, np.ndarray, np.ndarray],
) -> None:
    source, target, _rotation, _translation = known_transform_scene
    icp = PointToPointICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)

    result = icp.register(source, target)

    assert result.rmse < 1e-8
