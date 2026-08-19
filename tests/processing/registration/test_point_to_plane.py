"""
Regression suite for topocore.processing.registration.point_to_plane
.PointToPlaneICP -- PR19.

Includes a SEVERE bug found and fixed in this session: the
linearized point-to-plane system's right-hand side had the wrong
sign -- `rhs = n . (source - target)` instead of the correct
`n . (target - source)`. Confirmed by directly inspecting A/b/x for
independent, hand-derivable pure cases (see below) -- negating ONLY
the right-hand side fixed both a pure translation and a pure
rotation case exactly, confirming a single, uniform sign error, not
separate bugs in the cross-product convention. With the wrong sign,
every ICP iteration moved the source cloud further from the target,
causing correspondence counts to shrink each iteration until
registration failed entirely -- reproducible even with small,
realistic initial offsets (5 degrees), not merely large or
degenerate ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.registration.base import Transformation
from topocore.processing.registration.point_to_plane import PointToPlaneICP


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


# ----------------------------------------------------------------------
# Direct A/b/x inspection -- the decisive evidence, per the session's
# investigation plan: verify the linear system itself, not just the
# end-to-end registration outcome.
# ----------------------------------------------------------------------


def test_pure_z_translation_linear_system_solves_correct_sign() -> None:
    """
    Flat plane (normal (0,0,1) everywhere), source shifted +0.5 in Z.
    The correct incremental translation is -0.5 (move the source back
    down onto the target) -- before the fix, this resolved as +0.5
    (moving further away).
    """
    rng = np.random.default_rng(0)
    target_points = np.column_stack([rng.uniform(-5, 5, 50), rng.uniform(-5, 5, 50), np.zeros(50)])
    normals = np.tile([0.0, 0.0, 1.0], (50, 1))
    source_points = target_points.copy()
    source_points[:, 2] += 0.5

    icp = PointToPlaneICP()
    system_matrix, rhs = icp._build_linear_system(
        source_points=source_points, target_points=target_points, normals=normals
    )
    solution, *_ = np.linalg.lstsq(system_matrix, rhs, rcond=None)

    np.testing.assert_allclose(solution[:3], [0.0, 0.0, 0.0], atol=1e-9)  # omega
    np.testing.assert_allclose(solution[3:], [0.0, 0.0, -0.5], atol=1e-9)  # translation


def test_pure_y_rotation_linear_system_solves_correct_sign() -> None:
    """
    Extended flat plane, source rotated by a known +1 degree about Y
    (tilting it). The correct incremental omega_y is +0.01745 rad
    (matching the known rotation's sign, mapping source back onto
    target) -- before the fix, this resolved as -0.01745 (wrong sign,
    correct magnitude).
    """
    rng = np.random.default_rng(2)
    n = 300
    target_points = np.column_stack([rng.uniform(-10, 10, n), rng.uniform(-10, 10, n), np.zeros(n)])
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))

    angle = np.radians(1.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    known = Transformation.from_rotation_translation(rotation, np.zeros(3))
    source_points = known.inverse().apply_points(target_points)

    icp = PointToPlaneICP()
    system_matrix, rhs = icp._build_linear_system(
        source_points=source_points, target_points=target_points, normals=normals
    )
    solution, *_ = np.linalg.lstsq(system_matrix, rhs, rcond=None)

    assert solution[1] == pytest.approx(angle, abs=1e-3)  # omega_y, correct sign and magnitude
    np.testing.assert_allclose(solution[3:], [0.0, 0.0, 0.0], atol=1e-6)  # no translation


# ----------------------------------------------------------------------
# End-to-end registration -- confirms the sign fix through the full
# ICP loop, not just the isolated linear system.
# ----------------------------------------------------------------------


@pytest.fixture
def flat_target_cloud() -> tuple[PointCloud, np.ndarray]:
    rng = np.random.default_rng(0)
    points = rng.uniform(-5, 5, (200, 3))
    return _cloud(points), points


def test_small_realistic_offset_no_longer_diverges(
    flat_target_cloud: tuple[PointCloud, np.ndarray],
) -> None:
    """
    The exact reproduction: before the fix, even a small (5 degree)
    initial offset caused correspondence counts to shrink each
    iteration until registration failed with "Not enough
    correspondences".
    """
    target_cloud, target_points = flat_target_cloud

    theta = np.radians(5)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    translation = np.array([0.3, 0.2, 0.1])
    known = Transformation.from_rotation_translation(rotation, translation)
    source_points = known.inverse().apply_points(target_points)
    source_cloud = _cloud(source_points)

    icp = PointToPlaneICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)
    result = icp.register(source_cloud, target_cloud)  # must not raise

    assert result.converged is True
    np.testing.assert_allclose(result.transformation.rotation, rotation, atol=1e-3)
    np.testing.assert_allclose(result.transformation.translation, translation, atol=1e-3)


def test_larger_offset_also_converges(
    flat_target_cloud: tuple[PointCloud, np.ndarray],
) -> None:
    """
    The original 30-degree reproduction that first exposed the
    divergence, now also fixed.
    """
    target_cloud, target_points = flat_target_cloud

    theta = np.radians(30)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    translation = np.array([2.0, 3.0, 0.5])
    known = Transformation.from_rotation_translation(rotation, translation)
    source_points = known.inverse().apply_points(target_points)
    source_cloud = _cloud(source_points)

    icp = PointToPlaneICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)
    result = icp.register(source_cloud, target_cloud)

    assert result.converged is True
    np.testing.assert_allclose(result.transformation.rotation, rotation, atol=1e-3)
    np.testing.assert_allclose(result.transformation.translation, translation, atol=1e-3)


def test_zero_offset_still_works(
    flat_target_cloud: tuple[PointCloud, np.ndarray],
) -> None:
    """
    Sanity check: source == target exactly must still register
    trivially (this already worked before the fix -- confirms the
    fix didn't break the degenerate zero-offset case).
    """
    target_cloud, target_points = flat_target_cloud
    source_cloud = _cloud(target_points.copy())

    icp = PointToPlaneICP(max_iterations=5, tolerance=1e-6, max_correspondence_distance=20.0)
    result = icp.register(source_cloud, target_cloud)

    assert result.converged is True
    assert result.rmse == pytest.approx(0.0, abs=1e-9)


def test_pure_translation_along_each_axis_converges() -> None:
    """
    Per the session's investigation plan: pure translations along
    each axis, tested independently.
    """
    rng = np.random.default_rng(5)
    target_points = rng.uniform(-5, 5, (150, 3))
    target_cloud = _cloud(target_points)

    for axis in range(3):
        translation = np.zeros(3)
        translation[axis] = 0.4
        known = Transformation.from_rotation_translation(np.eye(3), translation)
        source_points = known.inverse().apply_points(target_points)
        source_cloud = _cloud(source_points)

        icp = PointToPlaneICP(max_iterations=50, tolerance=1e-6, max_correspondence_distance=20.0)
        result = icp.register(source_cloud, target_cloud)

        assert result.converged is True, f"failed to converge for translation along axis {axis}"
        np.testing.assert_allclose(result.transformation.translation, translation, atol=1e-3)
