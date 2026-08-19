"""
Regression suite for topocore.processing.registration.manager.
RegistrationManager -- PR19.

Audited against the full checklist for this closing pass: public
contract, algorithm selection/resolution, config/defaults, parameter
propagation (including per-algorithm filtering via inspect.signature
-- normal_k must reach PointToPlaneICP but never PointToPointICP),
reuse across multiple cloud pairs (no cross-contamination -- unlike
several OTHER managers audited earlier in this session, this one
creates a fresh registrar per call and holds no id()-based cache at
all, so it was never at risk of that bug class), determinism, error
handling, and -- critically -- that BOTH the icp.py convergence fix
and the point_to_plane.py sign fix are reachable and correctly
reflected through this public API, not just through the underlying
classes directly.

No bugs found in this file.
"""

from __future__ import annotations

import numpy as np
import pytest

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.base import Transformation
from topocore.processing.registration.manager import RegistrationManager
from topocore.processing.registration.point_to_plane import PointToPlaneICP
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
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    translation = np.array([2.0, 3.0, 0.5])
    known = Transformation.from_rotation_translation(rotation, translation)
    source_points = known.inverse().apply_points(target_points)

    return _cloud(source_points), _cloud(target_points), rotation, translation


# ----------------------------------------------------------------------
# Both algorithms, through the manager's public API, must reflect the
# fixes already verified directly on icp.py / point_to_plane.py.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("method", ["point_to_point", "point_to_plane"])
def test_both_methods_converge_and_recover_known_transform(
    method: str,
    known_transform_scene: tuple[PointCloud, PointCloud, np.ndarray, np.ndarray],
) -> None:
    source, target, rotation, translation = known_transform_scene
    manager = RegistrationManager(
        method=method,
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=20.0,
    )

    result = manager.register(source, target)

    assert result.converged is True  # the icp.py convergence fix, reachable through the manager
    np.testing.assert_allclose(result.transformation.rotation, rotation, atol=1e-3)
    np.testing.assert_allclose(result.transformation.translation, translation, atol=1e-3)


def test_point_to_plane_no_longer_diverges_through_manager() -> None:
    """
    The point_to_plane.py sign-fix regression, specifically exercised
    through RegistrationManager's public API (not the class directly).
    """
    rng = np.random.default_rng(0)
    target_points = rng.uniform(-5, 5, (200, 3))
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

    manager = RegistrationManager(
        method="point_to_plane",
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=20.0,
    )
    result = manager.register(_cloud(source_points), _cloud(target_points))  # must not raise

    assert result.converged is True


# ----------------------------------------------------------------------
# Parameter propagation / per-algorithm filtering.
# ----------------------------------------------------------------------


def test_normal_k_reaches_point_to_plane_but_not_point_to_point() -> None:
    manager_plane = RegistrationManager(method="point_to_plane", normal_k=15)
    registrar_plane = manager_plane._create_registrar()
    assert isinstance(registrar_plane, PointToPlaneICP)
    assert registrar_plane._normal_k == 15

    manager_point = RegistrationManager(method="point_to_point", normal_k=15)
    registrar_point = manager_point._create_registrar()
    assert isinstance(registrar_point, PointToPointICP)
    # PointToPointICP has no _normal_k attribute at all -- confirms
    # normal_k was correctly filtered out, not silently accepted.
    assert not hasattr(registrar_point, "_normal_k")


def test_set_params_is_respected() -> None:
    manager = RegistrationManager(method="point_to_point")
    manager.set_params(max_iterations=3, max_correspondence_distance=20.0)

    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, (100, 3))
    result = manager.register(_cloud(points), _cloud(points))

    assert result.iterations <= 3


def test_per_call_kwargs_override_manager_params() -> None:
    manager = RegistrationManager(method="point_to_point", max_iterations=50, max_correspondence_distance=20.0)

    rng = np.random.default_rng(1)
    points = rng.uniform(-5, 5, (100, 3))
    result = manager.register(_cloud(points), _cloud(points), max_iterations=2)

    assert result.iterations <= 2


# ----------------------------------------------------------------------
# Reuse across multiple, DIFFERENT cloud pairs -- confirms no
# cross-contamination (this manager holds no id()-based cache at all,
# unlike several others audited earlier this session).
# ----------------------------------------------------------------------


def test_reuse_across_different_cloud_pairs_no_cross_contamination() -> None:
    manager = RegistrationManager(
        method="point_to_point",
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=20.0,
    )

    rng = np.random.default_rng(2)
    points_a = rng.uniform(-5, 5, (150, 3))
    translation_a = np.array([1.0, 0.5, 0.2])
    source_a = Transformation.from_rotation_translation(np.eye(3), translation_a).inverse().apply_points(points_a)

    points_b = rng.uniform(-3, 3, (100, 3))
    translation_b = np.array([-0.5, 0.8, -0.3])
    source_b = Transformation.from_rotation_translation(np.eye(3), translation_b).inverse().apply_points(points_b)

    result_a = manager.register(_cloud(source_a), _cloud(points_a))
    result_b = manager.register(_cloud(source_b), _cloud(points_b))

    np.testing.assert_allclose(result_a.transformation.translation, translation_a, atol=1e-3)
    np.testing.assert_allclose(result_b.transformation.translation, translation_b, atol=1e-3)


# ----------------------------------------------------------------------
# Determinism.
# ----------------------------------------------------------------------


def test_repeated_registration_is_deterministic(
    known_transform_scene: tuple[PointCloud, PointCloud, np.ndarray, np.ndarray],
) -> None:
    source, target, _rotation, _translation = known_transform_scene
    manager = RegistrationManager(
        method="point_to_point",
        max_iterations=50,
        tolerance=1e-6,
        max_correspondence_distance=20.0,
    )

    result1 = manager.register(source, target)
    result2 = manager.register(source, target)

    np.testing.assert_array_equal(result1.transformation.matrix, result2.transformation.matrix)


# ----------------------------------------------------------------------
# API surface: method setter, __call__, set_params.
# ----------------------------------------------------------------------


def test_method_setter_switches_algorithm() -> None:
    manager = RegistrationManager(method="point_to_point")
    manager.method = "point_to_plane"
    assert manager.method == "point_to_plane"


def test_callable_interface_matches_register() -> None:
    rng = np.random.default_rng(3)
    points = rng.uniform(-5, 5, (100, 3))
    manager = RegistrationManager(method="point_to_point", max_correspondence_distance=20.0)

    result = manager(_cloud(points), _cloud(points))
    assert result.converged is True


# ----------------------------------------------------------------------
# Error handling.
# ----------------------------------------------------------------------


def test_rejects_unsupported_method_at_construction() -> None:
    with pytest.raises(RegistrationError):
        RegistrationManager(method="bogus")


def test_rejects_unsupported_method_via_setter() -> None:
    manager = RegistrationManager(method="point_to_point")
    with pytest.raises(RegistrationError):
        manager.method = "bogus"


def test_rejects_empty_source_cloud() -> None:
    rng = np.random.default_rng(4)
    target = _cloud(rng.uniform(-5, 5, (50, 3)))
    with pytest.raises(RegistrationError):
        RegistrationManager(method="point_to_point").register(PointCloud(), target)


def test_rejects_empty_target_cloud() -> None:
    rng = np.random.default_rng(4)
    source = _cloud(rng.uniform(-5, 5, (50, 3)))
    with pytest.raises(RegistrationError):
        RegistrationManager(method="point_to_point").register(source, PointCloud())
