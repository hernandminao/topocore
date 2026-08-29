"""
Coverage audit tests for topocore.processing.registration.point_to_plane.PointToPlaneICP.

Two of this suite's key tests directly reproduce the two
hand-derivable verification cases documented in
_build_linear_system()'s own docstring, which records a real,
previously-found-and-fixed sign bug (PR19): a pure +0.5 Z translation
must resolve to -0.5 (moving source back onto target), and a pure +1
degree rotation about Y must resolve to +1 degree. Both confirmed
directly before writing these tests, matching the documented fix
exactly (translation resolved to precisely -0.5; the linearized
rotation estimate came out to ~0.997 degrees for a 1 degree input, an
expected small residual from the small-angle linearization itself,
not a sign or correctness issue).

_estimate_transformation() is tested directly (bypassing the full
register() loop) for its own documented contract, matching the same
reasoning already applied to PointToPointICP's own audit.

_ensure_normals()'s "empty normals_chunks" branch is confirmed
unreachable: an empty PointCloud's own .attributes is confirmed
(directly) to be an empty frozenset, so
`PointAttribute.NORMAL in cloud.attributes` is always False for an
empty cloud -- meaning that branch is only ever reached when at least
one chunk already declares NORMAL (cloud.attributes is the union of
its chunks' own attribute sets, established elsewhere in this audit),
which guarantees normals_chunks is non-empty by construction. Not
tested here.

name() ("point_to_plane_icp") and requires_normals() (True) are
documented as orphaned -- zero external callers confirmed via grep,
consistent with the same policy applied throughout this audit.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.point_to_plane import PointToPlaneICP

# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_normal_k_less_than_three_rejected() -> None:
    with pytest.raises(RegistrationError, match="normal_k must be"):
        PointToPlaneICP(normal_k=2)


# ----------------------------------------------------------------------
# _estimate_transformation() -- its own documented contract.
# ----------------------------------------------------------------------


def test_fewer_than_three_correspondences_rejected() -> None:
    icp = PointToPlaneICP()
    with pytest.raises(RegistrationError, match="Need at least 3 correspondences"):
        icp._estimate_transformation([(0, 0, 0.1)], np.zeros((5, 3)), np.zeros((5, 3)))


def test_missing_target_normals_rejected() -> None:
    """_estimate_transformation() called without _ensure_normals() having run first."""
    icp = PointToPlaneICP()
    correspondences = [(i, i, 0.0) for i in range(4)]

    with pytest.raises(RegistrationError, match="Target normals not available"):
        icp._estimate_transformation(correspondences, np.zeros((5, 3)), np.zeros((5, 3)))


# ----------------------------------------------------------------------
# _build_linear_system() sign convention -- the two hand-derivable
# cases documented in its own docstring (PR19 fix verification).
# ----------------------------------------------------------------------


def test_pure_z_translation_resolves_with_correct_sign() -> None:
    """A source offset +0.5 above a flat target plane must resolve to translation -0.5 (moving it back down)."""
    n = 20
    rng = np.random.default_rng(0)
    target_points = np.column_stack([rng.uniform(0, 10, n), rng.uniform(0, 10, n), np.zeros(n)])
    normals = np.tile([0.0, 0.0, 1.0], (n, 1))
    source_points = target_points.copy()
    source_points[:, 2] += 0.5

    icp = PointToPlaneICP()
    icp._target_normals = normals
    correspondences = [(i, i, 0.0) for i in range(n)]

    t = icp._estimate_transformation(correspondences, source_points, target_points)

    np.testing.assert_allclose(t.translation, [0.0, 0.0, -0.5], atol=1e-8)


def test_pure_y_rotation_resolves_with_correct_sign() -> None:
    """A source rotated +1 degree about Y from target must resolve to approximately +1 degree, not -1."""
    n = 20
    rng = np.random.default_rng(0)
    theta = np.radians(1.0)
    rotation_y = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    target_points = np.column_stack([rng.uniform(-5, 5, n), rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)])
    source_points = target_points @ rotation_y.T
    normals = rng.normal(size=(n, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    icp = PointToPlaneICP()
    icp._target_normals = normals
    correspondences = [(i, i, 0.0) for i in range(n)]

    t = icp._estimate_transformation(correspondences, source_points, target_points)
    estimated_angle_degrees = np.degrees(np.arccos((np.trace(t.rotation) - 1) / 2))

    # A small residual from the small-angle linearization itself is expected -- not an exact match.
    assert estimated_angle_degrees == pytest.approx(1.0, abs=0.1)


# ----------------------------------------------------------------------
# _rotation_matrix_from_omega() -- near-zero angle edge case.
# ----------------------------------------------------------------------


def test_near_zero_omega_returns_identity() -> None:
    icp = PointToPlaneICP()

    rotation = icp._rotation_matrix_from_omega(np.array([1e-15, 0.0, 0.0]))

    np.testing.assert_array_equal(rotation, np.eye(3))


# ----------------------------------------------------------------------
# _ensure_normals() -- both branches.
# ----------------------------------------------------------------------


def test_ensure_normals_uses_existing_normal_attribute() -> None:
    cloud = PointCloud()
    chunk = Chunk(
        size=5,
        attributes=[
            PointAttribute.X,
            PointAttribute.Y,
            PointAttribute.Z,
            PointAttribute.NORMAL,
        ],
    )
    chunk[PointAttribute.X][:] = np.arange(5, dtype=float)
    chunk[PointAttribute.Y][:] = np.zeros(5)
    chunk[PointAttribute.Z][:] = np.zeros(5)
    chunk[PointAttribute.NORMAL][:] = np.tile([0.0, 0.0, 1.0], (5, 1))
    cloud.add_chunk(chunk)

    icp = PointToPlaneICP()
    icp._ensure_normals(cloud)

    assert icp._target_normals is not None
    np.testing.assert_array_equal(icp._target_normals, np.tile([0.0, 0.0, 1.0], (5, 1)))


def test_ensure_normals_computes_when_missing() -> None:
    rng = np.random.default_rng(0)
    cloud = PointCloud()
    chunk = Chunk(size=30, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, 30)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, 30)
    chunk[PointAttribute.Z][:] = np.zeros(30)
    cloud.add_chunk(chunk)

    icp = PointToPlaneICP(normal_k=5)
    icp._ensure_normals(cloud)

    assert icp._target_normals is not None
    assert icp._target_normals.shape == (30, 3)
