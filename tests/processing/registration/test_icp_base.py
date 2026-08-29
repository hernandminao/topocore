"""
Coverage audit tests for topocore.processing.registration.icp.ICPBase.

ICPBase is abstract (`_estimate_transformation` is an abstractmethod),
so a minimal concrete test double (_IdentityICP, always returning
Transformation.identity()) is used to exercise this class's own
loop/validation logic directly -- the same approach already used for
MachineLearningClassifier's _FakeModel elsewhere in this audit.

Confirmed via direct execution: Chunk.__init__ already requires X/Y/Z
at construction (same finding as processing._shared's
_validate_cloud_attributes audit) -- meaning _validate_inputs()'s own
"must contain X, Y, and Z" checks (for both source and target) are
unreachable: no PointCloud can ever lack these attributes, since
every constituent Chunk already enforces them. NOT tested here by
design.

name() ("icp_base") and requires_normals() (False) are documented as
orphaned -- zero external callers confirmed via grep, consistent with
the same policy already applied to TreeSegmenter.name,
PCANormalEstimator.name/requires_k/supports_weighted, etc. elsewhere
in this audit. Not tested here.

Separate finding, NOT resolved here (a performance/architecture
concern, not a coverage gap): _compute_fitness() contains the same
per-point query_point() loop pattern already found and fixed
elsewhere for this exact class of issue in PR21.8 -- confirmed
directly that replacing it with query_points_many() gives numerically
identical results. This was missed by PR21.8's own transversal audit
(which searched for `for ... in range()`/`for ... in enumerate()`
patterns; this one uses `for point in source_points:`, a different
loop-variable style). Flagged for a separate decision, not fixed as
part of this coverage audit.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError
from topocore.processing.registration.base import Transformation
from topocore.processing.registration.icp import ICPBase


class _IdentityICP(ICPBase):
    """Minimal concrete test double -- always estimates the identity transformation."""

    def _estimate_transformation(self, correspondences, source_points, target_points):  # type: ignore[no-untyped-def]
        return Transformation.identity()


def _cloud(n: int, offset: tuple[float, float, float] = (0.0, 0.0, 0.0), seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    x = rng.uniform(0, 20, n)
    y = rng.uniform(0, 20, n)
    z = rng.uniform(0, 20, n)
    chunk[PointAttribute.X][:] = x + offset[0]
    chunk[PointAttribute.Y][:] = y + offset[1]
    chunk[PointAttribute.Z][:] = z + offset[2]
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Constructor validation.
# ----------------------------------------------------------------------


def test_max_iterations_less_than_one_rejected() -> None:
    with pytest.raises(RegistrationError, match="max_iterations must be"):
        _IdentityICP(max_iterations=0)


def test_negative_tolerance_rejected() -> None:
    with pytest.raises(RegistrationError, match="tolerance must be"):
        _IdentityICP(tolerance=-1.0)


def test_non_positive_max_correspondence_distance_rejected() -> None:
    with pytest.raises(RegistrationError, match="max_correspondence_distance must be"):
        _IdentityICP(max_correspondence_distance=0.0)


# ----------------------------------------------------------------------
# _validate_inputs() -- empty clouds (reachable; missing-X/Y/Z branches
# are unreachable given Chunk's own construction contract, see module docstring).
# ----------------------------------------------------------------------


def test_empty_source_cloud_rejected() -> None:
    icp = _IdentityICP()
    target = _cloud(10)

    with pytest.raises(RegistrationError, match="Source point cloud is empty"):
        icp.register(PointCloud(), target)


def test_empty_target_cloud_rejected() -> None:
    icp = _IdentityICP()
    source = _cloud(10)

    with pytest.raises(RegistrationError, match="Target point cloud is empty"):
        icp.register(source, PointCloud())


# ----------------------------------------------------------------------
# register() -- not enough correspondences.
# ----------------------------------------------------------------------


def test_not_enough_correspondences_rejected() -> None:
    icp = _IdentityICP(max_correspondence_distance=1.0, use_adaptive_distance=False)
    source = _cloud(10, offset=(0.0, 0.0, 0.0), seed=0)
    target = _cloud(10, offset=(1000.0, 1000.0, 1000.0), seed=1)

    with pytest.raises(RegistrationError, match="Not enough correspondences"):
        icp.register(source, target)


# ----------------------------------------------------------------------
# register() -- happy path.
# ----------------------------------------------------------------------


def test_register_happy_path_returns_valid_result() -> None:
    target = _cloud(50, seed=1)
    source = _cloud(50, offset=(0.2, 0.1, -0.1), seed=1)  # small offset, same underlying points

    icp = _IdentityICP(max_iterations=5)
    result = icp.register(source, target)

    assert result.has_source_transformed is True
    assert result.source_transformed is not None
    assert result.source_transformed.point_count == 50
    assert result.iterations > 0
    assert 0.0 <= result.fitness <= 1.0
    assert result.rmse >= 0.0


# ----------------------------------------------------------------------
# _compute_rmse() -- direct contract test for the empty-correspondences case.
# ----------------------------------------------------------------------


def test_compute_rmse_of_empty_correspondences_is_infinite() -> None:
    icp = _IdentityICP()

    assert icp._compute_rmse([]) == float("inf")
