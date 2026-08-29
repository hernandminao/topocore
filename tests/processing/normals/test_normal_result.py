"""
Coverage audit tests for topocore.processing.normals.models.NormalResult.

Phase 5.3 (Integration) audit findings:

NormalResult is genuinely referenced by a real consumer
(features/protocols.py, as a typed field on a context-bundle
dataclass: `normals: NormalResult | None = None`) -- confirming the
module's own docstring claim that it exists for "downstream
consumers such as topocore.features", not a speculative/unrealized
design. However, confirmed via grep that `NormalResult(` (an actual
constructor call) has ZERO occurrences anywhere in the codebase,
including in features/protocols.py itself -- the class is referenced
as a type, but not yet populated by any real code path. This is
still treated as genuinely testable public contract, not merely
documented as orphaned: nothing gates its direct construction, and
the module's own docstring explicitly documents the intended
integration pattern (`NormalResult(normals=..., curvature=...)`
built from `NormalManager.estimate_both()`'s own tuple output),
exercised directly below.

Confirmed no duplication or bypass with base.py/pca.py/
weighted_pca.py/manager.py: none of them reference NormalResult at
all, matching the module's own explicit design note that it is kept
deliberately independent from the estimator hierarchy so
`estimate_both()`'s existing tuple return type stays unchanged (no
breaking API change).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import ProcessingError
from topocore.processing.normals.manager import NormalManager
from topocore.processing.normals.models import NormalResult


def _cloud(n: int = 30, seed: int = 0) -> PointCloud:
    rng = np.random.default_rng(seed)
    cloud = PointCloud()
    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Y][:] = rng.uniform(0, 10, n)
    chunk[PointAttribute.Z][:] = np.zeros(n)
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Happy path -- the exact documented integration pattern with
# NormalManager.estimate_both().
# ----------------------------------------------------------------------


def test_documented_integration_pattern_with_normal_manager() -> None:
    normal_manager = NormalManager(method="pca", k=5)
    normals, curvature = normal_manager.estimate_both(_cloud())

    result = NormalResult(normals=normals, curvature=curvature)

    assert result.point_count == 30
    assert result.has_curvature is True
    np.testing.assert_array_equal(result.normals, normals)
    np.testing.assert_array_equal(result.curvature, curvature)


def test_curvature_defaults_to_none() -> None:
    normals = np.zeros((10, 3))

    result = NormalResult(normals=normals)

    assert result.curvature is None
    assert result.has_curvature is False


# ----------------------------------------------------------------------
# __post_init__ validation.
# ----------------------------------------------------------------------


def test_non_2d_normals_rejected() -> None:
    with pytest.raises(ProcessingError, match="2D array"):
        NormalResult(normals=np.zeros(10))


def test_wrong_column_count_rejected() -> None:
    with pytest.raises(ProcessingError, match=r"shape \(n,3\)"):
        NormalResult(normals=np.zeros((10, 2)))


def test_curvature_size_mismatch_rejected() -> None:
    with pytest.raises(ProcessingError, match="does not match normals"):
        NormalResult(normals=np.zeros((10, 3)), curvature=np.zeros(5))


# ----------------------------------------------------------------------
# Immutability.
# ----------------------------------------------------------------------


def test_result_is_immutable() -> None:
    result = NormalResult(normals=np.zeros((5, 3)))

    with pytest.raises(FrozenInstanceError):
        result.normals = np.zeros((10, 3))  # type: ignore[misc]
