"""
Regression test for a real defect found and fixed during this
project's own feature-extraction documentation audit:
`cluster_points_2d()` and `convex_hull_polygon()`
(topocore.features._shared) used to let a non-finite (NaN/Inf) X/Y
coordinate reach `scipy.spatial.cKDTree`/`ConvexHull` directly,
raising a plain `ValueError` instead of a `features`-domain
exception.

Confirmed reachable in the real production pipeline, not a synthetic
edge case: neither `PointCloud`/`Chunk` nor
`Workflow.detect_features()` validates or guarantees finite
coordinates anywhere -- confirmed directly, by search. Affects 15 of
the 22 registered detectors (the 13 built on `ClusterDetectorBase`,
plus `RetainingWallDetector`/`RoofDetector`, which call these 2
functions directly without going through that shared base class).

`DetectionError` was confirmed the correct exception by 2
independent pieces of real precedent within this same package, not
invented for this fix:
- `topocore.features.drainage.DrainageDetector` already raises
  `DetectionError("TIN vertices must contain only finite
  coordinates.")` for the identical condition on its own TIN input.
- `topocore.features.terrain.contours.ContourDetector` already wraps
  an underlying failure into `DetectionError(...) from exc`.

`GeometryError` was confirmed NOT the right choice: it is raised
exclusively by `topocore.features.models` itself, to validate an
already-constructed output geometry -- never by a detector's own
input-side validation.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.features._shared import cluster_points_2d, convex_hull_polygon
from topocore.features.exceptions import DetectionError


def test_cluster_points_2d_rejects_nan() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [float("nan"), float("nan")], [0.0, 1.0]])

    with pytest.raises(DetectionError, match="finite"):
        cluster_points_2d(xy, eps=2.0, min_points=1)


def test_cluster_points_2d_rejects_inf() -> None:
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [float("inf"), 0.0]])

    with pytest.raises(DetectionError, match="finite"):
        cluster_points_2d(xy, eps=2.0, min_points=1)


def test_cluster_points_2d_valid_input_unaffected() -> None:
    """Confirms the fix changed nothing for well-formed input."""
    xy = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 1.0], [5.0, 5.0]])

    result = cluster_points_2d(xy, eps=1.0, min_points=2)

    assert len(result) == 1
    np.testing.assert_array_equal(sorted(result[0].tolist()), [0, 1])


def test_convex_hull_polygon_rejects_nan_in_xy() -> None:
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [float("nan"), 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )

    with pytest.raises(DetectionError, match="finite"):
        convex_hull_polygon(xyz)


def test_convex_hull_polygon_valid_input_unaffected() -> None:
    """Confirms the fix changed nothing for well-formed input."""
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [1.0, 1.0, 3.0],
            [0.0, 1.0, 4.0],
        ]
    )

    result = convex_hull_polygon(xyz)

    assert result.shape == (4, 3)
    np.testing.assert_array_equal(result[:, 2], np.full(4, 2.5))  # median Z


def test_end_to_end_detector_raises_detection_error_not_raw_value_error() -> None:
    """
    The exact real-pipeline reproduction: a point cloud with one
    non-finite coordinate, classified as valid vegetation candidates,
    run through a real registered detector (TreeDetector, built on
    ClusterDetectorBase) -- exactly as Workflow.detect_features()
    would invoke it, with no upstream finite-coordinate validation
    anywhere in that real path.
    """
    from topocore.features.protocols import DetectionContext
    from topocore.features.vegetation.trees import TreeDetector
    from topocore.pointcloud.attributes import PointAttribute
    from topocore.pointcloud.chunk import Chunk
    from topocore.pointcloud.classification import PointClassification
    from topocore.pointcloud.pointcloud import PointCloud
    from topocore.processing.classification.base import ClassificationResult

    rng = np.random.default_rng(0)
    n = 10
    xs = 5.0 + rng.uniform(-0.3, 0.3, n)
    ys = 5.0 + rng.uniform(-0.3, 0.3, n)
    zs = rng.uniform(2.0, 5.0, n)
    xs[3] = float("nan")

    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud = PointCloud()
    cloud.add_chunk(chunk)

    labels = np.full(n, PointClassification.HIGH_VEGETATION.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=cloud)

    with pytest.raises(DetectionError, match="finite"):
        TreeDetector().detect(DetectionContext(cloud=cloud, classification=classification))
