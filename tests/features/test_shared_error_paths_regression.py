"""
Regression tests for 2 real defects found and fixed during this
project's own feature-extraction documentation audit, both in
`topocore.features._shared`, both the same underlying pattern (an
external, non-`features` exception escaping this package's own
domain-exception boundary):

1. `cluster_points_2d()` and `convex_hull_polygon()` used to let a
   non-finite (NaN/Inf) X/Y coordinate reach
   `scipy.spatial.cKDTree`/`ConvexHull` directly, raising a plain
   `ValueError`. Confirmed reachable in the real production
   pipeline, not a synthetic edge case: neither `PointCloud`/`Chunk`
   nor `Workflow.detect_features()` validates or guarantees finite
   coordinates anywhere. Affects 15 of the 22 registered detectors.

2. `convex_hull_polygon()`'s own existing fallback covered fewer
   than 3 distinct XY positions, but not 3-or-more distinct, exactly
   collinear positions (a real, reachable case for a naturally
   linear feature, e.g. a fence or narrow wall) -- this instead
   reached `scipy.spatial.ConvexHull` directly and raised a raw
   `QhullError`. Confirmed reachable from a real registered detector
   (`BuildingDetector`). Fixed with a dedicated collinear "sliver"
   fallback (not reusing the `< 3` points fallback's own tiny fixed
   offset) specifically to preserve the cluster's own real spatial
   extent into the resulting `Feature`'s own `bounds`.

`DetectionError` was confirmed the correct exception, for both, by
real precedent already existing within this same package, not
invented for this fix:
- `topocore.features.drainage.DrainageDetector` already raises
  `DetectionError("TIN vertices must contain only finite
  coordinates.")` for the identical finite-coordinate condition on
  its own TIN input.
- `topocore.features.terrain.contours.ContourDetector` already wraps
  an underlying failure into `DetectionError(...) from exc`.

`GeometryError` was confirmed NOT the right choice for either: it is
raised exclusively by `topocore.features.models` itself, to validate
an already-constructed output geometry -- never by a detector's own
input-side validation or by a shared utility's own numerical
preconditions.

Collinearity is detected the same way
`terrain.algorithms.delaunay.DelaunayTriangulator
._validate_collinear_points()` already does elsewhere in this
project (rank of the mean-centered coordinates), for consistency
with that existing convention rather than inventing a new one.
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


def test_convex_hull_polygon_collinear_points_no_longer_raises() -> None:
    """
    The second real defect found and fixed in this same audit:
    3+ distinct but collinear points used to reach
    scipy.spatial.ConvexHull directly and raise a raw QhullError.
    """
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
        ]
    )

    result = convex_hull_polygon(xyz)

    assert result.shape[0] >= 3


def test_convex_hull_polygon_collinear_points_preserve_real_extent() -> None:
    """
    The specific reason a sliver fallback was used instead of
    reusing the tiny fixed-size <3-points fallback: a genuinely long
    collinear cluster's own real spatial extent must survive into
    the resulting geometry's own bounds, not collapse to ~0.
    """
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [4.0, 0.0, 1.0],
        ]
    )

    result = convex_hull_polygon(xyz)

    x_extent = result[:, 0].max() - result[:, 0].min()
    assert x_extent == pytest.approx(4.0)


def test_building_detector_collinear_points_no_longer_raises_qhull_error() -> None:
    """
    The exact real-pipeline reproduction: BuildingDetector (a real
    registered detector, geometry_type=POLYGON) given a straight
    line of building-classified points -- a realistic case (a narrow
    or sparsely-sampled structure), not a synthetic corner case.
    """
    from topocore.features.buildings.buildings import BuildingDetector
    from topocore.features.protocols import DetectionContext
    from topocore.pointcloud.attributes import PointAttribute
    from topocore.pointcloud.chunk import Chunk
    from topocore.pointcloud.classification import PointClassification
    from topocore.pointcloud.pointcloud import PointCloud
    from topocore.processing.classification.base import ClassificationResult

    n = 10
    xs = np.linspace(0, 9, n)
    ys = np.zeros(n)
    zs = np.full(n, 2.0)

    chunk = Chunk(size=n, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = xs
    chunk[PointAttribute.Y][:] = ys
    chunk[PointAttribute.Z][:] = zs
    cloud = PointCloud()
    cloud.add_chunk(chunk)

    labels = np.full(n, PointClassification.BUILDING.value, dtype=np.int64)
    classification = ClassificationResult(labels=labels, cloud=cloud)

    result = BuildingDetector(eps=2.0, min_points=5).detect(
        DetectionContext(cloud=cloud, classification=classification)
    )

    features = list(result)
    assert len(features) == 1

    bounds = features[0].geometry.bounds
    x_extent = bounds[3] - bounds[0]
    assert x_extent == pytest.approx(9.0)


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


def test_extract_xyz_empty_cloud_returns_empty_array_not_value_error() -> None:
    """
    The third real defect found and fixed in this same audit:
    `extract_xyz()` used to call `np.concatenate()` unconditionally,
    raising a raw `ValueError` for a genuinely empty `PointCloud` --
    a valid input (e.g. after an earlier pipeline stage filtered
    everything out of a tile), not an error condition.
    """
    from topocore.features._shared import extract_xyz
    from topocore.pointcloud.pointcloud import PointCloud

    result = extract_xyz(PointCloud())

    assert result.shape == (0, 3)


def test_end_to_end_detector_empty_cloud_returns_empty_collection() -> None:
    """
    The exact real-pipeline reproduction: a genuinely empty
    PointCloud, run through a real registered detector -- confirms
    the fix propagates correctly all the way to "no features found"
    rather than raising anywhere downstream either.
    """
    from topocore.features.protocols import DetectionContext
    from topocore.features.vegetation.trees import TreeDetector
    from topocore.pointcloud.pointcloud import PointCloud
    from topocore.processing.classification.base import ClassificationResult

    cloud = PointCloud()
    classification = ClassificationResult(labels=np.array([], dtype=np.int64), cloud=cloud)

    result = TreeDetector().detect(DetectionContext(cloud=cloud, classification=classification))

    assert len(list(result)) == 0
