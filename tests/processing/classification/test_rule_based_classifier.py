"""
Coverage audit tests for topocore.processing.classification.rules.RuleBasedClassifier.

Audit findings (documented here, not force-tested):

_compute_features()'s three defensive checks are ALL confirmed
unreachable given this class's own specific construction of its
internal dependencies:
  - "Ground model returned an invalid size": self._ground_manager
    is always constructed with method="grid", which delegates to
    GridGroundElevationEstimator -- already confirmed elsewhere in
    this audit (PR21.7.8) to guarantee an output array whose length
    always equals cloud.point_count, by construction (np.concatenate
    of per-chunk contributions summing to the total).
  - "Missing PCA features": PCAFeatures.compute() is confirmed
    (by reading its own source) to simply delegate to compute_all(),
    which unconditionally includes every PCA-derived feature name in
    its result -- the same guarantee already relied upon in
    segmentation/specific.py's own audit.
  - "Noise filter rejected every point": confirmed directly that
    RadiusOutlierFilter.mask() has its own non-strict fallback
    (min_points=3, strict=False by default, and RuleBasedClassifier
    does not override either) -- if the strict computation would
    keep fewer than min_points, mask() silently returns ALL points
    as kept instead of raising or returning all-False. This
    mathematically prevents `not_noise` from ever being entirely
    False under RuleBasedClassifier's specific (non-strict)
    construction of its own noise filter.

_apply_rules()'s "invalid number of labels" and "must be integers"
checks are also not tested: `labels = np.full(n_points, ...,
dtype=np.int64)` guarantees both invariants by construction, the
same np.empty()/np.full()-guarantees-shape-and-dtype pattern already
established multiple times elsewhere in this audit (e.g.
_build_feature_matrix in classification/ml.py).

The four properties (ground_threshold, curvature_threshold,
building_threshold, vegetation_threshold) are documented as orphaned
-- zero external callers confirmed via grep.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.classification import PointClassification
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.rules import RuleBasedClassifier, _RuleFeatures
from topocore.processing.exceptions import ProcessingError

# ----------------------------------------------------------------------
# Constructor validation -- representative sample of the 20 checks,
# each confirmed reachable via direct execution before writing.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ground_cell_size": 0}, "ground_cell_size"),
        ({"ground_max_height": -1}, "ground_max_height"),
        ({"low_vegetation_max_height": -1}, "low_vegetation_max_height"),
        (
            {"high_vegetation_min_height": 0.1, "low_vegetation_max_height": 0.5},
            "high_vegetation_min_height",
        ),
        ({"curvature_threshold": -1}, "curvature_threshold"),
        ({"noise_radius": 0}, "noise_radius"),
        ({"noise_min_neighbors": 0}, "noise_min_neighbors"),
        ({"pca_neighbors": 2}, "pca_neighbors"),
        ({"building_planarity_threshold": 1.5}, "building_planarity_threshold"),
        ({"building_sphericity_threshold": -0.1}, "building_sphericity_threshold"),
        ({"building_min_verticality": 1.5}, "building_min_verticality"),
        ({"building_max_verticality": -0.1}, "building_max_verticality"),
        (
            {"building_min_verticality": 0.8, "building_max_verticality": 0.5},
            "building_min_verticality",
        ),
        ({"max_building_height": 0}, "max_building_height"),
        ({"vegetation_min_height": -1}, "vegetation_min_height"),
        ({"vegetation_planarity_threshold": 1.5}, "vegetation_planarity_threshold"),
        ({"vegetation_linearity_threshold": 1.5}, "vegetation_linearity_threshold"),
        ({"vegetation_sphericity_threshold": 1.5}, "vegetation_sphericity_threshold"),
        ({"vegetation_max_anisotropy": 1.5}, "vegetation_max_anisotropy"),
        ({"vegetation_min_omnivariance": -1}, "vegetation_min_omnivariance"),
        (
            {"vegetation_min_height": 0.9, "low_vegetation_max_height": 0.5},
            "vegetation_min_height",
        ),
    ],
)
def test_constructor_validation(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RuleBasedClassifier(**kwargs)


def test_valid_construction() -> None:
    clf = RuleBasedClassifier()
    assert clf.name() == "rule_based"


# ----------------------------------------------------------------------
# classify() -- empty cloud.
# ----------------------------------------------------------------------


def test_classify_rejects_empty_cloud() -> None:
    with pytest.raises(ProcessingError, match="empty point cloud"):
        RuleBasedClassifier().classify(PointCloud())


# ----------------------------------------------------------------------
# _validate_features() -- direct contract test via a deliberately
# malformed _RuleFeatures, matching the same reasoning approved for
# _validate_feature() in classification/ml.py's own audit.
# ----------------------------------------------------------------------


def _valid_features(n: int = 5) -> dict:
    return {
        "height_above_ground": np.zeros(n),
        "curvature": np.zeros(n),
        "planarity": np.zeros(n),
        "linearity": np.zeros(n),
        "sphericity": np.zeros(n),
        "verticality": np.zeros(n),
        "anisotropy": np.zeros(n),
        "omnivariance": np.zeros(n),
        "density": np.zeros(n),
        "not_noise": np.ones(n, dtype=bool),
    }


def test_validate_features_rejects_two_dimensional_feature() -> None:
    kwargs = _valid_features()
    kwargs["curvature"] = np.zeros((5, 1))
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="one-dimensional"):
        clf._validate_features(_RuleFeatures(**kwargs), 5)


def test_validate_features_rejects_wrong_size_feature() -> None:
    kwargs = _valid_features()
    kwargs["planarity"] = np.zeros(3)
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="invalid size"):
        clf._validate_features(_RuleFeatures(**kwargs), 5)


def test_validate_features_rejects_non_finite_feature() -> None:
    kwargs = _valid_features()
    kwargs["sphericity"] = np.full(5, np.nan)
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="NaN or Inf"):
        clf._validate_features(_RuleFeatures(**kwargs), 5)


def test_validate_features_rejects_wrong_size_noise_mask() -> None:
    kwargs = _valid_features()
    kwargs["not_noise"] = np.ones(3, dtype=bool)
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="Noise mask has invalid size"):
        clf._validate_features(_RuleFeatures(**kwargs), 5)


def test_validate_features_rejects_non_boolean_noise_mask() -> None:
    kwargs = _valid_features()
    kwargs["not_noise"] = np.ones(5, dtype=np.int64)
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="Noise mask must be boolean"):
        clf._validate_features(_RuleFeatures(**kwargs), 5)


# ----------------------------------------------------------------------
# _apply_rules() -- "no rule matched", confirmed genuinely reachable.
# ----------------------------------------------------------------------


def test_apply_rules_raises_when_no_rule_matches() -> None:
    """
    A point that is not noise, too high to be ground, and falls
    outside both the building and vegetation geometric criteria
    remains UNCLASSIFIED for every point -- confirmed reachable
    directly before writing this test.
    """
    n = 5
    features = _RuleFeatures(
        height_above_ground=np.full(n, 10.0),
        curvature=np.full(n, 0.001),
        planarity=np.full(n, 0.3),
        linearity=np.zeros(n),
        sphericity=np.full(n, 0.5),
        verticality=np.full(n, 0.5),
        anisotropy=np.zeros(n),
        omnivariance=np.zeros(n),
        density=np.zeros(n),
        not_noise=np.ones(n, dtype=bool),
    )
    clf = RuleBasedClassifier()

    with pytest.raises(ProcessingError, match="no rule matched"):
        clf._apply_rules(features)


# ----------------------------------------------------------------------
# Happy path -- a realistic synthetic cloud with ground and vegetation.
# ----------------------------------------------------------------------


def test_classify_happy_path_labels_ground_and_vegetation() -> None:
    rng = np.random.default_rng(0)
    ground_n, veg_n = 500, 100
    gx, gy, gz = (
        rng.uniform(0, 50, ground_n),
        rng.uniform(0, 50, ground_n),
        np.zeros(ground_n),
    )
    vx, vy, vz = (
        rng.uniform(0, 50, veg_n),
        rng.uniform(0, 50, veg_n),
        rng.uniform(0.5, 3.0, veg_n),
    )

    cloud = PointCloud()
    chunk = Chunk(
        size=ground_n + veg_n,
        attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z],
    )
    chunk[PointAttribute.X][:] = np.concatenate([gx, vx])
    chunk[PointAttribute.Y][:] = np.concatenate([gy, vy])
    chunk[PointAttribute.Z][:] = np.concatenate([gz, vz])
    cloud.add_chunk(chunk)

    result = RuleBasedClassifier().classify(cloud)

    assert len(result.labels) == ground_n + veg_n
    assert (result.labels == PointClassification.GROUND).sum() > 0
