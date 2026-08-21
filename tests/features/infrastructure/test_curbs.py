"""
Regression suite for topocore.features.infrastructure.curbs.CurbDetector
-- PR19.

Verified a real, necessary design detail: BreaklineDetector._detect()
(CurbDetector's parent) hardcodes feature_type=FeatureType.BREAKLINE
as a literal, NOT self.feature_type -- so CurbDetector's own explicit
replace(feature, category=self.category, feature_type=self.feature_type)
rewrite after calling super()._detect() is genuinely required, not
redundant, to get feature_type=CURB in the final output instead of
BREAKLINE. Confirmed directly with real geometry (a small dihedral
step) that the final Features are correctly stamped CURB/INFRASTRUCTURE,
not BREAKLINE/TERRAIN. No bugs found.
"""

from __future__ import annotations

from topocore.features.infrastructure.curbs import CurbDetector
from topocore.features.models import FeatureCategory, FeatureType
from topocore.features.protocols import DetectionContext
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


def test_curb_detected_with_correct_type_not_breakline() -> None:
    """
    The decisive check: without CurbDetector's replace() rewrite,
    these features would be reported as FeatureType.BREAKLINE
    (BreaklineDetector's hardcoded literal) instead of CURB.
    """
    points = (
        Point3D(0, -10, 0.0),
        Point3D(0, 10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(10, 10, 0.0),
        Point3D(10.1, -10, 0.15),
        Point3D(10.1, 10, 0.15),
        Point3D(20, -10, 0.15),
        Point3D(20, 10, 0.15),
    )
    tin = TIN.from_points(points)

    detector = CurbDetector(angle_threshold_deg=5.0, min_length=0.1)
    result = detector.detect(DetectionContext(tin=tin))

    assert len(result) > 0
    for feature in result.features:
        assert feature.feature_type == FeatureType.CURB
        assert feature.feature_type != FeatureType.BREAKLINE
        assert feature.category == FeatureCategory.INFRASTRUCTURE


def test_curb_feature_ids_preserved_through_replace() -> None:
    points = (
        Point3D(0, -10, 0.0),
        Point3D(0, 10, 0.0),
        Point3D(10, -10, 0.0),
        Point3D(10, 10, 0.0),
        Point3D(10.1, -10, 0.15),
        Point3D(10.1, 10, 0.15),
        Point3D(20, -10, 0.15),
        Point3D(20, 10, 0.15),
    )
    tin = TIN.from_points(points)

    detector = CurbDetector(angle_threshold_deg=5.0, min_length=0.1)
    result = detector.detect(DetectionContext(tin=tin))

    ids = [f.feature_id for f in result.features]
    assert ids == list(range(1, len(ids) + 1))


def test_flat_terrain_has_no_curbs() -> None:
    points = (
        Point3D(0, 0, 0.0),
        Point3D(20, 0, 0.0),
        Point3D(0, 20, 0.0),
        Point3D(20, 20, 0.0),
    )
    tin = TIN.from_points(points)

    detector = CurbDetector()
    result = detector.detect(DetectionContext(tin=tin))
    assert len(result) == 0
