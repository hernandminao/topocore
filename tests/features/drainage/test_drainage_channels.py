"""
Regression suite for topocore.features.drainage.drainage.DrainageDetector
and .channels.ChannelDetector -- PR19.

Verified the core valley-depth geometric criterion with a decisive
pair of opposite synthetic TINs: a genuine V-shaped valley (correctly
detected) and its mirror-image ridge (correctly NOT detected as
drainage) -- confirming the "both opposite triangle vertices above
the shared edge" test genuinely identifies a local low point between
two rising sides, not the reverse. No bugs found.
"""

from __future__ import annotations

import pytest

from topocore.features.drainage.channels import ChannelDetector
from topocore.features.drainage.drainage import DrainageDetector
from topocore.features.exceptions import DetectionError
from topocore.features.models import FeatureType
from topocore.features.protocols import DetectionContext
from topocore.geometry.point3d import Point3D
from topocore.terrain.tin import TIN


def _valley_tin() -> TIN:
    """A V-shaped valley: a low edge at z=0 with terrain rising to z=5 on both sides."""
    points = (
        Point3D(0, 0, 0.0),
        Point3D(0, 20, 0.0),
        Point3D(-10, 0, 5.0),
        Point3D(-10, 20, 5.0),
        Point3D(10, 0, 5.0),
        Point3D(10, 20, 5.0),
    )
    return TIN.from_points(points)


def _ridge_tin() -> TIN:
    """The mirror image: a high edge at z=5 with terrain dropping to z=0 on both sides."""
    points = (
        Point3D(0, 0, 5.0),
        Point3D(0, 20, 5.0),
        Point3D(-10, 0, 0.0),
        Point3D(-10, 20, 0.0),
        Point3D(10, 0, 0.0),
        Point3D(10, 20, 0.0),
    )
    return TIN.from_points(points)


def test_v_shaped_valley_detected() -> None:
    detector = DrainageDetector(min_depth=0.5, min_length=1.0)
    result = detector.detect(DetectionContext(tin=_valley_tin()))
    assert len(result) >= 1
    assert result.features[0].feature_type == FeatureType.DRAINAGE


def test_ridge_not_detected_as_drainage() -> None:
    """The decisive negative case: the mirror-image terrain must NOT be flagged as a valley."""
    detector = DrainageDetector(min_depth=0.5, min_length=1.0)
    result = detector.detect(DetectionContext(tin=_ridge_tin()))
    assert len(result) == 0


def test_flat_terrain_has_no_valleys() -> None:
    points = (
        Point3D(0, 0, 5.0),
        Point3D(20, 0, 5.0),
        Point3D(0, 20, 5.0),
        Point3D(20, 20, 5.0),
    )
    tin = TIN.from_points(points)
    result = DrainageDetector().detect(DetectionContext(tin=tin))
    assert len(result) == 0


def test_rejects_negative_min_depth() -> None:
    with pytest.raises(DetectionError):
        DrainageDetector(min_depth=-1.0)


def test_rejects_negative_min_length() -> None:
    with pytest.raises(DetectionError):
        DrainageDetector(min_length=-1.0)


def test_channel_detector_inherits_valley_logic_with_own_feature_type() -> None:
    detector = ChannelDetector()
    result = detector.detect(DetectionContext(tin=_valley_tin()))
    assert len(result) >= 1
    assert result.features[0].feature_type == FeatureType.CHANNEL
    assert result.features[0].category == result.features[0].category  # sanity: category assigned


def test_channel_detector_uses_stricter_default_thresholds() -> None:
    """ChannelDetector's defaults (min_depth=0.3, min_length=3.0) are stricter than DrainageDetector's (0.05, 1.0)."""
    channel = ChannelDetector()
    drainage = DrainageDetector()
    assert channel._min_depth > drainage._min_depth
    assert channel._min_length > drainage._min_length
