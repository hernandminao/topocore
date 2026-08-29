"""
Coverage audit tests for topocore.processing.classification.base.ClassificationResult.

ClassificationResult itself is genuinely active: confirmed 2 real
construction sites (classification/ml.py, classification/rules.py).
Its __post_init__ validation is a plain dataclass's own defensive
contract with no wrapper preventing malformed field construction --
same reasoning as processing._shared.build_cloud()'s own validation
(see that audit) -- so every branch is genuinely testable regardless
of what today's 2 real callers happen to construct.

NOT tested here by design -- documented as orphaned (zero callers
anywhere in the codebase, confirmed via grep): has_confidence(),
get_class_mask(), extract_class(), extract_classes(), and
Classifier.__call__() (the class's own docstring example uses
`classifier.classify(cloud)`, never the callable-interface form).
These remain architectural debt, not coverage debt -- consistent
with the same policy applied to io.ascii.mapper.ColumnMapper,
_shared.extract_attribute()/concatenate_clouds(), and
math.numeric's orphaned functions elsewhere in this audit.
"""

from __future__ import annotations

import numpy as np
import pytest
from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.chunk import Chunk
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult


@pytest.fixture
def cloud() -> PointCloud:
    cloud = PointCloud()
    chunk = Chunk(size=3, attributes=[PointAttribute.X, PointAttribute.Y, PointAttribute.Z])
    chunk[PointAttribute.X][:] = [1.0, 2.0, 3.0]
    chunk[PointAttribute.Y][:] = [0.0, 0.0, 0.0]
    chunk[PointAttribute.Z][:] = [0.0, 0.0, 0.0]
    cloud.add_chunk(chunk)
    return cloud


# ----------------------------------------------------------------------
# Happy path -- with and without confidence, matching the 2 real
# construction patterns (ml.py sometimes provides confidence,
# rules.py never does).
# ----------------------------------------------------------------------


def test_construction_without_confidence(cloud: PointCloud) -> None:
    result = ClassificationResult(labels=np.array([1, 2, 3]), cloud=cloud)

    assert result.has_confidence is False
    assert result.confidence is None


def test_construction_with_confidence(cloud: PointCloud) -> None:
    result = ClassificationResult(labels=np.array([1, 2, 3]), cloud=cloud, confidence=np.array([0.9, 0.8, 0.7]))

    assert result.has_confidence is True
    np.testing.assert_array_equal(result.confidence, [0.9, 0.8, 0.7])


# ----------------------------------------------------------------------
# labels validation.
# ----------------------------------------------------------------------


def test_two_dimensional_labels_rejected(cloud: PointCloud) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ClassificationResult(labels=np.array([[1, 2, 3]]), cloud=cloud)


def test_label_count_mismatch_rejected(cloud: PointCloud) -> None:
    with pytest.raises(ValueError, match="does not match cloud point count"):
        ClassificationResult(labels=np.array([1, 2]), cloud=cloud)


def test_non_integer_labels_rejected(cloud: PointCloud) -> None:
    with pytest.raises(TypeError, match="integer-typed"):
        ClassificationResult(labels=np.array([1.0, 2.0, 3.0]), cloud=cloud)


# ----------------------------------------------------------------------
# confidence validation.
# ----------------------------------------------------------------------


def test_two_dimensional_confidence_rejected(cloud: PointCloud) -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        ClassificationResult(
            labels=np.array([1, 2, 3]),
            cloud=cloud,
            confidence=np.array([[0.9, 0.8, 0.7]]),
        )


def test_confidence_count_mismatch_rejected(cloud: PointCloud) -> None:
    with pytest.raises(ValueError, match="does not match"):
        ClassificationResult(labels=np.array([1, 2, 3]), cloud=cloud, confidence=np.array([0.9, 0.8]))


def test_non_float_confidence_rejected(cloud: PointCloud) -> None:
    with pytest.raises(TypeError, match="float-typed"):
        ClassificationResult(labels=np.array([1, 2, 3]), cloud=cloud, confidence=np.array([1, 0, 1]))
