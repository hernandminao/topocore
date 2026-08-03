"""
topocore.features.protocols
=============================

Structural typing contracts for the feature extraction module.

`DetectionContext` bundles every optional input a detector might
need, so detectors depend on a single, stable parameter rather than
importing and threading through `PointCloud`, `TIN`, `DTM`, and
classification/normals results individually.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from topocore.core.types import FloatArray1D
from topocore.features.models import FeatureCollection
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.classification.base import ClassificationResult
from topocore.processing.features.pca import PCAFeatureResult
from topocore.processing.normals.models import NormalResult
from topocore.terrain.dtm import DTM
from topocore.terrain.tin import TIN


@dataclass(slots=True)
class DetectionContext:
    """
    Bundle of inputs available to a detector for a single run.

    All fields are optional because different detectors need
    different subsets (e.g. a tree detector needs `cloud` +
    `classification` + `normals`; a contour detector needs only
    `tin` or `dtm`). A detector declares what it needs via
    `BaseFeatureDetector.required_inputs` (using `ContextField`
    members from `models.py`), and `detect()` validates presence
    before invoking the detector's algorithm — see `base.py`.

    Field names here are the source of truth that `ContextField`
    values must match, since validation resolves fields via
    ``getattr(context, field.value)``.

    Parameters
    ----------
    cloud
        Source point cloud, if features are derived from points.
    tin
        Triangulated terrain surface, if available.
    dtm
        Gridded terrain surface, if available.
    classification
        Point cloud classification result, aligned with `cloud`.
    normals
        Per-point normal/curvature result, aligned with `cloud`.
    pca_features
        Named geometric feature arrays (planarity, linearity, etc.),
        as produced by ``topocore.processing.features.pca``.
    slope
        Per-point or per-cell slope values, depending on source.
    """

    cloud: PointCloud | None = None
    tin: TIN | None = None
    dtm: DTM | None = None
    classification: ClassificationResult | None = None
    normals: NormalResult | None = None
    pca_features: PCAFeatureResult | None = None
    slope: FloatArray1D | None = None


@runtime_checkable
class FeatureDetectorProtocol(Protocol):
    """
    Structural contract every detector must satisfy.

    Detectors implement this via `features.base.BaseFeatureDetector`
    in practice (which also enforces `required_inputs` validation),
    but the protocol is kept independent so detectors can be
    duck-typed in tests without importing the ABC.
    """

    def detect(self, context: DetectionContext) -> FeatureCollection: ...

    def name(self) -> str: ...


__all__ = ["DetectionContext", "FeatureDetectorProtocol"]
