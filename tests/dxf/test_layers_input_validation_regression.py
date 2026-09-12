"""
Regression test for a real defect found and fixed during this
project's own DXF documentation audit: `is_index_contour()`
(topocore.dxf.layers) performed no validation at all on `elevation`,
`base`, or `interval` -- all 3 come from a `Feature`'s own
`attributes`/`metadata.extra`, untrusted data from `is_index_contour()`'s
own point of view, since `Feature`/`FeatureMetadata` place no
constraint on them.

Before this fix, an invalid value raised a raw `ValueError`
(non-finite, or `interval <= 0`), `OverflowError` (`base=inf` reaching
`round()`), or `TypeError` (a non-numeric value) -- none of which
`DXFExporter.export()`'s own `except (DXFGeometryError,
DXFExportError)` catches. Confirmed directly, with a real
`FeatureCollection` containing 2 valid features alongside 1 CONTOUR
feature with `base=inf`: `strict=False` aborted the WHOLE export (no
`.dxf` file produced at all), silently violating that option's own
documented "skip this feature, keep going" contract -- the same
failure shape `layer_for()`'s own historical PR19 fix (see
`layers.py`'s own `layer_for()`) already addressed for a different
unwrapped exception reaching the same `export()` boundary.

`interval=inf` was confirmed a distinct, non-crashing but still
dangerous case: it silently returned `True` (a contour would be
misclassified as an index/MAJOR contour) rather than raising
anything -- also corrected here.

Fixed by validating all 3 values (real number, finite) plus
`interval > 0` at the top of `is_index_contour()`, raising
`DXFExportError` -- not `DetectionError` (this is DXF's own concern:
the data is invalid from the exporter's point of view, regardless of
which upstream package produced the Feature) -- which is already one
of the exception types `DXFExporter.export()`'s own per-feature
try/except catches, so no change to `export()` itself was needed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from topocore.dxf.exceptions import DXFExportError
from topocore.dxf.exporter import DXFExporter
from topocore.dxf.layers import is_index_contour
from topocore.dxf.models import DXFExportOptions, ExportContext
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureMetadata,
    FeatureType,
    GeometryType,
)


@pytest.mark.parametrize(
    ("kwargs", "expected_message_fragment"),
    [
        ({"elevation": 5.0, "base": 0.0, "interval": 0.0, "every": 5}, "interval must be positive"),
        ({"elevation": 5.0, "base": 0.0, "interval": -1.0, "every": 5}, "interval must be positive"),
        ({"elevation": 5.0, "base": 0.0, "interval": float("nan"), "every": 5}, "interval must be finite"),
        ({"elevation": 5.0, "base": 0.0, "interval": float("inf"), "every": 5}, "interval must be finite"),
        ({"elevation": 5.0, "base": float("nan"), "interval": 1.0, "every": 5}, "base must be finite"),
        ({"elevation": 5.0, "base": float("inf"), "interval": 1.0, "every": 5}, "base must be finite"),
        ({"elevation": float("nan"), "base": 0.0, "interval": 1.0, "every": 5}, "elevation must be finite"),
        ({"elevation": float("inf"), "base": 0.0, "interval": 1.0, "every": 5}, "elevation must be finite"),
        ({"elevation": 5.0, "base": 0.0, "interval": "1.0", "every": 5}, "interval must be a real number"),
    ],
)
def test_is_index_contour_rejects_invalid_input(kwargs: dict, expected_message_fragment: str) -> None:
    with pytest.raises(DXFExportError, match=expected_message_fragment):
        is_index_contour(kwargs["elevation"], base=kwargs["base"], interval=kwargs["interval"], every=kwargs["every"])


def test_is_index_contour_valid_input_unaffected() -> None:
    """Confirms the fix changed nothing for well-formed input."""
    assert is_index_contour(10.0, base=0.0, interval=1.0, every=5) is True
    assert is_index_contour(7.0, base=0.0, interval=1.0, every=5) is False


def _contour_feature(feature_id: int, *, interval: float, base: float = 0.0, elevation: float = 5.0) -> Feature:
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    geometry = FeatureGeometry(GeometryType.POLYLINE, vertices)
    metadata = FeatureMetadata(detector="test", version="1.0", extra={"base": base, "interval": interval})
    return Feature(
        feature_id=feature_id,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTOUR,
        geometry=geometry,
        attributes={"elevation": elevation},
        metadata=metadata,
    )


def _valid_breakline_feature(feature_id: int) -> Feature:
    vertices = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    geometry = FeatureGeometry(GeometryType.POLYLINE, vertices)
    return Feature(
        feature_id=feature_id, category=FeatureCategory.TERRAIN, feature_type=FeatureType.BREAKLINE, geometry=geometry
    )


def test_strict_false_isolates_invalid_contour_and_writes_the_valid_features() -> None:
    """
    The exact real defect this fix addresses: before the fix, this
    exact scenario raised an unwrapped OverflowError and produced NO
    file at all, even though 2 features in the collection were
    genuinely valid.
    """
    collection = FeatureCollection()
    collection.add(_valid_breakline_feature(1))
    collection.add(_contour_feature(2, interval=1.0, base=float("inf")))
    collection.add(_valid_breakline_feature(3))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dxf"

        report = DXFExporter(ExportContext(options=DXFExportOptions(strict=False))).export(collection, path)

        assert path.exists()
        assert report.feature_count == 3
        assert report.skipped_features == 1
        assert any("base must be finite" in w for w in report.warnings)


def test_strict_true_aborts_the_whole_export() -> None:
    collection = FeatureCollection()
    collection.add(_valid_breakline_feature(1))
    collection.add(_contour_feature(2, interval=1.0, base=float("inf")))
    collection.add(_valid_breakline_feature(3))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dxf"

        with pytest.raises(DXFExportError, match="base must be finite"):
            DXFExporter(ExportContext(options=DXFExportOptions(strict=True))).export(collection, path)

        assert not path.exists()


def test_valid_collection_unaffected_by_the_fix() -> None:
    collection = FeatureCollection()
    collection.add(_valid_breakline_feature(1))
    collection.add(_contour_feature(2, interval=1.0, base=0.0, elevation=10.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.dxf"

        report = DXFExporter(ExportContext(options=DXFExportOptions(strict=True))).export(collection, path)

        assert path.exists()
        assert report.feature_count == 2
        assert report.skipped_features == 0
