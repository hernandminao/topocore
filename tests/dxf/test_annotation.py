"""
Regression tests for DXF V2 annotation (TEXT labels), per the
decided design: contour elevation labels on by default, point
labels opt-in, XDATA and layers always preserved unchanged.
"""

from __future__ import annotations

import ezdxf
import numpy as np
import pytest
from topocore.dxf import DXFExporter
from topocore.dxf.models import DXFExportOptions, ExportContext
from topocore.features.models import (
    Feature,
    FeatureCategory,
    FeatureCollection,
    FeatureGeometry,
    FeatureType,
    GeometryType,
)


def _tree_feature() -> Feature:
    return Feature(
        feature_id=1,
        category=FeatureCategory.VEGETATION,
        feature_type=FeatureType.TREE,
        geometry=FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[0.0, 0.0, 10.0]])),
        attributes={"survey_point_ids": ("ARBOL_42",)},
    )


def _contour_feature() -> Feature:
    return Feature(
        feature_id=2,
        category=FeatureCategory.TERRAIN,
        feature_type=FeatureType.CONTOUR,
        geometry=FeatureGeometry(
            geometry_type=GeometryType.POLYLINE,
            vertices=np.array([[0.0, 0.0, 205.0], [5.0, 5.0, 205.0], [10.0, 10.0, 205.0]]),
        ),
        attributes={"elevation": 205.0},
    )


def _text_entities(path: str) -> list:
    doc = ezdxf.readfile(path)
    return [e for e in doc.modelspace() if e.dxftype() == "TEXT"]


def test_default_options_label_contours_but_not_points(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_tree_feature())
    collection.add(_contour_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions())).export(collection, str(output))

    texts = _text_entities(str(output))
    assert [t.dxf.text for t in texts] == ["205.00"]


def test_point_labels_true_adds_survey_id_text(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_tree_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(collection, str(output))

    texts = _text_entities(str(output))
    assert len(texts) == 1
    assert texts[0].dxf.text == "ARBOL_42"
    assert texts[0].dxf.layer == "TREES"


def test_point_label_falls_back_to_feature_id_without_survey_id(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(
        Feature(
            feature_id=99,
            category=FeatureCategory.VEGETATION,
            feature_type=FeatureType.TREE,
            geometry=FeatureGeometry(geometry_type=GeometryType.POINT, vertices=np.array([[0.0, 0.0, 10.0]])),
            attributes={},
        )
    )

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(collection, str(output))

    texts = _text_entities(str(output))
    assert texts[0].dxf.text == "99"


def test_point_label_uses_the_real_field_id_from_a_real_featurebuilder_pipeline(tmp_path) -> None:
    """
    Regression: a real defect -- add_point_label() originally looked
    for attributes["survey_id"], a key that FeatureBuilder (the real
    path from a field survey) never actually produces. Confirmed with
    real execution: FeatureBuilder stores the original field id(s)
    under attributes["survey_point_ids"] (a tuple, even for a single
    point). Without this integration test, the earlier unit tests
    (using a hand-built Feature with a fictional "survey_id" key)
    would have kept passing while every real pipeline silently fell
    back to the numeric feature_id instead of the real field id.
    """
    from topocore.features.feature_builder import FeatureBuilder
    from topocore.features.feature_codes import FeatureCodeRegistry
    from topocore.survey.models import SurveyPoint, SurveyPointSet

    survey = SurveyPointSet(points=(SurveyPoint(id="ARBOL_42", x=0.0, y=0.0, z=10.0, code="ARBOL"),))
    result = FeatureBuilder(FeatureCodeRegistry.default()).build(survey)

    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    for f in result.features:
        collection.add(f)

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(point_labels=True))).export(collection, str(output))

    texts = _text_entities(str(output))
    assert texts[0].dxf.text == "ARBOL_42"


def test_contour_labels_false_produces_no_text(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(contour_labels=False))).export(collection, str(output))

    assert _text_entities(str(output)) == []


def test_labels_false_everywhere_produces_no_text_at_all(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_tree_feature())
    collection.add(_contour_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(contour_labels=False, point_labels=False))).export(
        collection, str(output)
    )

    assert _text_entities(str(output)) == []


def test_label_text_is_on_the_same_layer_as_its_geometry(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions())).export(collection, str(output))

    doc = ezdxf.readfile(str(output))
    msp = doc.modelspace()
    polyline_layer = next(e.dxf.layer for e in msp if e.dxftype() == "LWPOLYLINE")
    text_layer = next(e.dxf.layer for e in msp if e.dxftype() == "TEXT")
    assert polyline_layer == text_layer


def test_xdata_and_geometry_are_unaffected_by_labels(tmp_path) -> None:
    """The core guarantee: adding labels never changes the existing
    XDATA payload or the geometry entity itself."""
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature())

    without_labels = tmp_path / "without.dxf"
    with_labels = tmp_path / "with.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(contour_labels=False))).export(collection, str(without_labels))
    DXFExporter(ExportContext(options=DXFExportOptions(contour_labels=True))).export(collection, str(with_labels))

    def polyline_xdata(path: str) -> list:
        doc = ezdxf.readfile(path)
        entity = next(e for e in doc.modelspace() if e.dxftype() == "LWPOLYLINE")
        return entity.get_xdata("TOPOCORE")

    assert polyline_xdata(str(without_labels)) == polyline_xdata(str(with_labels))


def test_a_contour_without_a_numeric_elevation_attribute_gets_no_label(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(
        Feature(
            feature_id=1,
            category=FeatureCategory.TERRAIN,
            feature_type=FeatureType.CONTOUR,
            geometry=FeatureGeometry(
                geometry_type=GeometryType.POLYLINE,
                vertices=np.array([[0.0, 0.0, 205.0], [10.0, 10.0, 205.0]]),
            ),
            attributes={},  # sin elevation
        )
    )

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions())).export(collection, str(output))

    assert _text_entities(str(output)) == []


def test_custom_label_text_height_is_applied(tmp_path) -> None:
    collection = FeatureCollection()
    collection.crs = "EPSG:9377"
    collection.add(_contour_feature())

    output = tmp_path / "test.dxf"
    DXFExporter(ExportContext(options=DXFExportOptions(label_text_height=5.0))).export(collection, str(output))

    texts = _text_entities(str(output))
    assert texts[0].dxf.height == pytest.approx(5.0)


def test_nonpositive_label_text_height_is_rejected() -> None:
    with pytest.raises(ValueError, match="label_text_height"):
        DXFExportOptions(label_text_height=0.0)
