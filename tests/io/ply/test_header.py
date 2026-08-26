"""Regression tests for topocore.io.ply.header."""

from __future__ import annotations

import pytest

from topocore.io.ply.enums import PLYFormat, PLYScalarType
from topocore.io.ply.header import (
    PLYElement,
    PLYHeader,
    PLYListProperty,
    PLYProperty,
)


def test_ply_property_stores_definition() -> None:
    prop = PLYProperty(
        name="x",
        dtype=PLYScalarType.FLOAT,
    )

    assert prop.name == "x"
    assert prop.dtype is PLYScalarType.FLOAT


def test_ply_list_property_stores_definition() -> None:
    prop = PLYListProperty(
        name="vertex_indices",
        count_type=PLYScalarType.UCHAR,
        value_type=PLYScalarType.INT,
    )

    assert prop.name == "vertex_indices"
    assert prop.count_type is PLYScalarType.UCHAR
    assert prop.value_type is PLYScalarType.INT


def test_ply_scalar_property_is_immutable() -> None:
    prop = PLYProperty(
        name="x",
        dtype=PLYScalarType.FLOAT,
    )

    with pytest.raises(AttributeError):
        prop.name = "y"  # type: ignore[misc]


def test_ply_list_property_is_immutable() -> None:
    prop = PLYListProperty(
        name="vertex_indices",
        count_type=PLYScalarType.UCHAR,
        value_type=PLYScalarType.INT,
    )

    with pytest.raises(AttributeError):
        prop.name = "indices"  # type: ignore[misc]


def test_ply_element_default_properties_are_independent() -> None:
    first = PLYElement(name="vertex", count=10)
    second = PLYElement(name="face", count=5)

    first.properties.append(
        PLYProperty("x", PLYScalarType.FLOAT),
    )

    assert first.property_names == ("x",)
    assert second.property_names == ()


def test_ply_element_property_lookup() -> None:
    x = PLYProperty("x", PLYScalarType.FLOAT)
    y = PLYProperty("y", PLYScalarType.FLOAT)
    indices = PLYListProperty(
        "vertex_indices",
        PLYScalarType.UCHAR,
        PLYScalarType.INT,
    )

    element = PLYElement(
        name="vertex",
        count=3,
        properties=[x, y, indices],
    )

    assert element.property_names == (
        "x",
        "y",
        "vertex_indices",
    )

    assert element.has_property("x")
    assert element.has_property("vertex_indices")
    assert not element.has_property("z")

    assert element.get_property("x") is x
    assert element.get_property("vertex_indices") is indices
    assert element.get_property("missing") is None


def test_ply_header_lookup() -> None:
    vertex = PLYElement(
        name="vertex",
        count=100,
        properties=[
            PLYProperty("x", PLYScalarType.FLOAT),
            PLYProperty("y", PLYScalarType.FLOAT),
            PLYProperty("z", PLYScalarType.FLOAT),
        ],
    )

    face = PLYElement(
        name="face",
        count=50,
        properties=[
            PLYListProperty(
                "vertex_indices",
                PLYScalarType.UCHAR,
                PLYScalarType.INT,
            ),
        ],
    )

    header = PLYHeader(
        format=PLYFormat.ASCII,
        version="1.0",
        elements=[vertex, face],
        comments=["created by TopoCore"],
        obj_info=["example"],
        header_size=128,
    )

    assert header.format is PLYFormat.ASCII
    assert header.version == "1.0"
    assert header.header_size == 128
    assert header.comments == ["created by TopoCore"]
    assert header.obj_info == ["example"]

    assert header.vertex_element is vertex
    assert header.face_element is face
    assert header.vertex_count == 100

    assert header.has_element("vertex")
    assert header.has_element("face")
    assert not header.has_element("edge")

    assert header.get_element("vertex") is vertex
    assert header.get_element("face") is face
    assert header.get_element("edge") is None


def test_ply_header_without_vertex_or_face() -> None:
    header = PLYHeader(
        format=PLYFormat.ASCII,
        version="1.0",
        elements=[],
    )

    assert header.vertex_element is None
    assert header.face_element is None
    assert header.vertex_count == 0
    assert header.get_element("vertex") is None
    assert header.get_element("face") is None
    assert not header.has_element("vertex")


def test_ply_header_can_contain_other_element_types() -> None:
    edge = PLYElement(
        name="edge",
        count=7,
    )

    header = PLYHeader(
        format=PLYFormat.BINARY_LITTLE_ENDIAN,
        version="1.0",
        elements=[edge],
    )

    assert header.has_element("edge")
    assert header.get_element("edge") is edge
    assert header.vertex_element is None
    assert header.face_element is None
    assert header.vertex_count == 0
