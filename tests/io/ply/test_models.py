from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from topocore.io.ply.enums import (
    PLYFormat,
    PLYScalarType,
)
from topocore.io.ply.header import (
    PLYElement,
    PLYHeader,
    PLYListProperty,
    PLYProperty,
)

# =============================================================================
# PLYProperty
# =============================================================================


class TestPLYProperty:
    def test_constructor(self) -> None:

        prop = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        assert prop.name == "x"
        assert prop.dtype is PLYScalarType.FLOAT

    def test_is_frozen(self) -> None:

        prop = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        with pytest.raises(FrozenInstanceError):
            prop.name = "y"

    def test_equality(self) -> None:

        left = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        right = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        assert left == right

    def test_inequality(self) -> None:

        left = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        right = PLYProperty(
            name="y",
            dtype=PLYScalarType.FLOAT,
        )

        assert left != right

    def test_hash(self) -> None:

        left = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        right = PLYProperty(
            name="x",
            dtype=PLYScalarType.FLOAT,
        )

        assert hash(left) == hash(right)


# =============================================================================
# PLYListProperty
# =============================================================================


class TestPLYListProperty:
    def test_constructor(self) -> None:

        prop = PLYListProperty(
            name="vertex_indices",
            count_type=PLYScalarType.UCHAR,
            value_type=PLYScalarType.INT,
        )

        assert prop.name == "vertex_indices"
        assert prop.count_type is PLYScalarType.UCHAR
        assert prop.value_type is PLYScalarType.INT

    def test_is_frozen(self) -> None:

        prop = PLYListProperty(
            name="vertex_indices",
            count_type=PLYScalarType.UCHAR,
            value_type=PLYScalarType.INT,
        )

        with pytest.raises(FrozenInstanceError):
            prop.name = "indices"

    def test_equality(self) -> None:

        left = PLYListProperty(
            name="vertex_indices",
            count_type=PLYScalarType.UCHAR,
            value_type=PLYScalarType.INT,
        )

        right = PLYListProperty(
            name="vertex_indices",
            count_type=PLYScalarType.UCHAR,
            value_type=PLYScalarType.INT,
        )

        assert left == right


# =============================================================================
# PLYElement
# =============================================================================


class TestPLYElement:
    def test_empty_element(self) -> None:

        element = PLYElement(
            name="vertex",
            count=3,
        )

        assert element.name == "vertex"
        assert element.count == 3
        assert element.properties == []

    def test_property_names(self) -> None:

        element = PLYElement(
            name="vertex",
            count=3,
            properties=[
                PLYProperty("x", PLYScalarType.FLOAT),
                PLYProperty("y", PLYScalarType.FLOAT),
                PLYProperty("z", PLYScalarType.FLOAT),
            ],
        )

        assert element.property_names == (
            "x",
            "y",
            "z",
        )

    def test_has_property_true(self) -> None:

        element = PLYElement(
            name="vertex",
            count=1,
            properties=[
                PLYProperty(
                    "x",
                    PLYScalarType.FLOAT,
                )
            ],
        )

        assert element.has_property("x")

    def test_has_property_false(self) -> None:

        element = PLYElement(
            name="vertex",
            count=1,
        )

        assert not element.has_property("red")

    def test_get_property_found(self) -> None:

        prop = PLYProperty(
            "x",
            PLYScalarType.FLOAT,
        )

        element = PLYElement(
            name="vertex",
            count=1,
            properties=[prop],
        )

        assert element.get_property("x") is prop

    def test_get_property_not_found(self) -> None:

        element = PLYElement(
            name="vertex",
            count=1,
        )

        assert element.get_property("red") is None

    def test_list_property(self) -> None:

        prop = PLYListProperty(
            name="vertex_indices",
            count_type=PLYScalarType.UCHAR,
            value_type=PLYScalarType.INT,
        )

        element = PLYElement(
            name="face",
            count=10,
            properties=[prop],
        )

        assert element.has_property("vertex_indices")

        assert element.get_property("vertex_indices") == prop


# =============================================================================
# PLYHeader
# =============================================================================


class TestPLYHeader:
    def test_vertex_element(self) -> None:

        vertex = PLYElement(
            name="vertex",
            count=100,
        )

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[vertex],
        )

        assert header.vertex_element is vertex

    def test_vertex_element_not_found(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[
                PLYElement(
                    "face",
                    5,
                )
            ],
        )

        assert header.vertex_element is None

    def test_face_element(self) -> None:

        face = PLYElement(
            name="face",
            count=20,
        )

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[face],
        )

        assert header.face_element is face

    def test_face_element_not_found(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[
                PLYElement(
                    "vertex",
                    10,
                )
            ],
        )

        assert header.face_element is None

    def test_vertex_count(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[
                PLYElement(
                    "vertex",
                    42,
                )
            ],
        )

        assert header.vertex_count == 42

    def test_vertex_count_without_vertex(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
        )

        assert header.vertex_count == 0

    def test_has_element_true(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[
                PLYElement(
                    "vertex",
                    10,
                )
            ],
        )

        assert header.has_element("vertex")

    def test_has_element_false(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
        )

        assert not header.has_element("vertex")

    def test_get_element_found(self) -> None:

        vertex = PLYElement(
            "vertex",
            10,
        )

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[vertex],
        )

        assert header.get_element("vertex") is vertex

    def test_get_element_not_found(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
        )

        assert header.get_element("vertex") is None

    def test_comments(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
            comments=[
                "comment one",
                "comment two",
            ],
        )

        assert len(header.comments) == 2

    def test_obj_info(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
            obj_info=[
                "scanner",
                "project",
            ],
        )

        assert len(header.obj_info) == 2

    def test_header_size(self) -> None:

        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[],
            header_size=123,
        )

        assert header.header_size == 123
