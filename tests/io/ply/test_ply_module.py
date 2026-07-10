"""
Unit tests for the PLY module (100% coverage).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from topocore.io.common.records import PointRecordBatch
from topocore.io.ply.converter import PLYConverter
from topocore.io.ply.enums import PLY_NUMPY_DTYPES, PLYFormat, PLYScalarType
from topocore.io.ply.exceptions import InvalidPLYError, PLYError
from topocore.io.ply.header import (
    PLYElement,
    PLYHeader,
    PLYListProperty,
    PLYProperty,
)
from topocore.io.ply.header_parser import PLYHeaderParser
from topocore.io.ply.reader import PLYReader
from topocore.pointcloud.attributes import PointAttribute

# =============================================================================
# Helpers and Fixtures
# =============================================================================


def build_header_stream(text: str) -> io.BytesIO:
    """Return a binary stream containing a PLY header snippet."""
    return io.BytesIO(text.encode("utf-8"))


class FakeScalarProperty:
    """Minimal scalar property stub for reader unit tests."""

    def __init__(self, name: str, *, numpy_dtype: np.dtype | None = None) -> None:
        self.name = name
        dtype = numpy_dtype or np.dtype("float32")
        self.dtype = type("DTypeWrapper", (), {"numpy_dtype": dtype})()


class FakeListProperty:
    """Minimal list property stub (no scalar dtype)."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeVertex:
    """Vertex element stub for reader unit tests."""

    def __init__(self, count: int = 3, *, include_list_property: bool = False) -> None:
        self.name = "vertex"
        self.count = count
        self.properties: list[FakeScalarProperty | FakeListProperty] = [
            FakeScalarProperty("x"),
            FakeScalarProperty("y"),
            FakeScalarProperty("z"),
        ]
        if include_list_property:
            self.properties.append(FakeListProperty("indices"))


class FakeHeader:
    """Header stub for reader unit tests."""

    def __init__(
        self,
        fmt: PLYFormat = PLYFormat.ASCII,
        *,
        vertex_count: int = 3,
        include_list_property: bool = False,
    ) -> None:
        self.format = fmt
        self.elements = [
            FakeVertex(
                vertex_count,
                include_list_property=include_list_property,
            )
        ]


class PLYHeaderStubWithoutVertex:
    elements = [PLYElement(name="face", count=1, properties=[])]


class FakeChunk(dict):
    def __init__(self, attributes):
        super().__init__(attributes)

    def has_attribute(self, attribute):
        return attribute in self


@pytest.fixture
def identity_converter():
    """Patch PLYReader converter to return the batch unchanged."""

    def _apply(reader) -> None:
        reader._converter.convert = lambda batch: batch

    return _apply


def write_ascii_ply(path: Path, rows: list[tuple[float, float, float]]) -> None:
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    body = "".join(f"{x} {y} {z}\n" for x, y, z in rows)
    path.write_text(header + body, encoding="utf-8")


def write_binary_ply(path: Path, rows: list[tuple[float, float, float]], *, fmt: PLYFormat) -> None:
    if fmt == PLYFormat.BINARY_LITTLE_ENDIAN:
        format_line = "binary_little_endian"
        endian = "<"
    else:
        format_line = "binary_big_endian"
        endian = ">"
    header = (
        "ply\n"
        f"format {format_line} 1.0\n"
        f"element vertex {len(rows)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", f"{endian}f4"), ("y", f"{endian}f4"), ("z", f"{endian}f4")])
    data = np.array(rows, dtype=dtype)
    with path.open("wb") as stream:
        stream.write(header.encode("ascii"))
        data.tofile(stream)


# =============================================================================
# Test Enums
# =============================================================================


class TestPLYEnums:
    def test_ply_format_values(self) -> None:
        assert PLYFormat("ascii") == PLYFormat.ASCII
        assert PLYFormat("binary_little_endian") == PLYFormat.BINARY_LITTLE_ENDIAN
        assert PLYFormat("binary_big_endian") == PLYFormat.BINARY_BIG_ENDIAN
        with pytest.raises(ValueError):
            PLYFormat("invalid")

    def test_ply_scalar_type_dtypes(self) -> None:
        assert PLYScalarType.FLOAT.numpy_dtype == np.dtype("f4")
        assert PLYScalarType.UCHAR.numpy_dtype == np.dtype("u1")
        assert len(PLY_NUMPY_DTYPES) == len(PLYScalarType)


# =============================================================================
# Test Exceptions
# =============================================================================


class TestPLYExceptions:
    def test_inheritance(self) -> None:
        assert issubclass(PLYError, Exception)
        assert issubclass(InvalidPLYError, PLYError)

    def test_raise_errors(self) -> None:
        with pytest.raises(PLYError):
            raise PLYError("fail")
        with pytest.raises(InvalidPLYError):
            raise InvalidPLYError("fail")


# =============================================================================
# Test Header Models
# =============================================================================


class TestPLYHeaderModels:
    def test_ply_property(self) -> None:
        prop = PLYProperty(name="x", dtype=PLYScalarType.FLOAT)
        assert prop.name == "x"
        assert prop.dtype == PLYScalarType.FLOAT

    def test_ply_list_property(self) -> None:
        list_prop = PLYListProperty(name="indices", count_type=PLYScalarType.UCHAR, value_type=PLYScalarType.INT)
        assert list_prop.name == "indices"
        assert list_prop.count_type == PLYScalarType.UCHAR

    def test_ply_element(self) -> None:
        prop = PLYProperty(name="x", dtype=PLYScalarType.FLOAT)
        elem = PLYElement(name="vertex", count=10, properties=[prop])
        assert elem.name == "vertex"
        assert elem.count == 10
        assert elem.property_names == ("x",)
        assert elem.has_property("x")
        assert not elem.has_property("y")
        assert elem.get_property("x") == prop
        assert elem.get_property("y") is None

    def test_ply_header(self) -> None:
        vertex = PLYElement(name="vertex", count=10)
        face = PLYElement(name="face", count=5)
        header = PLYHeader(
            format=PLYFormat.ASCII,
            version="1.0",
            elements=[vertex, face],
            comments=["test"],
            obj_info=["info"],
            header_size=100,
        )
        assert header.vertex_element == vertex
        assert header.face_element == face
        assert header.vertex_count == 10
        assert header.has_element("vertex")
        assert not header.has_element("edge")
        assert header.get_element("vertex") == vertex
        assert header.get_element("edge") is None
        assert len(header.comments) == 1
        assert len(header.obj_info) == 1
        assert header.header_size == 100

    def test_ply_header_no_vertex(self) -> None:
        header = PLYHeader(format=PLYFormat.ASCII, version="1.0", elements=[])
        assert header.vertex_element is None
        assert header.face_element is None
        assert header.vertex_count == 0


# =============================================================================
# Test Header Parser
# =============================================================================


class TestPLYHeaderParser:
    def test_parse_ascii_valid(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert header.format == PLYFormat.ASCII
        assert len(header.elements) == 1

    def test_parse_binary_le_valid(self) -> None:
        header_text = "ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert header.format == PLYFormat.BINARY_LITTLE_ENDIAN

    def test_parse_binary_be_valid(self) -> None:
        header_text = "ply\nformat binary_big_endian 1.0\nelement vertex 2\nproperty float x\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert header.format == PLYFormat.BINARY_BIG_ENDIAN

    def test_parse_comments_and_objinfo(self) -> None:
        header_text = "ply\ncomment my comment\nobj_info my info\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert len(header.comments) == 1
        assert len(header.obj_info) == 1

    def test_parse_list_property(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert header.has_element("face")
        assert header.get_element("face").properties[0].name == "vertex_indices"

    def test_parse_with_extra_whitespace(self) -> None:
        header_text = "ply\n\nformat ascii 1.0\n\nelement vertex 1\nproperty float x\n\nend_header\n"
        header = PLYHeaderParser.parse(build_header_stream(header_text))
        assert header.format == PLYFormat.ASCII

    def test_error_unexpected_eof(self) -> None:
        with pytest.raises(InvalidPLYError, match="Unexpected end of file"):
            PLYHeaderParser.parse(io.BytesIO(b"ply\nformat ascii 1.0\n"))

    def test_error_invalid_utf8(self) -> None:
        with pytest.raises(InvalidPLYError, match="not valid UTF-8"):
            PLYHeaderParser.parse(io.BytesIO(b"ply\n\xff\xfe\nend_header\n"))

    def test_error_empty_magic(self) -> None:
        with pytest.raises(InvalidPLYError, match="Empty PLY file"):
            PLYHeaderParser._validate_magic([])

    def test_error_invalid_magic(self) -> None:
        with pytest.raises(InvalidPLYError, match="Invalid PLY signature"):
            PLYHeaderParser._validate_magic(["not_ply"])

    def test_error_no_elements(self) -> None:
        header_text = "ply\nformat ascii 1.0\nend_header\n"
        with pytest.raises(InvalidPLYError, match="no elements"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_missing_format(self) -> None:
        header_text = "ply\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="does not declare a format"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_missing_version(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def broken_format(cls, tokens, state):
            state.fmt = PLYFormat.ASCII

        monkeypatch.setattr(PLYHeaderParser, "_handle_format", classmethod(broken_format))
        header_text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="does not declare a version"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_unknown_keyword(self) -> None:
        header_text = "ply\nformat ascii 1.0\nunknown_token value\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Unknown PLY header keyword"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_malformed_format(self) -> None:
        header_text = "ply\nformat ascii\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Malformed format declaration"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_duplicate_format(self) -> None:
        header_text = "ply\nformat ascii 1.0\nformat ascii 1.0\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Duplicate format declaration"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_unsupported_format(self) -> None:
        header_text = "ply\nformat json 1.0\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Unsupported PLY format"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_unsupported_version(self) -> None:
        header_text = "ply\nformat ascii 2.0\nelement vertex 1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Unsupported PLY version"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_malformed_element(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Malformed element declaration"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_invalid_element_count(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex abc\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Invalid element count"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_negative_element_count(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex -1\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="cannot be negative"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_duplicated_element(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex 1\nelement vertex 2\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Duplicated element"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_property_before_element(self) -> None:
        header_text = "ply\nformat ascii 1.0\nproperty float x\nelement vertex 1\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Property declared before an element"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_duplicated_property(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Duplicated property"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_malformed_property(self) -> None:
        with pytest.raises(InvalidPLYError, match="Malformed property declaration"):
            PLYHeaderParser._parse_property(["property"])

    def test_error_malformed_scalar_property(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty float\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Malformed scalar property"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_unsupported_scalar_type(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement vertex 1\nproperty imaginary x\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Unsupported scalar type"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_malformed_list_property(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Malformed list property"):
            PLYHeaderParser.parse(build_header_stream(header_text))

    def test_error_unsupported_list_property_type(self) -> None:
        header_text = "ply\nformat ascii 1.0\nelement face 1\nproperty list imaginary int vertex_indices\nend_header\n"
        with pytest.raises(InvalidPLYError, match="Unsupported list property type"):
            PLYHeaderParser.parse(build_header_stream(header_text))


# =============================================================================
# Test Converter
# =============================================================================


class TestPLYConverter:
    def test_attribute_mapping(self) -> None:
        converter = PLYConverter()
        mapping = converter.attribute_mapping
        assert mapping["red"] is PointAttribute.COLOR
        assert mapping["nx"] is PointAttribute.NORMAL

    def test_populate_color(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({PointAttribute.COLOR: np.zeros((2, 3), dtype=np.uint8)})
        batch = PointRecordBatch(
            arrays={
                "red": np.array([10, 20], dtype=np.uint8),
                "green": np.array([30, 40], dtype=np.uint8),
                "blue": np.array([50, 60], dtype=np.uint8),
            }
        )
        converter._populate_color(chunk, batch)
        np.testing.assert_array_equal(
            chunk[PointAttribute.COLOR], np.array([[10, 30, 50], [20, 40, 60]], dtype=np.uint8)
        )

    def test_populate_color_missing_component(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({PointAttribute.COLOR: np.zeros((2, 3), dtype=np.uint8)})
        batch = PointRecordBatch(arrays={"red": np.array([1, 2]), "green": np.array([3, 4])})
        before = chunk[PointAttribute.COLOR].copy()
        converter._populate_color(chunk, batch)
        np.testing.assert_array_equal(chunk[PointAttribute.COLOR], before)

    def test_populate_color_without_attribute(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({})
        batch = PointRecordBatch(arrays={"red": np.array([1]), "green": np.array([1]), "blue": np.array([1])})
        converter._populate_color(chunk, batch)
        assert PointAttribute.COLOR not in chunk

    def test_populate_normals(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({PointAttribute.NORMAL: np.zeros((2, 3), dtype=np.float32)})
        batch = PointRecordBatch(
            arrays={
                "nx": np.array([1.0, 2.0], dtype=np.float32),
                "ny": np.array([3.0, 4.0], dtype=np.float32),
                "nz": np.array([5.0, 6.0], dtype=np.float32),
            }
        )
        converter._populate_normals(chunk, batch)
        np.testing.assert_array_equal(chunk[PointAttribute.NORMAL], np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float32))

    def test_populate_normals_missing_component(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({PointAttribute.NORMAL: np.zeros((2, 3), dtype=np.float32)})
        batch = PointRecordBatch(arrays={"nx": np.array([1]), "ny": np.array([2])})
        before = chunk[PointAttribute.NORMAL].copy()
        converter._populate_normals(chunk, batch)
        np.testing.assert_array_equal(chunk[PointAttribute.NORMAL], before)

    def test_populate_normals_without_attribute(self) -> None:
        converter = PLYConverter()
        chunk = FakeChunk({})
        batch = PointRecordBatch(arrays={"nx": np.array([1]), "ny": np.array([1]), "nz": np.array([1])})
        converter._populate_normals(chunk, batch)
        assert PointAttribute.NORMAL not in chunk

    def test_populate_special_attributes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        converter = PLYConverter()
        called = {"color": False, "normal": False}
        monkeypatch.setattr(converter, "_populate_color", lambda c, b: called.update(color=True))
        monkeypatch.setattr(converter, "_populate_normals", lambda c, b: called.update(normal=True))
        converter._populate_special_attributes(FakeChunk({}), PointRecordBatch(arrays={}))
        assert called["color"]
        assert called["normal"]


# =============================================================================
# Test Reader
# =============================================================================


class TestPLYReader:
    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            PLYReader("dummy.ply", chunk_size=0)

    def test_vertex_requires_header(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"")
        with pytest.raises(InvalidPLYError, match="header not loaded"):
            reader._vertex_element()

    def test_missing_vertex_element(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._header = PLYHeaderStubWithoutVertex()
        with pytest.raises(InvalidPLYError, match="no vertex element"):
            reader._vertex_element()

    def test_empty_elements_raises(self) -> None:
        reader = PLYReader("dummy.ply")
        reader._header = type("Header", (), {"elements": []})()
        with pytest.raises(InvalidPLYError):
            reader._vertex_element()

    def test_ascii_iteration_with_mock_header(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"1 2 3\n4 5 6\n7 8 9\n")
        reader._header = header
        identity_converter(reader)
        chunks = list(reader._iter_ascii(header.elements[0]))
        assert len(chunks) == 1

    def test_ascii_skips_empty_lines(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"\n\n1 2 3\n")
        identity_converter(reader)
        chunks = list(reader._iter_ascii(header.elements[0]))
        assert len(chunks) == 1

    def test_ascii_utf8_decode_error(self) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"\xff\xff\xff\n")
        with pytest.raises(InvalidPLYError, match="not valid UTF-8"):
            next(reader._iter_ascii(header.elements[0]))

    def test_ascii_eof_before_total(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII, vertex_count=5)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"1 2 3\n")
        identity_converter(reader)
        chunks = list(reader._iter_ascii(header.elements[0]))
        assert len(chunks) == 1

    def test_ascii_chunk_flush_on_boundary(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII, vertex_count=4)
        reader = PLYReader("dummy.ply", chunk_size=2)
        reader._header = header
        reader._stream = io.BytesIO(b"1 2 3\n4 5 6\n7 8 9\n10 11 12\n")
        identity_converter(reader)
        chunks = list(reader._iter_ascii(header.elements[0]))
        assert len(chunks) == 2

    def test_flush_ascii_skips_list_properties(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII, include_list_property=True)
        reader = PLYReader("dummy.ply")
        reader._header = header
        reader._stream = io.BytesIO(b"1 2 3\n")
        identity_converter(reader)
        chunks = list(reader._iter_ascii(header.elements[0]))
        assert len(chunks) == 1

    @patch("topocore.io.ply.reader.PLYHeaderParser.parse")
    def test_iter_closes_stream(self, mock_parse, identity_converter) -> None:
        header = FakeHeader(PLYFormat.ASCII)
        mock_parse.return_value = header
        reader = PLYReader("dummy.ply")
        reader._stream = io.BytesIO(b"1 2 3\n")
        reader._header = header
        identity_converter(reader)
        list(reader)
        assert reader._stream is None

    def test_read_from_file_opens_stream(self, tmp_path, identity_converter) -> None:
        ply_path = tmp_path / "points.ply"
        write_ascii_ply(ply_path, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
        reader = PLYReader(ply_path, chunk_size=1)
        identity_converter(reader)
        chunks = list(reader)
        assert len(chunks) == 2
        assert reader._stream is None

    def test_open_is_idempotent(self, tmp_path) -> None:
        ply_path = tmp_path / "points.ply"
        write_ascii_ply(ply_path, [(0.0, 0.0, 0.0)])
        reader = PLYReader(ply_path)
        reader._open()
        stream = reader._stream
        header = reader._header
        reader._open()
        assert reader._stream is stream
        assert reader._header is header
        reader.close()

    @patch("numpy.fromfile")
    def test_binary_iteration(self, mock_fromfile, identity_converter) -> None:
        mock_fromfile.return_value = np.array([(1.0, 2.0, 3.0)], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        header = FakeHeader(PLYFormat.BINARY_LITTLE_ENDIAN, vertex_count=1)
        reader = PLYReader("dummy.ply")
        reader._header = header
        identity_converter(reader)
        chunks = list(reader._iter_binary(header.elements[0]))
        assert len(chunks) == 1

    @patch("numpy.fromfile")
    def test_binary_stops_on_empty_read(self, mock_fromfile, identity_converter) -> None:
        mock_fromfile.return_value = np.array([], dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
        header = FakeHeader(PLYFormat.BINARY_LITTLE_ENDIAN, vertex_count=5)
        reader = PLYReader("dummy.ply")
        reader._header = header
        identity_converter(reader)
        chunks = list(reader._iter_binary(header.elements[0]))
        assert chunks == []

    def test_iter_routes_to_binary_path(self, tmp_path, identity_converter) -> None:
        ply_path = tmp_path / "binary_le.ply"
        write_binary_ply(ply_path, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], fmt=PLYFormat.BINARY_LITTLE_ENDIAN)
        reader = PLYReader(ply_path, chunk_size=1)
        identity_converter(reader)
        chunks = list(reader)
        assert len(chunks) == 2

    def test_little_endian_dtype_prefix(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.BINARY_LITTLE_ENDIAN, vertex_count=1)
        reader = PLYReader("dummy.ply")
        reader._header = header
        captured = {}

        def fake_fromfile(stream, dtype, count):
            captured["dtype"] = dtype
            return np.array([(1.0, 2.0, 3.0)], dtype=dtype)

        with patch("numpy.fromfile", side_effect=fake_fromfile):
            identity_converter(reader)
            list(reader._iter_binary(header.elements[0]))
        dtype = captured["dtype"]
        assert dtype["x"].str.startswith("<")

    def test_big_endian_dtype_prefix(self, identity_converter) -> None:
        header = FakeHeader(PLYFormat.BINARY_BIG_ENDIAN, vertex_count=1)
        reader = PLYReader("dummy.ply")
        reader._header = header
        captured = {}

        def fake_fromfile(stream, dtype, count):
            captured["dtype"] = dtype
            return np.array([(1.0, 2.0, 3.0)], dtype=dtype)

        with patch("numpy.fromfile", side_effect=fake_fromfile):
            identity_converter(reader)
            list(reader._iter_binary(header.elements[0]))
        dtype = captured["dtype"]
        assert dtype["x"].str.startswith(">")

    def test_iter_routes_to_binary_be(self, tmp_path, identity_converter) -> None:
        ply_path = tmp_path / "binary_be.ply"
        write_binary_ply(ply_path, [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)], fmt=PLYFormat.BINARY_BIG_ENDIAN)
        reader = PLYReader(ply_path, chunk_size=1)
        identity_converter(reader)
        chunks = list(reader)
        assert len(chunks) == 2
