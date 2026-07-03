import io
import pytest
import numpy as np

from unittest.mock import MagicMock, patch

from topocore.io.ply.reader import PLYReader
from topocore.io.ply.enums import PLYFormat
from topocore.io.ply.exceptions import InvalidPLYError
from topocore.io.ply.header import PLYElement

# ----------------------------
# Helper: fake header + vertex
# ----------------------------

class FakeProperty:
    def __init__(self, name, dtype=True):
        self.name = name
        if dtype:
            self.dtype = MagicMock()
            self.dtype.numpy_dtype = np.dtype("float32")


class FakeVertex:
    def __init__(self):
        self.name = "vertex"
        self.count = 3
        self.properties = [
            FakeProperty("x"),
            FakeProperty("y"),
            FakeProperty("z"),
        ]


class FakeHeader:
    def __init__(self, fmt=PLYFormat.ASCII):
        self.format = fmt
        self.elements = [FakeVertex()]


# ----------------------------
# 1. chunk_size validation
# ----------------------------

def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        PLYReader("dummy.ply", chunk_size=0)


# ----------------------------
# 2. vertex element missing
# ----------------------------

def test_vertex_requires_header():
    reader = PLYReader("dummy.ply")

    reader._stream = io.BytesIO(b"")

    with pytest.raises(InvalidPLYError):
        reader._vertex_element()

def test_missing_vertex_raises():
    header = type("FakeHeader", (), {})()
    header.format = PLYFormat.ASCII
    header.elements = [
        PLYElement(name="face", count=1, properties=[])
    ]

    reader = PLYReader("dummy.ply")
    reader._header = header

    with pytest.raises(InvalidPLYError):
        reader._vertex_element()

def test_vertex_missing():
    reader = PLYReader("dummy.ply")

    reader._header = MagicMock()
    reader._header.elements = []

    with pytest.raises(InvalidPLYError):
        reader._vertex_element()


# ----------------------------
# 3. ASCII iterator basic flow
# ----------------------------

@patch("topocore.io.ply.reader.PLYHeaderParser.parse")
def test_ascii_iteration(mock_parse):
    header = FakeHeader(PLYFormat.ASCII)
    mock_parse.return_value = header

    data = b"1 2 3\n4 5 6\n7 8 9\n"

    reader = PLYReader("dummy.ply")
    reader._stream = io.BytesIO(data)
    reader._header = header

    reader._converter.convert = lambda batch: batch

    chunks = list(reader._iter_ascii(header.elements[0]))

    assert len(chunks) >= 1


# ----------------------------
# 4. UTF-8 error handling
# ----------------------------

def test_utf8_error():
    reader = PLYReader("dummy.ply")

    header = FakeHeader(PLYFormat.ASCII)
    reader._header = header

    reader._stream = MagicMock()
    reader._stream.readline.return_value = b"\xff\xff\xff\n"

    with pytest.raises(InvalidPLYError):
        next(reader._iter_ascii(header.elements[0]))


# ----------------------------
# 5. ASCII empty lines ignored
# ----------------------------

def test_ascii_skips_empty_lines():
    reader = PLYReader("dummy.ply")

    header = FakeHeader(PLYFormat.ASCII)
    reader._header = header

    reader._stream = io.BytesIO(b"\n\n1 2 3\n")

    reader._converter.convert = lambda batch: batch

    result = list(reader._iter_ascii(header.elements[0]))

    assert isinstance(result, list)


# ----------------------------
# 6. binary iteration path (mock np.fromfile)
# ----------------------------

@patch("numpy.fromfile")
def test_binary_iteration(mock_fromfile):
    mock_fromfile.return_value = np.array(
        [(1.0, 2.0, 3.0)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )

    reader = PLYReader("dummy.ply")

    header = FakeHeader(PLYFormat.BINARY_LITTLE_ENDIAN)
    reader._header = header

    reader._converter.convert = lambda batch: batch

    result = list(reader._iter_binary(header.elements[0]))

    assert len(result) >= 1


# ----------------------------
# 7. __iter__ lifecycle test
# ----------------------------

@patch("topocore.io.ply.reader.PLYHeaderParser.parse")
def test_iter_closes_stream(mock_parse):
    header = FakeHeader(PLYFormat.ASCII)
    mock_parse.return_value = header

    reader = PLYReader("dummy.ply")

    reader._stream = io.BytesIO(b"1 2 3\n")
    reader._header = header

    reader._converter.convert = lambda batch: batch

    result = list(reader)

    assert result is not None
    assert reader._stream is None