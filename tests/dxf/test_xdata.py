"""
Regression suite for topocore.dxf.xdata -- PR19.

Verified a full encode -> write -> read -> decode round trip against
a real ezdxf-written/read DXF file (not mocks): every scalar type
(str, int, float, bool) preserved exactly including the bool-vs-int
distinction (bool is an int subclass in Python -- confirmed the
encoder correctly checks bool before int), and scalar collections
correctly round-trip as JSON. No bugs found.
"""

from __future__ import annotations

import json

import ezdxf  # type: ignore[import-untyped]

from topocore.dxf.constants import APPID
from topocore.dxf.xdata import XDataDecoder, XDataEncoder


def test_full_round_trip_preserves_types_and_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    doc = ezdxf.new("R2010")
    doc.appids.new(APPID)
    msp = doc.modelspace()
    point = msp.add_point((0, 0, 0))

    payload: dict[str, object] = {
        "schema_version": 1,
        "feature_id": 42,
        "confidence": 0.95,
        "is_verified": True,
        "is_deleted": False,
        "name": "test feature",
        "codes": ["W1", "W2", "W3"],
    }
    point.set_xdata(APPID, XDataEncoder.encode(payload))  # type: ignore[arg-type]

    path = str(tmp_path / "xdata.dxf")
    doc.saveas(path)

    reread = ezdxf.readfile(path)
    reread_point = next(iter(reread.modelspace()))
    tags = reread_point.get_xdata(APPID)
    decoded = XDataDecoder.decode([(t.code, t.value) for t in tags])

    assert decoded["schema_version"] == 1
    assert decoded["feature_id"] == 42
    assert decoded["confidence"] == 0.95
    assert decoded["is_verified"] is True
    assert decoded["is_deleted"] is False
    assert decoded["name"] == "test feature"
    assert json.loads(decoded["codes"]) == ["W1", "W2", "W3"]  # type: ignore[arg-type]


def test_bool_not_confused_with_int() -> None:
    """bool is an int subclass in Python -- confirms encode() checks
    bool before int, so True/False don't get miscoded as 1071 ints."""
    encoded = XDataEncoder.encode({"flag": True})
    codes = [code for code, _ in encoded]
    assert 1070 in codes  # bool group code
    assert 1071 not in codes  # int group code


def test_rejects_non_scalar_dict_value() -> None:
    import pytest

    with pytest.raises(TypeError):
        XDataEncoder.encode({"bad": {"nested": "dict"}})  # type: ignore[dict-item]


def test_decoder_rejects_malformed_odd_length_tags() -> None:
    import pytest

    with pytest.raises(ValueError):
        XDataDecoder.decode([(1000, "key")])  # missing value tag
