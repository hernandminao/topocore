"""
Regression suite for topocore.geodesy.crs.CRS -- PR19.
"""

from __future__ import annotations

import pytest

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import CRSError, ValidationError


def test_cannot_instantiate_directly() -> None:
    with pytest.raises(RuntimeError):
        CRS()


def test_from_epsg_wgs84() -> None:
    crs = CRS.from_epsg(4326)
    assert crs.epsg == 4326
    assert crs.is_geographic is True
    assert crs.is_projected is False


def test_from_epsg_utm_is_projected() -> None:
    crs = CRS.from_epsg(32617)
    assert crs.is_projected is True
    assert crs.is_geographic is False


def test_from_epsg_rejects_invalid_code() -> None:
    with pytest.raises(ValidationError):
        CRS.from_epsg(-1)


def test_from_epsg_rejects_nonexistent_code() -> None:
    with pytest.raises(CRSError):
        CRS.from_epsg(999999999)


def test_from_wkt_round_trips() -> None:
    original = CRS.from_epsg(4326)
    wkt = original.to_wkt()

    reconstructed = CRS.from_wkt(wkt)

    assert reconstructed.epsg == original.epsg


def test_from_proj4() -> None:
    crs = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")
    assert crs.is_geographic is True


def test_from_authority() -> None:
    crs = CRS.from_authority("EPSG", 4326)
    assert crs.epsg == 4326


def test_from_authority_accepts_int_or_str_code() -> None:
    from_int = CRS.from_authority("EPSG", 4326)
    from_str = CRS.from_authority("EPSG", "4326")
    assert from_int.epsg == from_str.epsg


def test_from_json_round_trips() -> None:
    original = CRS.from_epsg(4326)
    json_str = original.to_json()

    reconstructed = CRS.from_json(json_str)

    assert reconstructed.epsg == original.epsg


def test_ellipsoid_returns_real_wgs84_values() -> None:
    crs = CRS.from_epsg(4326)
    ellipsoid = crs.ellipsoid

    assert ellipsoid is not None
    assert ellipsoid.semi_major_axis == pytest.approx(6378137.0)
    assert ellipsoid.inverse_flattening == pytest.approx(298.257223563, rel=1e-6)


def test_equality_same_epsg() -> None:
    assert CRS.from_epsg(4326) == CRS.from_epsg(4326)


def test_equality_different_epsg() -> None:
    assert CRS.from_epsg(4326) != CRS.from_epsg(32617)


def test_equality_against_non_crs_object() -> None:
    assert (CRS.from_epsg(4326) == "not a crs") is False


def test_hash_consistent_with_equality() -> None:
    a = CRS.from_epsg(4326)
    b = CRS.from_epsg(4326)
    assert hash(a) == hash(b)


def test_repr_includes_epsg_and_type() -> None:
    text = repr(CRS.from_epsg(4326))
    assert "4326" in text
    assert "Geographic" in text


def test_name_is_nonempty() -> None:
    crs = CRS.from_epsg(4326)
    assert crs.name  # non-empty string


def test_equals_ignores_axis_order_when_requested() -> None:
    """
    EPSG:4326 (lat,lon axis order per its authority definition) and
    a proj4-built equivalent (typically lon,lat) should compare
    equal when axis order is ignored, since both describe the same
    geographic system.
    """
    epsg_crs = CRS.from_epsg(4326)
    proj4_crs = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")

    assert epsg_crs.equals(proj4_crs, ignore_axis_order=True) is True
