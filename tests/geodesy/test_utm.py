"""
Regression suite for UTMZone -- PR19, module Geodesy.

Includes a real bug found and fixed while writing this suite: at
longitude=180.0 exactly (explicitly permitted by validate_lat_lon's
inclusive [-180, 180] range), the zone number formula produced 61 --
a UTM zone that does not exist (valid zones are 1-60), yielding a
nonexistent EPSG code 32661. Fixed by clamping to 60, matching how
zone boundaries already behave everywhere else (the western edge of
zone 1 at longitude=-180.0 correctly gives zone 1, not zone 0).
"""

from __future__ import annotations

import pytest

from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import CRSError, ValidationError
from topocore.geodesy.utm import UTMZone

# ----------------------------------------------------------------------
# Standard zone number formula, across the full longitude range.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("longitude", "expected_zone"),
    [
        (-180.0, 1),
        (-179.9999, 1),
        (-174.0001, 1),  # just inside the eastern edge of zone 1
        (-174.0, 2),  # zone 2 spans [-174, -168) -- -174.0 itself is zone 2
        (-173.9999, 2),
        (-3.0, 30),  # zone containing the prime meridian's west side
        (0.0, 31),
        (2.9999, 31),
        (3.0, 31),  # zone 31 spans [0, 6) -- 3.0 is still zone 31, not 32
        (3.0001, 31),
        (177.0, 60),
        (179.9999, 60),
    ],
)
def test_standard_zone_number_across_longitude_range(longitude: float, expected_zone: int) -> None:
    zone = UTMZone.from_latlon(0.0, longitude)
    assert zone.zone_number == expected_zone


def test_antimeridian_edge_case_clamps_to_zone_60_not_61() -> None:
    """
    Regression test for the real bug found in this session: exactly
    longitude=180.0 (valid input per validate_lat_lon's inclusive
    upper bound) must resolve to zone 60 -- zone 61 does not exist,
    and EPSG:32661 is not a real code.
    """
    zone = UTMZone.from_latlon(0.0, 180.0)

    assert zone.zone_number == 60
    assert zone.epsg == 32660


def test_western_edge_still_gives_zone_1_not_0() -> None:
    """
    Sanity check that the fix above didn't accidentally clamp the
    WESTERN boundary too -- longitude=-180.0 must still be zone 1.
    """
    zone = UTMZone.from_latlon(0.0, -180.0)
    assert zone.zone_number == 1


# ----------------------------------------------------------------------
# Hemisphere and EPSG code.
# ----------------------------------------------------------------------


def test_northern_hemisphere_epsg_in_326xx_range() -> None:
    zone = UTMZone.from_latlon(40.0, -75.0)
    assert zone.hemisphere == "N"
    assert 32601 <= zone.epsg <= 32660


def test_southern_hemisphere_epsg_in_327xx_range() -> None:
    zone = UTMZone.from_latlon(-33.9, 151.2)  # Sydney
    assert zone.hemisphere == "S"
    assert 32701 <= zone.epsg <= 32760


def test_equator_itself_is_northern_hemisphere() -> None:
    """latitude >= 0.0 -- the equator itself belongs to 'N'."""
    zone = UTMZone.from_latlon(0.0, 10.0)
    assert zone.hemisphere == "N"


# ----------------------------------------------------------------------
# Norway special case: 56<=lat<64, 3<=lon<12 -> forced to zone 32.
# ----------------------------------------------------------------------


def test_norway_special_case_forces_zone_32() -> None:
    # Bergen, Norway -- would naturally compute to a different zone
    # via the standard formula, but the Norway carve-out forces 32.
    zone = UTMZone.from_latlon(60.0, 5.0)
    assert zone.zone_number == 32


def test_norway_special_case_lower_latitude_boundary() -> None:
    zone_in = UTMZone.from_latlon(56.0, 5.0)  # inclusive lower bound
    zone_out = UTMZone.from_latlon(55.9999, 5.0)  # just outside

    assert zone_in.zone_number == 32
    assert zone_out.zone_number != 32  # standard formula applies instead


def test_norway_special_case_upper_latitude_boundary_exclusive() -> None:
    zone_in = UTMZone.from_latlon(63.9999, 5.0)
    zone_out = UTMZone.from_latlon(64.0, 5.0)  # exclusive upper bound

    assert zone_in.zone_number == 32
    assert zone_out.zone_number != 32


def test_norway_special_case_does_not_apply_outside_longitude_band() -> None:
    zone = UTMZone.from_latlon(60.0, 2.9999)  # just west of the 3-12 band
    assert zone.zone_number != 32


# ----------------------------------------------------------------------
# Svalbard special cases: 72<=lat<=84, longitude bands -> 31/33/35/37.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("longitude", "expected_zone"),
    [
        (5.0, 31),  # 0 <= lon < 9
        (8.9999, 31),
        (9.0, 33),  # 9 <= lon < 21
        (20.9999, 33),
        (21.0, 35),  # 21 <= lon < 33
        (32.9999, 35),
        (33.0, 37),  # 33 <= lon < 42
        (41.9999, 37),
    ],
)
def test_svalbard_special_case_longitude_bands(longitude: float, expected_zone: int) -> None:
    zone = UTMZone.from_latlon(78.0, longitude)  # well within 72-84
    assert zone.zone_number == expected_zone


def test_svalbard_special_case_latitude_boundaries() -> None:
    zone_in_lower = UTMZone.from_latlon(72.0, 10.0)  # inclusive
    zone_out_lower = UTMZone.from_latlon(71.9999, 10.0)
    zone_in_upper = UTMZone.from_latlon(84.0, 10.0)  # inclusive

    assert zone_in_lower.zone_number == 33
    assert zone_out_lower.zone_number != 33  # standard/Norway rules apply instead
    assert zone_in_upper.zone_number == 33


def test_svalbard_special_case_does_not_apply_for_negative_longitude() -> None:
    zone = UTMZone.from_latlon(78.0, -10.0)
    assert zone.zone_number != 31  # Svalbard carve-out requires longitude >= 0


# ----------------------------------------------------------------------
# MGRS latitude band letters.
# ----------------------------------------------------------------------


def test_letter_band_south_edge_is_c() -> None:
    zone = UTMZone.from_latlon(-80.0, 10.0)
    assert zone.zone_letter == "C"


def test_letter_band_north_edge_is_x() -> None:
    """
    The X band is double-width (72-84, 12 degrees instead of the
    usual 8) -- both its start and its end (84.0, the maximum
    latitude UTM covers) must resolve to 'X', not crash or roll
    into a nonexistent 21st band.
    """
    zone_start = UTMZone.from_latlon(72.0, 10.0)
    zone_end = UTMZone.from_latlon(84.0, 10.0)

    assert zone_start.zone_letter == "X"
    assert zone_end.zone_letter == "X"


def test_letter_band_empty_outside_utm_coverage() -> None:
    """
    Beyond +/-80/84 degrees, UTM doesn't define a letter band --
    the field must be an empty string, not crash or return a bogus
    letter, for latitudes validate_lat_lon still permits (up to 90).
    """
    zone = UTMZone.from_latlon(89.0, 10.0)
    assert zone.zone_letter == ""


def test_letter_band_boundary_between_adjacent_bands() -> None:
    # W band: 64-72 (index 18); X band starts at 72 (index 19).
    zone_w = UTMZone.from_latlon(71.9999, 10.0)
    zone_x = UTMZone.from_latlon(72.0, 10.0)

    assert zone_w.zone_letter == "W"
    assert zone_x.zone_letter == "X"


# ----------------------------------------------------------------------
# central_meridian / false_easting / false_northing.
# ----------------------------------------------------------------------


def test_central_meridian_formula() -> None:
    zone = UTMZone.from_latlon(0.0, 3.0)  # zone 31 (or 32 if in Norway band -- lat=0 isn't)
    assert zone.central_meridian == pytest.approx((zone.zone_number * 6) - 183)


def test_false_easting_always_500000() -> None:
    assert UTMZone.from_latlon(10.0, 10.0).false_easting == 500000.0
    assert UTMZone.from_latlon(-10.0, 10.0).false_easting == 500000.0


def test_false_northing_zero_in_north_and_10million_in_south() -> None:
    north = UTMZone.from_latlon(10.0, 10.0)
    south = UTMZone.from_latlon(-10.0, 10.0)

    assert north.false_northing == 0.0
    assert south.false_northing == 10000000.0


# ----------------------------------------------------------------------
# from_epsg
# ----------------------------------------------------------------------


def test_from_epsg_northern_zone() -> None:
    zone = UTMZone.from_epsg(32633)  # UTM 33N
    assert zone.zone_number == 33
    assert zone.hemisphere == "N"


def test_from_epsg_southern_zone() -> None:
    zone = UTMZone.from_epsg(32733)  # UTM 33S
    assert zone.zone_number == 33
    assert zone.hemisphere == "S"


def test_from_epsg_rejects_non_utm_code() -> None:
    with pytest.raises(CRSError):
        UTMZone.from_epsg(4326)  # WGS84 geographic, not a UTM projected zone


def test_from_epsg_rejects_out_of_range_utm_like_code() -> None:
    with pytest.raises(CRSError):
        UTMZone.from_epsg(32661)  # the exact nonexistent code the bug above produced


# ----------------------------------------------------------------------
# from_crs
# ----------------------------------------------------------------------


def test_from_crs_resolves_real_utm_crs() -> None:
    crs = CRS.from_epsg(32633)
    zone = UTMZone.from_crs(crs)

    assert zone is not None
    assert zone.zone_number == 33
    assert zone.hemisphere == "N"


def test_from_crs_returns_none_for_non_utm_crs() -> None:
    crs = CRS.from_epsg(4326)  # geographic, no EPSG-derivable UTM zone
    assert UTMZone.from_crs(crs) is None


# ----------------------------------------------------------------------
# Input validation (delegated to validate_lat_lon).
# ----------------------------------------------------------------------


def test_rejects_out_of_range_latitude() -> None:
    with pytest.raises(ValidationError):
        UTMZone.from_latlon(91.0, 0.0)


def test_rejects_out_of_range_longitude() -> None:
    with pytest.raises(ValidationError):
        UTMZone.from_latlon(0.0, 181.0)
