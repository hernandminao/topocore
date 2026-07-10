"""
Unit tests for topocore.geodesy (100% coverage).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from topocore.geodesy._cache import (
    cache_info,
    clear_cache,
)
from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import (
    CRSError,
    GeodesicError,
    TransformationError,
    ValidationError,
)
from topocore.geodesy.geodesic import GeodesicCalculator
from topocore.geodesy.transformer import CoordinateTransformer
from topocore.geodesy.utm import UTMZone
from topocore.geodesy.validation import (
    validate_array,
    validate_bbox,
    validate_coordinate_arrays,
    validate_epsg,
)


class TestCRS:
    def test_init_raises_error(self) -> None:
        with pytest.raises(
            RuntimeError,
            match="cannot be instantiated directly",
        ):
            CRS()  # type: ignore

    def test_from_epsg_valid(self) -> None:
        crs = CRS.from_epsg(4326)

        assert crs.is_geographic
        assert crs.epsg == 4326
        assert crs.authority == ("EPSG", "4326")

    def test_from_epsg_invalid(self) -> None:
        with pytest.raises(CRSError):
            CRS.from_epsg(9999999)

    def test_from_wkt_valid(self) -> None:
        wkt = (
            'GEOGCS["WGS 84",'
            'DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],'
            'UNIT["degree",0.0174532925199433]]'
        )

        crs = CRS.from_wkt(wkt)

        assert crs.is_geographic

    def test_from_wkt_invalid(self) -> None:
        with pytest.raises(CRSError):
            CRS.from_wkt("invalid_wkt")

    def test_from_proj4_valid(self) -> None:
        crs = CRS.from_proj4("+proj=longlat +datum=WGS84 +no_defs")

        assert crs.is_geographic

    def test_from_proj4_invalid(self) -> None:
        with pytest.raises(CRSError):
            CRS.from_proj4("invalid_proj")

    def test_from_authority_valid(self) -> None:
        crs = CRS.from_authority("EPSG", 4326)

        assert crs.epsg == 4326

    def test_from_authority_invalid(self) -> None:
        with pytest.raises(CRSError):
            CRS.from_authority("FAKE", 123)

    def test_from_json_valid(self) -> None:
        json_str = (
            '{"type":"GeographicCRS",'
            '"name":"WGS 84",'
            '"datum":{'
            '"type":"GeodeticReferenceFrame",'
            '"name":"World Geodetic System 1984",'
            '"ellipsoid":{'
            '"name":"WGS 84",'
            '"semi_major_axis":6378137,'
            '"inverse_flattening":298.257223563'
            "}},"
            '"coordinate_system":{'
            '"subtype":"ellipsoidal",'
            '"axis":['
            '{"name":"Geodetic latitude",'
            '"abbreviation":"Lat",'
            '"direction":"north",'
            '"unit":"degree"},'
            '{"name":"Geodetic longitude",'
            '"abbreviation":"Lon",'
            '"direction":"east",'
            '"unit":"degree"}'
            "]}"
            "}"
        )

        crs = CRS.from_json(json_str)

        assert crs.is_geographic

    def test_from_json_invalid(self) -> None:
        with pytest.raises(CRSError):
            CRS.from_json('{"invalid":"json"}')

    def test_immutability(self) -> None:
        crs = CRS.from_epsg(4326)

        with pytest.raises(AttributeError):
            crs._native = None  # type: ignore[attr-defined]

    def test_repr_hash_equals(self) -> None:
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_epsg(4326)

        assert isinstance(repr(crs1), str)
        assert "4326" in repr(crs1)
        assert hash(crs1) == hash(crs2)
        assert crs1 == crs2
        assert crs1.equals(crs2)
        assert not crs1.equals("not_a_crs")

    def test_equals_ignore_axis_order(self) -> None:
        crs1 = CRS.from_epsg(4326)
        crs2 = CRS.from_epsg(4326)

        assert crs1.equals(
            crs2,
            ignore_axis_order=True,
        )

    def test_export_methods(self) -> None:
        crs = CRS.from_epsg(4326)

        assert isinstance(crs.to_wkt(), str)
        assert isinstance(crs.to_proj4(), str)
        assert isinstance(crs.to_json(), str)

    def test_compound_and_geocentric(self) -> None:
        # EPSG:9518 is WGS 84 + EGM96 height (Compound CRS)
        compound = CRS.from_epsg(9518)

        assert compound.is_compound
        assert compound.is_vertical

        geocentric = CRS.from_epsg(4978)

        assert geocentric.is_geocentric

    def test_datum_and_projection_info(self) -> None:
        crs = CRS.from_epsg(3116)

        assert crs.datum is not None
        assert crs.projection is not None
        assert crs.projection.method_name is not None

    def test_authority_none(self) -> None:
        crs = CRS.from_proj4("+proj=aeqd +lat_0=0 +lon_0=0")

        assert crs.authority is None
        assert crs.epsg is None

    def test_repr_projected(self) -> None:
        crs = CRS.from_epsg(3116)

        assert "Projected" in repr(crs)

    def test_repr_geographic(self) -> None:
        crs = CRS.from_epsg(4326)

        assert "Geographic" in repr(crs)

    def test_repr_geocentric(self) -> None:
        crs = CRS.from_epsg(4978)

        assert "Geocentric" in repr(crs)

    def test_repr_compound(self) -> None:
        crs = CRS.from_epsg(9518)

        assert "Compound" in repr(crs)


class TestValidations:
    def test_validate_epsg_valid(self) -> None:
        validate_epsg(4326)

    def test_validate_epsg_invalid(self) -> None:
        with pytest.raises(ValidationError):
            validate_epsg(-1)

        with pytest.raises(ValidationError):
            validate_epsg("4326")  # type: ignore[arg-type]

        with pytest.raises(ValidationError):
            validate_epsg(True)  # type: ignore[arg-type]

    def test_validate_bbox_valid(self) -> None:
        validate_bbox((0.0, 0.0, 10.0, 10.0))

    def test_validate_bbox_invalid(self) -> None:
        with pytest.raises(ValidationError):
            validate_bbox((0.0, 0.0, -10.0, 10.0))

        with pytest.raises(ValidationError):
            validate_bbox((0.0, 0.0, 10.0))  # type: ignore[arg-type]

    def test_validate_array_valid(self) -> None:
        validate_array(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            2,
        )

    def test_validate_array_invalid(self) -> None:
        with pytest.raises(ValidationError):
            validate_array(
                np.array([1.0, 2.0, 3.0]),
                2,
            )

    def test_validate_coordinate_arrays_valid(self) -> None:
        x, y = validate_coordinate_arrays(
            [1, 2],
            [3, 4],
        )
        assert len(x) == 2

    def test_validate_coordinate_arrays_invalid_length(self) -> None:
        with pytest.raises(
            ValidationError,
        ):
            validate_coordinate_arrays(
                [1],
                [1, 2],
            )

    def test_validate_coordinate_arrays_nan(self) -> None:
        with pytest.raises(
            ValidationError,
        ):
            validate_coordinate_arrays(
                [1, np.nan],
                [2, 3],
            )


class TestUTMZone:
    def test_from_latlon_north(self) -> None:
        zone = UTMZone.from_latlon(
            4.0,
            -74.0,
        )

        assert zone.zone_number == 18
        assert zone.hemisphere == "N"
        assert zone.epsg == 32618

    def test_from_latlon_south(self) -> None:
        zone = UTMZone.from_latlon(
            -4.0,
            -74.0,
        )

        assert zone.hemisphere == "S"

    def test_from_latlon_norway(self) -> None:
        zone = UTMZone.from_latlon(
            60.0,
            5.0,
        )

        assert zone.zone_number == 32

    def test_from_latlon_svalbard(self) -> None:
        zone = UTMZone.from_latlon(
            75.0,
            15.0,
        )

        assert zone.zone_number == 33

    def test_from_latlon_invalid_lat(self) -> None:
        with pytest.raises(
            ValidationError,
            match="Latitude",
        ):
            UTMZone.from_latlon(
                100.0,
                0.0,
            )

    def test_from_latlon_invalid_lon(self) -> None:
        with pytest.raises(
            ValidationError,
            match="Longitude",
        ):
            UTMZone.from_latlon(
                0.0,
                200.0,
            )

    def test_from_crs_valid(self) -> None:
        crs = CRS.from_epsg(32618)

        zone = UTMZone.from_crs(crs)

        assert zone is not None
        assert zone.zone_number == 18

    def test_from_crs_not_utm(self) -> None:
        crs = CRS.from_epsg(4326)

        zone = UTMZone.from_crs(crs)

        assert zone is None

    def test_from_epsg_valid(self) -> None:
        zone = UTMZone.from_epsg(32719)

        assert zone.zone_number == 19
        assert zone.hemisphere == "S"

    def test_from_epsg_invalid(self) -> None:
        with pytest.raises(CRSError):
            UTMZone.from_epsg(4326)


class TestCoordinateTransformer:
    def test_transform_point_no_z(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        x, y, z = transformer.transform_point(
            -74.0,
            4.6,
        )

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert z is None

    def test_transform_point_with_z(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        x, y, z = transformer.transform_point(
            -74.0,
            4.6,
            2600.0,
        )

        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)

    def test_transform_array_no_z(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        x, y, z = transformer.transform_array(
            [-74.0, -74.1],
            [4.6, 4.7],
        )

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == 2
        assert len(y) == 2
        assert z is None

    def test_transform_array_with_z(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        x, y, z = transformer.transform_array(
            [-74.0, -74.1],
            [4.6, 4.7],
            [2600.0, 2600.0],
        )

        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert isinstance(z, np.ndarray)
        assert len(z) == 2

    def test_transform_bbox(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        bbox = (
            -74.0,
            4.6,
            -73.9,
            4.7,
        )

        minx, miny, maxx, maxy = transformer.transform_bbox(
            bbox,
        )

        assert minx < maxx
        assert miny < maxy

    def test_transform_bbox_invalid(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        transformer = CoordinateTransformer(
            source,
            target,
        )

        with pytest.raises(ValidationError):
            transformer.transform_bbox(
                (
                    0.0,
                    0.0,
                    -1.0,
                    1.0,
                )
            )

    def test_transform_invalid_crs(self) -> None:
        source = CRS.from_epsg(4326)
        target = CRS.from_epsg(3116)

        # Simulate a failure in the underlying pyproj library
        with (
            patch(
                "topocore.geodesy._cache.get_transformer",
                side_effect=Exception("Mocked pyproj error"),
            ),
            pytest.raises(TransformationError),
        ):
            CoordinateTransformer(source, target)


class TestGeodesicCalculator:
    def test_distance_valid(self) -> None:
        crs = CRS.from_epsg(4326)
        calc = GeodesicCalculator(crs)
        dist = calc.distance(-74.0, 4.6, -75.5, 6.2)
        assert dist > 200000.0

    def test_azimuth_valid(self) -> None:
        crs = CRS.from_epsg(4326)
        calc = GeodesicCalculator(crs)
        az = calc.azimuth(-74.0, 4.6, -74.0, 5.0)
        assert az == pytest.approx(0.0, abs=1e-5)

    def test_inverse_valid(self) -> None:
        crs = CRS.from_epsg(4326)
        calc = GeodesicCalculator(crs)
        az1, az2, dist = calc.inverse(-74.0, 4.6, -75.5, 6.2)
        assert dist > 200000.0

    def test_forward_valid(self) -> None:
        crs = CRS.from_epsg(4326)
        calc = GeodesicCalculator(crs)
        lon, lat, _ = calc.forward(-74.0, 4.6, 0.0, 100000.0)
        assert lat > 4.6

    def test_polygon_area_valid(self) -> None:
        crs = CRS.from_epsg(4326)
        calc = GeodesicCalculator(crs)

        lons = [-74.0, -74.0, -73.9, -73.9]
        lats = [4.6, 4.7, 4.7, 4.6]

        area = calc.polygon_area(lons, lats)

        assert area > 100000.0

    def test_invalid_crs(self) -> None:
        with pytest.raises(GeodesicError):
            crs = CRS.from_epsg(5773)
            GeodesicCalculator(crs)


class TestDistance:
    def test_distance2d(self) -> None:
        from topocore.geodesy.distance import distance2d

        assert (
            distance2d(
                0.0,
                0.0,
                3.0,
                4.0,
            )
            == 5.0
        )

    def test_distance3d(self) -> None:
        from topocore.geodesy.distance import distance3d

        assert (
            distance3d(
                0.0,
                0.0,
                0.0,
                3.0,
                4.0,
                12.0,
            )
            == 13.0
        )


class TestCache:
    def test_cache(self) -> None:
        clear_cache()

        CRS.from_epsg(4326)
        CRS.from_epsg(4326)

        info = cache_info()

        assert info["crs"].currsize >= 1
