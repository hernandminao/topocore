# `GeodesicCalculator`

```python
from topocore.geodesy import CRS, GeodesicCalculator

crs = CRS.from_epsg(4326)
geod = GeodesicCalculator(crs)
```

A thin wrapper over `pyproj.Geod`, built from a `CRS`'s own
ellipsoid — every calculation genuinely uses that specific
ellipsoid's own semi-major axis and inverse flattening, not a
hardcoded WGS84 assumption.

```python
>>> GeodesicCalculator(crs_with_no_ellipsoid)
GeodesicError: Failed to initialize Geodesic calculator.
```

## Verified against a known real-world distance

The great-circle (ellipsoidal geodesic) distance between Bogotá and
Medellín is approximately 240 km. Confirmed directly:

```python
>>> geod.distance(-74.0721, 4.7110, -75.5636, 6.2476)
237043.87134926795  # meters — ≈237 km, matching the known real-world figure
```

## Methods

```python
geod.distance(lon1, lat1, lon2, lat2) -> float          # meters
geod.azimuth(lon1, lat1, lon2, lat2) -> float            # degrees, forward azimuth
geod.inverse(lon1, lat1, lon2, lat2) -> (az1, az2, distance)
geod.forward(lon, lat, azimuth, distance) -> (lon2, lat2, back_azimuth)
geod.polygon_area(lons, lats) -> float                   # square meters, always positive
```

Every method wraps its own `pyproj.Geod` call in a `try`/`except`,
re-raising as the appropriately-named `GeodesicError` (e.g.
`"Geodesic distance calculation failed."`) rather than letting a raw
`pyproj`/`proj` error escape.

## What this does *not* do

`GeodesicCalculator` computes distances/azimuths/areas directly on
the ellipsoid from longitude/latitude — it does not project
coordinates, does not accept projected (`x, y`) input, and has no
relationship to `CoordinateTransformer`. If your data is in a
projected CRS, transform to a geographic one first (see
[`transformer.md`](./transformer.md)) before using this class.
