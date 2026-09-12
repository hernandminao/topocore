# `UTMZone`

```python
from topocore.geodesy import UTMZone
```

A self-contained, pure-Python UTM zone calculator — no `pyproj` call
needed, since the UTM zoning scheme is a fixed mathematical rule, not
something the PROJ database resolves.

## `UTMZone.from_latlon(latitude, longitude)`

```python
>>> UTMZone.from_latlon(4.7, -74.0)   # Bogotá
UTMZone(zone_number=18, zone_letter='N', hemisphere='N', epsg=32618,
        central_meridian=-75, false_easting=500000.0, false_northing=0.0)
```

### The Norway and Svalbard special cases — verified, not assumed

The UTM grid has two well-known exceptions to the plain
longitude-based zone formula, both implemented and confirmed
directly:

```python
>>> UTMZone.from_latlon(60.0, 5.0).zone_number   # would be zone 31 by longitude alone
32
>>> UTMZone.from_latlon(78.0, 20.0).zone_number  # Svalbard region
33
```

## `UTMZone.from_epsg(epsg)`

Resolves UTM parameters directly from a `326xx`/`327xx`-range EPSG
code (WGS84 UTM North/South). Any other code is rejected:

```python
>>> UTMZone.from_epsg(4326)
CRSError: EPSG:4326 is not a standard UTM zone.
```

## `UTMZone.from_crs(crs)`

Convenience wrapper: extracts `crs.epsg` and calls `from_epsg()`,
returning `None` (never raising) if the CRS has no EPSG code or isn't
a standard UTM zone:

```python
>>> UTMZone.from_crs(CRS.from_epsg(4326))
None
>>> UTMZone.from_crs(CRS.from_epsg(32617)).zone_number
17
```

## What this does *not* do

`UTMZone` only computes and describes zone *parameters* — it never
constructs a `CRS` or performs any coordinate transformation itself.
To actually transform coordinates into a UTM zone this class
identifies, build a `CRS.from_epsg(zone.epsg)` and a
`CoordinateTransformer` yourself.
