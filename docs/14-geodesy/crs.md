# `CRS`

```python
from topocore.geodesy import CRS

crs = CRS.from_epsg(4326)
```

A thin, verified wrapper over `pyproj.CRS`. Direct instantiation is
blocked — you must use one of the 5 factory methods:

```python
>>> CRS()
RuntimeError: CRS cannot be instantiated directly. Use CRS.from_epsg(), CRS.from_wkt(), ...
```

## Construction

| Factory | Source | Verified error path |
|---|---|---|
| `CRS.from_epsg(code)` | EPSG registry code | Rejects negative/bool/non-integer codes with `ValidationError`; a syntactically valid but nonexistent code raises `CRSError` |
| `CRS.from_wkt(wkt)` | WKT string | `CRSError` on invalid WKT |
| `CRS.from_proj4(proj4)` | PROJ4 string | `CRSError` on invalid PROJ4 |
| `CRS.from_authority(authority, code)` | e.g. `("EPSG", 4326)` | `CRSError` on invalid authority/code pair |
| `CRS.from_json(json_str)` | PROJJSON | `CRSError` on invalid JSON |

### `from_epsg`'s own validation is a sibling exception, not `CRSError`

`from_epsg()` calls `validate_epsg()` **before** its own
`try:`/`except CRSError:` block — confirmed directly, an invalid
EPSG code raises `ValidationError`, not `CRSError`, even though both
are the same family (`GeodesyError`). The other 4 factory methods
wrap their entire body uniformly and always raise `CRSError` for
their own invalid input. A caller catching only `except CRSError`
around `from_epsg()` would miss an invalid-code error; catching the
broader `except GeodesyError` catches either.

```python
>>> CRS.from_epsg(-1)
ValidationError: EPSG code must be positive, got -1.

>>> CRS.from_epsg(999999999)
CRSError: Failed to create CRS from EPSG:999999999.
```

## Properties

```python
crs.epsg              # int | None
crs.name               # str
crs.authority          # str | None
crs.is_geographic      # bool
crs.is_projected       # bool
crs.is_vertical        # bool
crs.is_compound        # bool
crs.is_geocentric      # bool
crs.ellipsoid          # Ellipsoid | None
crs.datum              # Datum | None
crs.projection         # ProjectionInfo | None -- None for a geographic CRS (no coordinate operation)
```

### The `ESRI:54032` special case in `authority`

Confirmed directly: a CRS whose authority is the custom `ESRI`
namespace for a specific code (`54032`, a Lambert Azimuthal
Equal-Area projection with no real EPSG equivalent) reports
`authority=None` and `epsg=None` — deliberately excluded, since it's
not a genuine EPSG-registered CRS.

```python
>>> esri_crs = CRS.from_authority("ESRI", "54032")
>>> esri_crs.authority
None
>>> esri_crs.epsg
None
```

## Equality and hashing

```python
>>> CRS.from_epsg(4326) == CRS.from_epsg(4326)
True
>>> CRS.from_epsg(4326) == CRS.from_epsg(32617)
False
>>> CRS.from_epsg(4326) == "not a crs"
False
>>> hash(CRS.from_epsg(4326)) == hash(CRS.from_epsg(4326))
True
```

`.equals(other)` behaves identically to `==`, including returning
`False` (never raising) for a non-`CRS` argument.

## Ellipsoid / datum / projection — real pyproj data, verified

```python
>>> CRS.from_epsg(4326).ellipsoid
Ellipsoid(name='WGS 84', semi_major_axis=6378137.0, ...)
>>> CRS.from_epsg(4326).datum.name
'World Geodetic System 1984 ensemble'
>>> CRS.from_epsg(32617).projection.name
'UTM zone 17N'
>>> CRS.from_epsg(4326).projection
None
```

## `__repr__`

```python
>>> repr(CRS.from_epsg(4326))
"CRS(EPSG:4326, 'WGS 84', Geographic)"
>>> repr(CRS.from_epsg(32617))
"CRS(EPSG:32617, 'WGS 84 / UTM zone 17N', Projected)"
```
