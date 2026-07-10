"""
topocore.geodesy._cache
=======================

Internal caching mechanisms for heavy geodesy objects.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from functools import lru_cache

from pyproj import CRS as PyprojCRS
from pyproj import Geod
from pyproj import Transformer as PyprojTransformer


#
# NOTE
# ----
# CRS, Transformer and Geod objects are immutable after creation and
# relatively expensive to instantiate. Keeping a shared cache
# significantly improves performance in GIS workloads that repeatedly
# use the same reference systems.
#


@lru_cache(maxsize=1024)
def get_crs(epsg: int) -> PyprojCRS:
    """
    Return a cached CRS.

    Parameters
    ----------
    epsg
        EPSG code.

    Returns
    -------
    pyproj.CRS
    """
    return PyprojCRS.from_epsg(epsg)


@lru_cache(maxsize=1024)
def get_transformer(
    source_epsg: int,
    target_epsg: int,
) -> PyprojTransformer:
    """
    Return a cached coordinate transformer.

    Parameters
    ----------
    source_epsg
        Source CRS EPSG.

    target_epsg
        Target CRS EPSG.

    Returns
    -------
    pyproj.Transformer
    """
    return PyprojTransformer.from_crs(
        f"EPSG:{source_epsg}",
        f"EPSG:{target_epsg}",
        always_xy=True,
    )


@lru_cache(maxsize=1024)
def get_geod(
    semi_major_axis: float,
    inverse_flattening: float,
) -> Geod:
    """
    Return a cached Geod instance.

    Parameters
    ----------
    semi_major_axis
        Ellipsoid semi-major axis.

    inverse_flattening
        Ellipsoid inverse flattening.

    Returns
    -------
    pyproj.Geod
    """
    return Geod(
        a=semi_major_axis,
        rf=inverse_flattening,
    )


def clear_cache() -> None:
    """
    Clear every internal cache.

    Intended primarily for unit tests.
    """
    get_crs.cache_clear()
    get_transformer.cache_clear()
    get_geod.cache_clear()


def cache_info() -> dict[str, object]:
    """
    Return cache statistics.

    Returns
    -------
    dict
        Cache information for all cached objects.
    """
    return {
        "crs": get_crs.cache_info(),
        "transformer": get_transformer.cache_info(),
        "geod": get_geod.cache_info(),
    }


__all__ = [
    "get_crs",
    "get_transformer",
    "get_geod",
    "clear_cache",
    "cache_info",
]