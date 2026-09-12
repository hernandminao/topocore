"""
topocore.terrain.interpolation
==============================

Terrain interpolation facade.

Provides a unified API for all interpolation methods.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from topocore.geometry.point3d import Point3D
from topocore.terrain.barycentric import BarycentricInterpolator
from topocore.terrain.enums import InterpolationMethod
from topocore.terrain.exceptions import InterpolationError
from topocore.terrain.idw import IDWInterpolator
from topocore.terrain.linear import LinearInterpolator
from topocore.terrain.nearest import NearestInterpolator
from topocore.terrain.tin import TIN


class TerrainInterpolator:
    """
    High-level interpolation interface.

    Examples
    --------
    >>> interpolator = TerrainInterpolator(tin)
    >>> z = interpolator.interpolate(100.0, 200.0)

    >>> interpolator.method = InterpolationMethod.IDW
    >>> z = interpolator.interpolate(100.0, 200.0)
    """

    __slots__ = (
        "_barycentric",
        "_idw",
        "_linear",
        "_method",
        "_nearest",
    )

    def __init__(
        self,
        tin: TIN,
        *,
        method: InterpolationMethod = InterpolationMethod.LINEAR,
        power: float = 2.0,
    ) -> None:

        self._linear = LinearInterpolator(tin)
        self._barycentric = BarycentricInterpolator(tin)
        self._idw = IDWInterpolator(
            tin,
            power=power,
        )
        self._nearest = NearestInterpolator(tin)

        self._method = method

    @property
    def method(
        self,
    ) -> InterpolationMethod:
        return self._method

    @method.setter
    def method(
        self,
        value: InterpolationMethod,
    ) -> None:
        self._method = value

    @property
    def tin(
        self,
    ) -> TIN:
        return self._linear.tin

    def interpolate(
        self,
        x: float,
        y: float,
    ) -> float:
        """
        Interpolate elevation.

        Raises
        ------
        InterpolationError
            If ``self.method`` is not one of the 4 known
            ``InterpolationMethod`` members.

        Notes
        -----
        Found and fixed during this project's own terrain
        documentation audit: this dispatch used to compare
        ``self._method`` with ``is`` and silently fall through to
        ``NEAREST`` for anything that didn't match ``LINEAR``,
        ``BARYCENTRIC``, or ``IDW`` by identity -- including a plain
        string equal to the right value (e.g. ``method="linear"``),
        since ``InterpolationMethod`` is a ``StrEnum`` and
        ``"linear" == InterpolationMethod.LINEAR`` is ``True`` while
        ``"linear" is InterpolationMethod.LINEAR`` is ``False``. This
        is the same failure mode as an earlier, separately-fixed bug
        in this class (2 duplicate ``InterpolationMethod`` enums,
        also compared with ``is``) -- that fix addressed only the
        specific duplicate-enum symptom, not this dispatch's own
        root cause. Now compares with ``==`` and raises explicitly
        for anything else, rather than silently returning a
        plausible-looking but wrong elevation.
        """
        if self._method == InterpolationMethod.LINEAR:
            return self._linear.interpolate(x, y)

        if self._method == InterpolationMethod.BARYCENTRIC:
            return self._barycentric.interpolate(x, y)

        if self._method == InterpolationMethod.IDW:
            return self._idw.interpolate(x, y)

        if self._method == InterpolationMethod.NEAREST:
            return self._nearest.interpolate(x, y)

        raise InterpolationError(f"Unrecognized interpolation method: {self._method!r}.")

    def interpolate_point(
        self,
        point: Point3D,
    ) -> float:
        return self.interpolate(
            point.x,
            point.y,
        )

    def __call__(
        self,
        x: float,
        y: float,
    ) -> float:
        return self.interpolate(
            x,
            y,
        )


__all__ = [
    "InterpolationMethod",
    "TerrainInterpolator",
]
