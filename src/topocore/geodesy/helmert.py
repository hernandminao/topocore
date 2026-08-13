"""
topocore.geodesy.helmert
============================

Represents Helmert transformation parameters (7-parameter or
14-parameter, same class) -- pure data, no transformation logic.
Applying a Helmert transformation belongs to a later PR, once
`CoordinateOperation` exists to describe when/how it's used.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class HelmertParameters:
    """
    7-parameter Helmert transformation (translation + rotation +
    scale), optionally extended to 14 parameters with time-dependent
    rates and a reference epoch -- the same class covers both; the
    rate fields are simply `None` for a static (7-parameter)
    transformation.

    Parameters
    ----------
    tx, ty, tz
        Translation, in meters.
    rx, ry, rz
        Rotation, in arc-seconds. Default 0.0 (no rotation) --
        translation-only transformations are common enough that this
        is a genuine identity default, not a guessed algorithm choice.
    scale
        Scale difference, in parts per million (ppm). Default 0.0
        (no scale change).
    dtx, dty, dtz, drx, dry, drz, dscale
        Time-dependent rates (per year), for the 14-parameter form.
        `None` when not applicable.
    reference_epoch
        Reference epoch (decimal year, e.g. 2018.0) the rates are
        relative to. `None` for a static (7-parameter) transformation.
        Required if any rate field is given -- a rate without a
        reference point in time is meaningless.
    """

    tx: float
    ty: float
    tz: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    scale: float = 0.0
    dtx: float | None = None
    dty: float | None = None
    dtz: float | None = None
    drx: float | None = None
    dry: float | None = None
    drz: float | None = None
    dscale: float | None = None
    reference_epoch: float | None = None

    def __post_init__(self) -> None:
        rate_fields = (
            self.dtx,
            self.dty,
            self.dtz,
            self.drx,
            self.dry,
            self.drz,
            self.dscale,
        )
        if any(field is not None for field in rate_fields) and self.reference_epoch is None:
            raise ValidationError(
                "HelmertParameters: time-dependent rate parameters were given "
                "without a reference_epoch -- a rate is meaningless without a "
                "reference point in time."
            )

    @property
    def is_time_dependent(self) -> bool:
        """True for the 14-parameter form (has a reference_epoch)."""
        return self.reference_epoch is not None

    @property
    def parameter_count(self) -> int:
        """7 for a static transformation, 14 for a time-dependent one -- derived from `is_time_dependent`, never stored separately."""
        return 14 if self.is_time_dependent else 7


__all__ = ["HelmertParameters"]
