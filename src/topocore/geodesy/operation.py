"""
topocore.geodesy.operation
==============================

`CoordinateOperation`: describes a coordinate transformation between
two CRS -- source, target, what kind of operation, its parameters,
and its accuracy. Never executes anything; this is a pure
description, composed from the models PR18A.2 already built
(`GridShift`, `HelmertParameters`) plus `OperationType` and
`TransformationAccuracy` from this PR.

`CoordinateTransformer.from_operation(...)` (a future PR) will
consume this to actually transform coordinates -- see the frozen
PR18A contract: `CoordinateTransformer(source, target)` is never
replaced, only extended with an alternate constructor.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.accuracy import TransformationAccuracy
from topocore.geodesy.crs import CRS
from topocore.geodesy.exceptions import ValidationError
from topocore.geodesy.grid_shift import GridShift
from topocore.geodesy.helmert import HelmertParameters
from topocore.geodesy.operation_type import OperationType


@dataclass(frozen=True, slots=True)
class CoordinateOperation:
    """
    Parameters
    ----------
    source_crs, target_crs
        The two CRS this operation transforms between.
    operation_type
        See `OperationType`. Determines which of `grid_shift`/
        `helmert` must be present -- enforced in `__post_init__`,
        never left to the caller to get right by convention.
    accuracy
        Optional `TransformationAccuracy` for this specific operation.
    grid_shift
        Required (and only meaningful) when `operation_type` is
        `OperationType.GRID_SHIFT`.
    helmert
        Required (and only meaningful) when `operation_type` is
        `OperationType.HELMERT`.
    authority
        Optional authority/registry this operation is defined by
        (e.g. "EPSG:1234"), if it corresponds to a registered operation.
    remarks
        Free-text remarks.
    """

    source_crs: CRS
    target_crs: CRS
    operation_type: OperationType
    accuracy: TransformationAccuracy | None = None
    grid_shift: GridShift | None = None
    helmert: HelmertParameters | None = None
    authority: str | None = None
    remarks: str | None = None

    def __post_init__(self) -> None:
        if self.operation_type is OperationType.GRID_SHIFT:
            if self.grid_shift is None:
                raise ValidationError("CoordinateOperation: operation_type is GRID_SHIFT but grid_shift was not given.")
            if self.helmert is not None:
                raise ValidationError(
                    "CoordinateOperation: operation_type is GRID_SHIFT but helmert was also given "
                    "-- a single operation describes one kind of transformation."
                )

        elif self.operation_type is OperationType.HELMERT:
            if self.helmert is None:
                raise ValidationError("CoordinateOperation: operation_type is HELMERT but helmert was not given.")
            if self.grid_shift is not None:
                raise ValidationError(
                    "CoordinateOperation: operation_type is HELMERT but grid_shift was also given "
                    "-- a single operation describes one kind of transformation."
                )

        elif self.operation_type is OperationType.IDENTITY:
            if self.grid_shift is not None or self.helmert is not None:
                raise ValidationError(
                    "CoordinateOperation: operation_type is IDENTITY but transformation "
                    "parameters (grid_shift/helmert) were given -- an identity operation "
                    "carries none."
                )

        if self.source_crs == self.target_crs and self.operation_type is not OperationType.IDENTITY:
            # GRID_SHIFT always corrects a genuine datum difference --
            # source == target is never meaningful for it.
            #
            # HELMERT is the one exception: a 14-parameter (time-
            # dependent) Helmert transformation is the standard way to
            # correct for tectonic-plate motion between two OBSERVATION
            # EPOCHS of the nominally same datum/CRS (e.g. WGS84 in
            # 2010 vs. WGS84 in 2020, both under EPSG:4326). `CRS`
            # doesn't model observation epoch today, so two CRS
            # instances for the "same" datum at different real epochs
            # still compare equal here -- meaning this check can't
            # currently distinguish a genuinely meaningless Helmert
            # (same CRS, no reason to transform) from a legitimate
            # time-dependent one. If `CRS` ever gains epoch tracking,
            # revisit this: the carve-out below should probably also
            # require the two epochs to actually differ, not just that
            # a reference_epoch was given.
            if (
                self.operation_type is OperationType.HELMERT
                and self.helmert is not None
                and self.helmert.is_time_dependent
            ):
                return

            raise ValidationError(
                f"CoordinateOperation: source_crs and target_crs are the same "
                f"({self.source_crs!r}), but operation_type is "
                f"{self.operation_type.value!r}, not IDENTITY. A time-dependent "
                f"(14-parameter) HELMERT operation is the one exception -- see "
                f"the comment above this check."
            )


__all__ = ["CoordinateOperation"]
