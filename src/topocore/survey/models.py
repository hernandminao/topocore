"""
topocore.survey.models
=========================

Field survey point models.

A ``SurveyPoint`` is deliberately not a ``topocore.pointcloud.Chunk``
entry: it carries a free-text field code / description, which has no
equivalent in ``PointAttribute`` (every attribute there is numeric,
matching the LAS/E57/PLY point cloud formats it was designed around).
Field codes are survey metadata, not point cloud geometry -- they
exist to drive automatic linework construction
(``topocore.features``), a concept LiDAR data simply doesn't have.
Keeping the two models separate avoids distorting either one.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from topocore.geodesy.crs import CRS


@dataclass(frozen=True, slots=True)
class SurveyPoint:
    """
    A single field-surveyed point.

    Parameters
    ----------
    id
        Point identifier / name, as recorded in the field (total
        station point number, GNSS point name, etc.).
    x
        Easting.
    y
        Northing.
    z
        Elevation.
    code
        Field code / description (e.g. ``"CERCA1"``, ``"ARBOL"``),
        or ``None`` if the point carries no code.
    """

    id: str
    x: float
    y: float
    z: float
    code: str | None = None


@dataclass(frozen=True, slots=True)
class SurveyPointSet:
    """
    An ordered collection of survey points.

    Order matters: it reflects survey/shot order, which
    ``topocore.features`` relies on to group consecutive same-code
    points into linework.

    ``crs`` is set once, at construction -- ``SurveyPointSet`` is
    frozen, so there is deliberately no setter, unlike
    ``topocore.pointcloud.PointCloud.crs`` (which is mutable and does
    have one). ``None`` means "no CRS known", never "coordinates are
    invalid" -- confirmed throughout this project's own geodesy work,
    an unknown CRS never blocks processing. Set by
    ``topocore.survey.reader.SurveyTXTReader.read()`` when a `.prj`
    sidecar resolves to a real CRS; updated to the target CRS by
    ``topocore.geodesy.transform.transform_survey()`` (matching
    ``PointCloud.crs``'s own post-transform update); left unchanged
    by ``topocore.geodesy.vertical.transform.transform_survey_vertical()``,
    since a vertical-only shift never changes what horizontal CRS the
    `x`/`y` values are already expressed in.
    """

    points: tuple[SurveyPoint, ...]
    crs: CRS | None = None

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[SurveyPoint]:
        return iter(self.points)

    def __getitem__(self, index: int) -> SurveyPoint:
        return self.points[index]

    def codes(self) -> frozenset[str]:
        """
        Every distinct code present in the set.
        """
        return frozenset(point.code for point in self.points if point.code is not None)

    def by_code(self, code: str) -> tuple[SurveyPoint, ...]:
        """
        Every point carrying exactly ``code``, in survey order.
        """
        return tuple(point for point in self.points if point.code == code)


__all__ = [
    "SurveyPoint",
    "SurveyPointSet",
]
