"""
topocore.alignment.models
===========================

Horizontal alignment orchestration model.

``LineElement``/``ArcElement``/``HorizontalElement`` live in
``topocore.alignment.elements`` (low-level, no dependency on
algorithms); ``Alignment`` here is the orchestrating model, and is
the only piece of this module that depends on both ``elements`` and
``algorithms`` -- same layering as ``topocore.terrain.tin.TIN``
relative to ``topocore.geometry.point3d.Point3D`` and
``topocore.terrain.algorithms``.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.alignment.algorithms.horizontal import element_length, station_to_point
from topocore.alignment.elements import HorizontalElement
from topocore.alignment.exceptions import (
    AlignmentError,
    AlignmentGeometryError,
    AlignmentStationError,
)
from topocore.alignment.vertical_elements import VerticalElement
from topocore.geometry.point2d import Point2D
from topocore.math.tolerance import is_close


@dataclass(frozen=True, slots=True)
class Alignment:
    """
    An ordered chain of horizontal elements (LandXML ``<Alignment>``
    / ``<CoordGeom>``).

    ``elements`` must chain continuously: element ``i``'s ``end``
    must coincide with element ``i + 1``'s ``start``. This is
    validated at construction -- a discontinuous alignment is
    invalid data, not something to silently accept.
    """

    name: str
    elements: tuple[HorizontalElement, ...]
    start_station: float = 0.0
    desc: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise AlignmentError("Alignment name must not be blank.")

        if not self.elements:
            raise AlignmentError(f"Alignment '{self.name}' must contain at least one element.")

        for index in range(len(self.elements) - 1):
            end_point = self.elements[index].end
            next_start = self.elements[index + 1].start

            if not end_point.almost_equals(next_start):
                raise AlignmentGeometryError(
                    f"Alignment '{self.name}': element {index} ends at "
                    f"({end_point.x}, {end_point.y}) but element {index + 1} starts at "
                    f"({next_start.x}, {next_start.y}) -- horizontal elements must chain "
                    "continuously (end of one == start of the next)."
                )

    @property
    def length(self) -> float:
        """
        Total alignment length: the sum of every element's length.
        """
        return sum(element_length(element) for element in self.elements)

    @property
    def end_station(self) -> float:
        """
        Station at the end of the alignment (``start_station + length``).
        """
        return self.start_station + self.length

    def station_to_point(self, station: float) -> Point2D:
        """
        Compute the XY position at a given station along the alignment.

        Parameters
        ----------
        station
            Station value, in the same coordinate space as
            ``start_station``/``end_station``.

        Returns
        -------
        Point2D

        Raises
        ------
        AlignmentStationError
            If ``station`` falls outside
            ``[start_station, end_station]`` (with tolerance).
        """
        return station_to_point(self.elements, self.start_station, station)


@dataclass(frozen=True, slots=True)
class DesignProfile:
    """
    A design vertical profile (LandXML ``<Profile>``/``<ProfAlign>``):
    an ordered chain of ``GradeSegment``/``VerticalCurve`` elements,
    referenced to an alignment by name (not by object -- avoids
    coupling ``Alignment``/``DesignProfile`` into a reference cycle,
    same reasoning as elsewhere in this package).

    ``elements`` must chain continuously in station, elevation, AND
    grade (slope) -- a design profile with a slope discontinuity
    between a grade segment and an adjoining vertical curve is not
    physically valid, and is rejected at construction rather than
    silently accepted.
    """

    alignment_name: str
    elements: tuple[VerticalElement, ...]
    desc: str | None = None

    def __post_init__(self) -> None:
        if not self.alignment_name.strip():
            raise AlignmentError("DesignProfile.alignment_name must not be blank.")

        if not self.elements:
            raise AlignmentError(f"DesignProfile for '{self.alignment_name}' must contain at least one element.")

        for index in range(len(self.elements) - 1):
            current = self.elements[index]
            following = self.elements[index + 1]

            if not is_close(current.end_station, following.start_station):
                raise AlignmentGeometryError(
                    f"DesignProfile '{self.alignment_name}': element {index} ends at station "
                    f"{current.end_station} but element {index + 1} starts at station "
                    f"{following.start_station} -- vertical elements must chain continuously in station."
                )

            current_end_elevation = current.elevation_at(current.end_station)
            following_start_elevation = following.elevation_at(following.start_station)

            if not is_close(current_end_elevation, following_start_elevation):
                raise AlignmentGeometryError(
                    f"DesignProfile '{self.alignment_name}': element {index} ends at elevation "
                    f"{current_end_elevation} but element {index + 1} starts at elevation "
                    f"{following_start_elevation} -- vertical elements must chain continuously in elevation."
                )

            current_end_grade = current.grade_at(current.end_station)
            following_start_grade = following.grade_at(following.start_station)

            if not is_close(current_end_grade, following_start_grade):
                raise AlignmentGeometryError(
                    f"DesignProfile '{self.alignment_name}': element {index} ends at grade "
                    f"{current_end_grade} but element {index + 1} starts at grade "
                    f"{following_start_grade} -- vertical elements must chain continuously in slope."
                )

    @property
    def start_station(self) -> float:
        return self.elements[0].start_station

    @property
    def end_station(self) -> float:
        return self.elements[-1].end_station

    def elevation_at(self, station: float) -> float:
        """
        Design elevation at ``station``.

        Raises
        ------
        AlignmentStationError
            If ``station`` is not covered by any element.
        """
        for element in self.elements:
            if is_close(station, element.start_station) or is_close(station, element.end_station):
                return element.elevation_at(station)

            if element.start_station < station < element.end_station:
                return element.elevation_at(station)

        raise AlignmentStationError(
            f"Station {station} is not covered by DesignProfile '{self.alignment_name}' "
            f"(range [{self.start_station}, {self.end_station}])."
        )


__all__ = [
    "Alignment",
    "DesignProfile",
]
