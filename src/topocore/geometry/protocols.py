from __future__ import annotations

from typing import Protocol


class HasArea(Protocol):
    @property
    def area(self) -> float: ...


class HasLength(Protocol):
    @property
    def length(self) -> float: ...


class HasVolume(Protocol):
    @property
    def volume(self) -> float: ...


class HasCentroid(Protocol):
    @property
    def centroid(self): ...


class Bounded(Protocol):
    def bounding_box(self): ...


class Serializable(Protocol):
    def to_wkt(self) -> str: ...