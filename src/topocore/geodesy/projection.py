from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class ProjectionInfo:
    name: str
    method_name: str
    accuracy: float | None = None
    remarks: str | None = None
    scope: str | None = None