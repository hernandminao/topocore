from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Datum:
    name: str
    remarks: str | None = None
    scope: str | None = None
