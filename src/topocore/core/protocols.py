from typing import Protocol


class Serializable(Protocol):
    def to_wkt(self) -> str: ...
