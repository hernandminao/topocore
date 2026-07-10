class TopoCoreError(Exception):
    """Base exception for TopoCore."""


class GeometryError(TopoCoreError):
    pass


class MathError(TopoCoreError):
    pass


class TopologyError(TopoCoreError):
    pass


class TerrainError(TopoCoreError):
    pass
