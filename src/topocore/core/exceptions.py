class TopoCoreError(Exception):
    """Base exception for TopoCore."""


class MathError(TopoCoreError):
    pass


class TopologyError(TopoCoreError):
    pass
