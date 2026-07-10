from enum import StrEnum


class GeometryType(StrEnum):
    """Supported geometry types."""

    POINT = "Point"

    SEGMENT = "Segment"

    TRIANGLE = "Triangle"

    POLYGON = "Polygon"

    POLYLINE = "Polyline"

    BOUNDING_BOX = "BoundingBox"

    SURFACE = "Surface"
