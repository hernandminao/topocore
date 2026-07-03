from enum import Enum


class GeometryType(str, Enum):
    """Supported geometry types."""

    POINT = "Point"

    SEGMENT = "Segment"

    TRIANGLE = "Triangle"

    POLYGON = "Polygon"

    POLYLINE = "Polyline"

    BOUNDING_BOX = "BoundingBox"

    SURFACE = "Surface"