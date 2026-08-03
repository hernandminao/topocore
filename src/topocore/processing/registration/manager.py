"""
topocore.processing.registration.manager
========================================

Registration manager with automatic method selection.

This module provides a high-level manager that selects the appropriate
registration method based on the point cloud characteristics and
user preferences.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar, Final, TypedDict

from topocore.pointcloud.attributes import PointAttribute
from topocore.pointcloud.pointcloud import PointCloud
from topocore.processing.exceptions import RegistrationError

from .base import RegistrationResult
from .icp import ICPBase
from .point_to_plane import PointToPlaneICP
from .point_to_point import PointToPointICP


class ICPParams(TypedDict):
    max_iterations: int
    tolerance: float
    max_correspondence_distance: float
    use_adaptive_distance: bool
    normal_k: int


class RegistrationManager:
    """
    High-level manager for point cloud registration.

    This class provides a unified interface for registration with
    automatic method selection.

    Supported methods:
    - "point_to_point": Point-to-Point ICP (Besl & McKay 1992)
    - "point_to_plane": Point-to-Plane ICP (Chen & Medioni 1992)

    Future methods:
    - "gicp": Generalized ICP
    - "colored": Colored ICP
    - "fgr": Fast Global Registration
    - "ransac": RANSAC feature-based registration

    Examples
    --------
    >>> manager = RegistrationManager(method="point_to_plane", max_iterations=50)
    >>> result = manager.register(source, target)
    >>> aligned_source = result.source_transformed
    """

    __slots__ = ("_method", "_params")

    _SUPPORTED_METHODS: ClassVar[dict[str, type[ICPBase]]] = {
        "point_to_point": PointToPointICP,
        "point_to_plane": PointToPlaneICP,
    }

    _DEFAULT_PARAMS: ClassVar[dict[str, object]] = {
        "max_iterations": 50,
        "tolerance": 1e-6,
        "max_correspondence_distance": 1.0,
        "use_adaptive_distance": True,
        "normal_k": 10,
    }

    _COORDINATE_ERROR_MESSAGE: Final[str] = "Point cloud has no X/Y/Z coordinates."

    def __init__(
        self,
        method: str = "point_to_plane",
        **kwargs: Any,
    ) -> None:
        self._validate_method(method)
        self._method = method
        self._params: dict[str, Any] = dict(kwargs)

    @property
    def method(self) -> str:
        """Get the current registration method."""
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        """Set the current registration method."""
        self._validate_method(value)
        self._method = value

    def set_params(self, **kwargs: Any) -> None:
        """Update parameters for the current method."""
        self._params.update(kwargs)

    def register(
        self,
        source: PointCloud,
        target: PointCloud,
        **kwargs: Any,
    ) -> RegistrationResult:
        """
        Register the source point cloud to the target.

        Parameters
        ----------
        source
            Source point cloud (to be transformed).
        target
            Target point cloud (fixed reference).
        **kwargs
            Additional arguments passed to the registrar.

        Returns
        -------
        RegistrationResult
            The registration result.

        Raises
        ------
        RegistrationError
            If registration fails or input clouds are invalid.
        """
        self._validate_inputs(source, target)
        registrar = self._create_registrar(**kwargs)
        return registrar.register(source, target)

    @classmethod
    def _validate_method(cls, method: str) -> None:
        """Validate that the registration method is supported."""
        if method not in cls._SUPPORTED_METHODS:
            supported = ", ".join(sorted(cls._SUPPORTED_METHODS))
            raise RegistrationError(f"Unsupported method: {method!r}. Supported methods: {supported}.")

    @classmethod
    def _validate_inputs(
        cls,
        source: PointCloud,
        target: PointCloud,
    ) -> None:
        """Validate input point clouds."""
        if source.is_empty:
            raise RegistrationError("Source point cloud is empty.")

        if target.is_empty:
            raise RegistrationError("Target point cloud is empty.")

        cls._validate_coordinates(source, "Source")
        cls._validate_coordinates(target, "Target")

    @classmethod
    def _validate_coordinates(
        cls,
        cloud: PointCloud,
        cloud_name: str,
    ) -> None:
        """Validate that a point cloud contains coordinate attributes."""
        if PointAttribute.X not in cloud.attributes:
            raise RegistrationError(f"{cloud_name} {cls._COORDINATE_ERROR_MESSAGE}")

    def _create_registrar(self, **kwargs: Any) -> ICPBase:
        """Create and configure the appropriate registrar instance."""
        params = {**self._DEFAULT_PARAMS, **self._params, **kwargs}

        try:
            registrar_class = self._SUPPORTED_METHODS[self._method]
        except KeyError as exc:
            raise RegistrationError(f"Unsupported method: {self._method}") from exc

        normalized_params: ICPParams = {
            "max_iterations": int(params["max_iterations"]),
            "tolerance": float(params["tolerance"]),
            "max_correspondence_distance": float(params["max_correspondence_distance"]),
            "use_adaptive_distance": bool(params["use_adaptive_distance"]),
            "normal_k": int(params["normal_k"]),
        }

        signature = inspect.signature(registrar_class.__init__)

        valid_args: dict[str, Any] = {
            name: value for name, value in normalized_params.items() if name in signature.parameters
        }

        return registrar_class(**valid_args)

    def __call__(
        self,
        source: PointCloud,
        target: PointCloud,
        **kwargs: Any,
    ) -> RegistrationResult:
        """Callable interface."""
        return self.register(source, target, **kwargs)


__all__ = ["RegistrationManager"]
