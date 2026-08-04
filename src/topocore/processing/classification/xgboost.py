"""
topocore.processing.classification.xgboost
===========================================

XGBoost classifier for point cloud classification.

XGBoost is an optional dependency, handled with the same lazy-import
strategy as LightGBM (see ``lightgbm.py``).

Limitation
----------
XGBoost's multiclass objective expects non-negative integer labels
and infers the number of classes as ``max(labels) + 1``. ASPRS codes
already satisfy this (0-18), but if the training labels only cover a
sparse subset of codes (e.g. only 2, 6 and 9), XGBoost will still
allocate output slots for every code up to the maximum one, which is
harmless but slightly wasteful. This is not remapped internally so
that predicted labels remain directly comparable to ASPRS codes.

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

import importlib
from typing import Any, override

from topocore.pointcloud.classification import PointClassification
from topocore.processing.exceptions import ClassificationError

from .ml import MachineLearningClassifier

# Se declara como Any para permitir asignarle None sin error [assignment]
_xgb: Any = None

try:
    _xgb = importlib.import_module("xgboost")
    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without xgboost installed
    _XGBOOST_AVAILABLE = False

_NOT_INSTALLED_ERROR = (
    "XGBoost is not installed. Install it with `pip install xgboost` (or `uv add xgboost`) to use XGBoostClassifier."
)


class XGBoostClassifier(MachineLearningClassifier):
    """
    XGBoost classifier for point cloud classification.

    Requires the optional ``xgboost`` package. Use
    :meth:`is_available` to check installation status without
    triggering an exception.

    Parameters
    ----------
    n_estimators
        Number of boosting rounds.
    max_depth
        Maximum tree depth.
    learning_rate
        Shrinkage applied to each tree's contribution.
    subsample
        Fraction of samples used per tree.
    colsample_bytree
        Fraction of features used per tree.
    random_state
        Random seed for reproducibility.
    n_jobs
        Number of parallel threads.
    feature_names
        Features used by the classifier.
    k
        Number of neighbors for geometric features.
    radius
        Radius used for neighborhood computations.
    ground_class
        Classification code representing ground points.

    Raises
    ------
    ClassificationError
        If the ``xgboost`` package is not installed.
    """

    __slots__ = (
        "_n_estimators",
        "_max_depth",
        "_learning_rate",
        "_subsample",
        "_colsample_bytree",
        "_random_state",
        "_n_jobs",
    )

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.3,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int | None = 42,
        n_jobs: int = -1,
        feature_names: list[str] | None = None,
        k: int = 10,
        radius: float = 1.0,
        ground_class: PointClassification = PointClassification.GROUND,
    ) -> None:
        """
        Initialize XGBoost classifier.
        """
        if not _XGBOOST_AVAILABLE:
            raise ClassificationError(_NOT_INSTALLED_ERROR)

        if n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero.")

        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero.")

        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in the range (0, 1].")

        if not 0.0 < colsample_bytree <= 1.0:
            raise ValueError("colsample_bytree must be in the range (0, 1].")

        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._random_state = random_state
        self._n_jobs = n_jobs

        assert _xgb is not None  # narrows type for mypy after the availability check above

        model: Any = _xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            n_jobs=n_jobs,
            eval_metric="mlogloss",
        )

        super().__init__(
            model=model,
            feature_names=feature_names,
            k=k,
            radius=radius,
            ground_class=ground_class,
        )

    @staticmethod
    def is_available() -> bool:
        """
        Return whether the optional XGBoost dependency is installed.
        """
        return _XGBOOST_AVAILABLE

    @override
    def name(self) -> str:
        """
        Return classifier name.
        """
        return "xgboost"
