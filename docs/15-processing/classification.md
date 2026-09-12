# Classification

Assigns ASPRS classification codes (2=Ground, 3/4/5=Low/Medium/High
Vegetation, 6=Building, 7=Noise, per the standard ASPRS LAS
convention) to every point — one rule-based classifier, plus 4
machine-learning backends sharing a common `fit`/`classify` contract.

```python
from topocore.processing.classification import RuleBasedClassifier

result = RuleBasedClassifier().classify(cloud)
result.labels           # IntArray1D, shape (N,) -- ASPRS codes
result.get_class_mask(2)  # BoolArray1D -- True where classified as Ground
```

Verified directly on a synthetic 3-tier cloud (ground, low, and high
elevation bands): `RuleBasedClassifier().classify(cloud)` correctly
separates ground (code 2) and high vegetation (code 5); points with
ambiguous local geometry can resolve to "unclassified" (code 1)
rather than being forced into a class the geometric rules don't
confidently support.

## `RuleBasedClassifier` — 19 tunable thresholds, no training data needed

```python
RuleBasedClassifier(
    ground_cell_size=1.0, ground_max_height=0.2,
    low_vegetation_max_height=0.5, high_vegetation_min_height=2.0,
    curvature_threshold=0.02, noise_radius=1.0, noise_min_neighbors=4,
    pca_neighbors=10, vegetation_min_height=0.1,
    vegetation_planarity_threshold=0.6, vegetation_linearity_threshold=0.8,
    vegetation_sphericity_threshold=0.2, vegetation_max_anisotropy=0.8,
    vegetation_min_omnivariance=0.01, building_planarity_threshold=0.5,
    building_sphericity_threshold=0.3, building_min_verticality=0.3,
    building_max_verticality=0.7, max_building_height=50.0,
)
```

Every threshold has a real default and can be tuned per dataset — it
runs `ground` classification and `features`'s PCA-derived descriptors
(planarity, sphericity, anisotropy, omnivariance) internally, then
applies hand-crafted geometric rules. No labeled training data is
required, unlike the 4 ML backends below.

## The 4 ML backends — a shared `fit`/`classify` contract, 2 optional dependencies

```python
classifier = RandomForestClassifier()
classifier.fit(cloud, labels)       # labels: IntArray1D of known ASPRS codes, same length as cloud
result = classifier.classify(cloud)  # or a different cloud with the same feature characteristics
```

```python
RandomForestClassifier()    # scikit-learn -- already a TopoCore dependency
GradientBoostClassifier()   # scikit-learn -- already a TopoCore dependency
XGBoostClassifier()         # requires the separate `xgboost` package
LightGBMClassifier()        # requires the separate `lightgbm` package
```

**`XGBoostClassifier`/`LightGBMClassifier` raise immediately at
construction if their own package isn't installed** — confirmed
directly:

```python
>>> XGBoostClassifier()
ClassificationError: XGBoost is not installed. Install it with
`pip install xgboost` (or `uv add xgboost`) to use XGBoost

>>> LightGBMClassifier()
ClassificationError: LightGBM is not installed. Install it with
`pip install lightgbm` ...
```

`RandomForestClassifier`/`GradientBoostClassifier` never raise this
way — both build on scikit-learn, already a core dependency.

## `ClassificationManager`

```python
ClassificationManager(method="auto", model_params=None, rule_params=None)
```

Selects between `RuleBasedClassifier` and the 4 ML backends by name
(`"auto"`, `"rules"`, `"random_forest"`, `"gradient_boost"`,
`"xgboost"`, `"lightgbm"`), matching the manager pattern used
throughout this section — but note this is the one manager whose
underlying classifiers have real, divergent *installation*
requirements (unlike, say, `NormalManager`'s 2 estimators, which both
run on NumPy alone).

## Integration with `Workflow`

`Workflow.classify_points(method="auto", **manager_kwargs)` wraps
`ClassificationManager.classify()` directly — this is the multi-class
counterpart to `Workflow.classify_ground()`'s binary ground/non-ground
split (a genuinely different algorithm, not a convenience alias for
it). See [`workflow-integration.md`](./workflow-integration.md).
