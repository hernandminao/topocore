"""
topocore.geodesy.vertical.transformer
========================================

`VerticalTransformer`: applies `H = h - N` (ellipsoidal-to-orthometric)
or its inverse, using a real `GeoidGrid`. Deliberately independent of
`topocore.geodesy.CoordinateTransformer` -- see this package's own
`README`-equivalent, `../transform.py`'s own module docstring, and
`../../14-geodesy/vertical-reference-design.md` for why horizontal
and vertical transformation are kept as two separate, composable
components rather than one class doing both.

The one rule this entire module exists to enforce
---------------------------------------------------
A vertical transformation must never return a height that was left
uncorrected because the required geoid grid was unavailable. This is
not a style preference -- it directly addresses a confirmed, real
danger: plain `pyproj.Transformer` was confirmed (during this
package's own design review) to silently return an unchanged height
when its own required vertical grid is missing, rather than raising.
`VerticalTransformer` never does this: every code path that would
otherwise produce an uncorrected height raises `MissingGeoidGridError`
instead (via `GeoidGrid.undulation_at()`, which already raises this
for an out-of-extent or nodata point -- this class adds no separate
"skip correction" branch of its own).

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from dataclasses import dataclass

from topocore.geodesy.vertical.geoid_grid import GeoidGrid
from topocore.geodesy.vertical_datum import VerticalDatum


@dataclass(frozen=True, slots=True)
class VerticalTransformer:
    """
    Parameters
    ----------
    source_datum
        The vertical datum the input heights are already in.
    target_datum
        The vertical datum the output heights should be in.
    geoid
        The `GeoidGrid` providing `N` at any `(longitude, latitude)`
        needed to convert between `source_datum` and `target_datum`.

    `source_datum`/`target_datum` are carried for traceability
    (matching `VerticalDatum`'s own existing role as pure,
    already-established domain data -- see
    `topocore.geodesy.VerticalDatum`) -- this class does not
    currently branch its own math on which specific datums they are,
    since exactly one geoid-based conversion is implemented today:
    ellipsoidal height <-> orthometric height via a single geoid
    model. A future version supporting datum-to-datum shifts beyond
    "ellipsoidal vs. this one geoid's own orthometric surface" would
    extend this class, not silently reinterpret it.
    """

    source_datum: VerticalDatum
    target_datum: VerticalDatum
    geoid: GeoidGrid

    def ellipsoidal_to_orthometric(self, longitude: float, latitude: float, height: float) -> float:
        """
        `H = h - N`. Raises `MissingGeoidGridError` (propagated
        directly from `GeoidGrid.undulation_at()`) if no usable
        undulation value exists at `(longitude, latitude)` --
        never returns `height` uncorrected.
        """
        undulation = self.geoid.undulation_at(longitude, latitude)
        return height - undulation

    def orthometric_to_ellipsoidal(self, longitude: float, latitude: float, height: float) -> float:
        """`h = H + N`. Same failure behavior as the inverse direction."""
        undulation = self.geoid.undulation_at(longitude, latitude)
        return height + undulation


__all__ = ["VerticalTransformer"]
