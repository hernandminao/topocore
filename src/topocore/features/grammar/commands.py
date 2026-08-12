"""
topocore.features.grammar.commands
======================================

Author
------
Hernán Mina

License
-------
MIT
"""

from __future__ import annotations

from enum import StrEnum


class FieldCodeCommand(StrEnum):
    """
    A figure-assembly instruction attached to a field code, under
    the optional grammar syntax (``BASE.FIGURE.COMMAND``).
    """

    #: Opens a new figure. Must appear exactly once before any
    #: CONTINUE/END/CLOSE for the same (base_code, figure_id).
    START = "start"

    #: Adds this point to an already-open figure. This is the
    #: *implicit* command: ``BASE.FIGURE`` with no third segment.
    CONTINUE = "continue"

    #: Closes the figure without forcing it geometrically closed
    #: (open polyline/polygon, subject to `definition.closed` /
    #: `closure_tolerance` as usual).
    END = "end"

    #: Closes the figure AND forces it geometrically closed,
    #: overriding `closure_tolerance` -- an explicit instruction from
    #: the survey takes priority over geometric inference.
    CLOSE = "close"


__all__ = ["FieldCodeCommand"]
