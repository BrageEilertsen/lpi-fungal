"""Biosynthetic grammars behind the one engine interface.

Each grammar implements :class:`~lpi.grammars.base.Grammar`; the engine mines through any
of them with shared observability + verdict layers. ``REGISTRY`` maps a name to its singleton.
"""
from __future__ import annotations

from lpi.grammars.base import Grammar
from lpi.grammars.hybrid import HYBRID
from lpi.grammars.lanthipeptide import LANTHI
from lpi.grammars.nrps import NRPS
from lpi.grammars.pks import PKS

__all__ = ["Grammar", "PKS", "NRPS", "HYBRID", "LANTHI", "REGISTRY", "get"]

# LANTHI is a SCAFFOLD: it implements the protocol and enumerates the sound topology DOF, but its
# structural renderers raise pending chemist-signed SMARTS (see lpi.grammars.lanthipeptide). It is
# registered so it is discoverable behind infer_cluster; calling its enumerate raises until the
# renderers land. Nothing iterates REGISTRY calling enumerate, so this is safe.
REGISTRY: dict[str, Grammar] = {PKS.name: PKS, NRPS.name: NRPS, HYBRID.name: HYBRID, LANTHI.name: LANTHI}


def get(name: str) -> Grammar:
    """Look up a grammar singleton by name (``"pks"``, ``"nrps"``, ...)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown grammar {name!r}; have {sorted(REGISTRY)}") from None
