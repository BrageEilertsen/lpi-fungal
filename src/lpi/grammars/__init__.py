"""Biosynthetic grammars behind the one engine interface.

Each grammar implements :class:`~lpi.grammars.base.Grammar`; the engine mines through any
of them with shared observability + verdict layers. ``REGISTRY`` maps a name to its singleton.
"""
from __future__ import annotations

from lpi.grammars.base import Grammar
from lpi.grammars.nrps import NRPS
from lpi.grammars.pks import PKS

__all__ = ["Grammar", "PKS", "NRPS", "REGISTRY", "get"]

REGISTRY: dict[str, Grammar] = {PKS.name: PKS, NRPS.name: NRPS}


def get(name: str) -> Grammar:
    """Look up a grammar singleton by name (``"pks"``, ``"nrps"``, ...)."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown grammar {name!r}; have {sorted(REGISTRY)}") from None
