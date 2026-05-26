"""The grammar interface: one seam, many biosynthetic families.

A :class:`Grammar` is the family-specific half of the engine -- it knows how a
cluster's catalytic domains map to a program *alphabet*, and how to enumerate the
structures that alphabet can produce (as ranked SMILES candidates), optionally
prefiltered by an MS-measured molecular formula.

Everything the engine adds on top -- the accurate-mass filter, the MS/MS fragment
matcher, the observable ladder, and the three-state reconstructibility verdict
(``verified`` / ``under-observed`` / ``out-of-grammar``) -- operates only on the
returned candidates' ``.smiles``. So those layers are written once and shared across
every grammar; adding a family means implementing this protocol, not touching the
engine. The polyketide reference implementation is :mod:`lpi.grammars.pks`; the
peptide one is :mod:`lpi.grammars.nrps`.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# The canonical candidate/result containers live with the original PKS generator; we
# reuse them verbatim so every grammar speaks the same currency to the engine. Only
# ``.smiles`` and ``.score`` are read by the engine; ``.program`` is an opaque,
# grammar-specific provenance object (a PKS Program, an NRPS Peptide, ...).
from lpi.search.generate import Candidate, GenResult  # re-exported

__all__ = ["Grammar", "Candidate", "GenResult"]


@runtime_checkable
class Grammar(Protocol):
    """A biosynthetic grammar the engine can mine through.

    ``min_len``/``max_len`` bound the number of elongation/condensation steps (PKS
    cycles, NRPS modules, ...). ``target_cho`` is the ``(C, H, O)`` of an MS-measured
    molecular formula; when given, a grammar should prefilter its enumeration so that
    only on-formula programs are executed (the per-step combinatorics never blow up).
    """

    #: short identifier, e.g. ``"pks"`` / ``"nrps"`` (used in verdicts and demos).
    name: str

    def alphabet_from_domains(self, domains: Any, **kwargs: Any) -> Any:
        """Map a cluster's catalytic domain set (the ``G`` in ``y=Exec_Theta(z;G)``) to
        the program alphabet this grammar enumerates over."""
        ...

    def enumerate(self, alphabet: Any, min_len: int, max_len: int,
                  target_cho: tuple[int, int, int] | None = None) -> GenResult:
        """Enumerate alphabet-allowed programs, execute each with the family's sound
        executor, and return distinct producible structures as ranked candidates."""
        ...

    def mass_prefilter_keys(self, neutral_mass: float, ppm: float) -> list[tuple[int, ...]]:
        """The product molecular-formula keys (in this grammar's element convention) whose
        monoisotopic mass falls within ``ppm`` of ``neutral_mass``. The tolerant mass observable
        runs ``enumerate(target_cho=key)`` once per returned key and unions the results, giving a
        sound ppm-tolerant candidate set without enumerating the full program space."""
        ...
