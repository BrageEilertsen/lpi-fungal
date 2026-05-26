"""Polyketide grammar -- the reference implementation of :class:`~lpi.grammars.base.Grammar`.

A thin adapter over the original, validated PKS generator (:func:`lpi.search.generate.generate`)
and the domain->alphabet bridge (formerly ``engine.alphabet_from_domains``). No PKS chemistry
lives here; it only presents the existing machinery through the family-agnostic interface so the
engine can treat polyketides as one grammar among several.
"""
from __future__ import annotations

from typing import Any

from lpi.chem.program import ReductionState, Release
from lpi.search.generate import Alphabet, GenResult, generate

__all__ = ["PKSGrammar", "PKS", "alphabet_from_domains"]


def alphabet_from_domains(domains: Any, starters: tuple[str, ...] = ("acetyl",)) -> Alphabet:
    """Map a cluster's catalytic domain set (the ``G`` in ``y=Exec_Theta(z;G)``) to the program
    grammar it permits.

    The reductive cascade is gated by domain presence (ER presupposes DH presupposes KR); a
    C-methyltransferase (cMT) enables alpha-methylation; release modes are mapped from PT/TE
    presence. The release mapping is intentionally permissive (PT->aromatic, TE->lactone,
    otherwise all): an over-broad release set costs candidates, not correctness, because the
    mass/MS-MS observables do the real pruning downstream.
    """
    d = {str(x).upper() for x in domains}

    def has(*names):
        return any(n in d for n in names)

    if has("ER"):
        reductions = (ReductionState.KETO, ReductionState.KR, ReductionState.DH, ReductionState.ER)
    elif has("DH"):
        reductions = (ReductionState.KETO, ReductionState.KR, ReductionState.DH)
    elif has("KR"):
        reductions = (ReductionState.KETO, ReductionState.KR)
    else:
        reductions = (ReductionState.KETO,)
    allow_cmet = has("CMT", "C-MET")  # backbone C-methyltransferase (not O-/N-MeT tailoring)
    releases = [Release.HYDROLYSIS]
    if has("TE"):
        releases.append(Release.LACTONIZATION)
    if has("PT"):
        releases += [Release.ALDOL_AROMATIC, Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE]
    if len(releases) == 1:  # no release-informative domain -> stay permissive
        releases = [Release.HYDROLYSIS, Release.LACTONIZATION, Release.ALDOL_AROMATIC,
                    Release.DIHYDROISOCOUMARIN, Release.PT_NAPHTHALENE]
    return Alphabet(reductions, tuple(releases), starters, allow_cmet)


class PKSGrammar:
    """Iterative fungal polyketide synthase grammar (the paper's case study)."""

    name = "pks"

    def alphabet_from_domains(self, domains: Any,
                              starters: tuple[str, ...] = ("acetyl",)) -> Alphabet:
        return alphabet_from_domains(domains, starters)

    def enumerate(self, alphabet: Alphabet, min_len: int, max_len: int,
                  target_cho: tuple[int, int, int] | None = None) -> GenResult:
        return generate(alphabet, min_len, max_len, target_cho=target_cho)


#: module singleton; the engine's default grammar.
PKS = PKSGrammar()
