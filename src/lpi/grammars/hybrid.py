"""PKS-NRPS hybrid grammar -- typed composition of two grammars behind one engine interface.

The unification step: a hybrid program is a PKS sub-program joined by a typed handoff operator to an
NRPS peptide continuation (:mod:`lpi.executor.hybrid`). This grammar does not re-implement either
family -- it COMPOSES :func:`lpi.grammars.pks.PKS.enumerate` with the NRPS monomer set, and the same
mass / MS-MS / observable-ladder / three-state-verdict layer in :mod:`lpi.engine` / :mod:`lpi.observe`
runs over the hybrid candidates unchanged.

Mass conditioning is sound, not censored: given a target hybrid formula, each residue combo fixes the
nitrogen and the required PK (C,H,O), so the PKS grammar's *exact* enumeration runs once per
(combo, PK-formula) -- the combinatorial PK space is never enumerated wholesale.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Any

from lpi.chem import mol as M
from lpi.chem.formula import enumerate_chno
from lpi.chem.program import Release
from lpi.executor import hybrid as _hyb
from lpi.executor import nrps as _nrps
from lpi.executor.hybrid import HybridRelease
from lpi.grammars.pks import PKS
from lpi.search.generate import Alphabet, Candidate, GenResult

__all__ = ["HybridAlphabet", "HybridGrammar", "HYBRID"]

_DEFAULT_RESIDUES = tuple(sorted(_nrps.RESIDUES))


@dataclass(frozen=True)
class HybridAlphabet:
    """The hybrid program space: a PKS sub-alphabet, the NRPS handoff monomers, and releases."""

    pks: Alphabet = field(default_factory=Alphabet)
    residues: tuple[str, ...] = _DEFAULT_RESIDUES
    releases: tuple[HybridRelease, ...] = (HybridRelease.HYDROLYSIS, HybridRelease.TETRAMIC_ACID)
    max_residues: int = 1   # tenellin-like single-amino-acid handoff by default


def _score(n_cycles: int, residues: tuple[str, ...]) -> float:
    return -float(n_cycles + len(residues))   # parsimony: fewer cycles + fewer residues


def _ranked(best: dict[str, Candidate]) -> list[Candidate]:
    return sorted(best.values(), key=lambda c: -c.score)


class HybridGrammar:
    """Minimal PKS-NRPS hybrid grammar (v0): typed compositionality, not biochemical completeness."""

    name = "hybrid"

    def alphabet_from_domains(self, domains: Any, *_a: Any, **_k: Any) -> HybridAlphabet:
        # PK part is released as the LINEAR acid (the handoff checkpoint); the NRPS monomers are the
        # default set. A tetramic-acid release is DECLARED (typed) though unimplemented, so a cluster
        # whose true product is a tetramic acid yields an explicit operator-gap verdict.
        pks_alpha = replace(PKS.alphabet_from_domains(domains), releases=(Release.HYDROLYSIS,))
        return HybridAlphabet(pks_alpha, _DEFAULT_RESIDUES,
                              (HybridRelease.HYDROLYSIS, HybridRelease.TETRAMIC_ACID))

    def _combos(self, alphabet: HybridAlphabet):
        for n in range(1, alphabet.max_residues + 1):
            yield from itertools.product(alphabet.residues, repeat=n)

    def enumerate(self, alphabet: HybridAlphabet, min_len: int, max_len: int,
                  target_cho: tuple[int, ...] | None = None,
                  program_cap: int = 20000) -> GenResult:
        pks_alpha = replace(alphabet.pks, releases=(Release.HYDROLYSIS,))
        # Only render releases the executor actually implements; a cluster whose declared release is
        # unimplemented (e.g. tetramic-acid Dieckmann) therefore yields NO candidate and an explicit
        # out-of-grammar verdict, rather than silently substituting hydrolysis.
        renderable = tuple(r for r in alphabet.releases if r in _hyb.IMPLEMENTED_RELEASES)
        best: dict[str, Candidate] = {}
        tried = 0

        def add(pk_program, combo) -> None:
            for rel in renderable:
                try:
                    mol = _hyb.run(_hyb.Hybrid(pk_program, combo, rel))
                except _hyb.HybridError:
                    continue
                smi = M.canonical_smiles(mol)
                s = _score(len(pk_program.cycles), combo)
                if smi not in best or s > best[smi].score:
                    best[smi] = Candidate(smi, _hyb.Hybrid(pk_program, combo, rel), s)

        if target_cho is not None:
            tc, th, tn, to = target_cho
            for combo in self._combos(alphabet):
                qc, qh, qn, qo = _nrps.peptide_formula(combo, _nrps.Release.HYDROLYSIS)
                if qn != tn:                      # the residues must supply exactly the target N
                    continue
                pk_key = (tc - qc, th - qh + 2, to - qo + 1)   # derived PK (C,H,O); + handoff water
                if any(v < 0 for v in pk_key):
                    continue
                # SOUND: the PKS grammar's exact enumeration, one call per (combo, PK-formula)
                for pk in PKS.enumerate(pks_alpha, min_len, max_len, target_cho=pk_key).candidates:
                    tried += 1
                    if tried > program_cap:
                        return GenResult(_ranked(best), tried, capped=True)
                    add(pk.program, combo)
        else:
            # genome rung: compose the (possibly combinatorial) PK space with residue combos
            pk_cands = PKS.enumerate(pks_alpha, min_len, max_len).candidates
            for pk in pk_cands:
                for combo in self._combos(alphabet):
                    tried += 1
                    if tried > program_cap:
                        return GenResult(_ranked(best), tried, capped=True)
                    add(pk.program, combo)
        return GenResult(_ranked(best), tried, capped=False)

    def mass_prefilter_keys(self, neutral_mass: float, ppm: float,
                            max_c: int = 60) -> list[tuple[int, int, int, int]]:
        """Hybrid product (C, H, N, O) formulas within ``ppm`` of ``neutral_mass`` (peptides carry N)."""
        return enumerate_chno(neutral_mass, ppm, with_n=True, max_c=max_c)


#: module singleton.
HYBRID = HybridGrammar()
