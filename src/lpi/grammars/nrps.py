"""Non-ribosomal peptide grammar -- the second family behind ``engine.infer()``.

The peptide analogue of :mod:`lpi.grammars.pks`: it maps a cluster's NRPS domains to a
program alphabet (which residues, which releases) and enumerates the producible peptides
with the sound :mod:`lpi.executor.nrps` executor, exact-formula-prefiltered by an
MS-measured molecular formula. Its existence is the architecture-generalization claim of
the engine: the mass/MS-MS/three-state-verdict machinery in :mod:`lpi.engine` mines through
this grammar with no change, because it reads only candidate SMILES.

Domain->alphabet (v0): A-domain *substrate specificity* is not parsed, so the residue set
defaults to the implemented monomer table; a thioesterase (TE) or terminal reductase (TD)
unlocks head-to-tail macrolactamization, otherwise only the linear acid is produced. Parsing
A-domain specificity codes (Stachelhaus / SANDPUMA) to constrain the residue set per module
is the obvious next step and is flagged in QUESTIONS.md.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from lpi.chem import mol as M
from lpi.chem.formula import enumerate_chno
from lpi.executor import nrps as _nrps
from lpi.executor.nrps import Release
from lpi.search.generate import Candidate, GenResult

__all__ = ["NRPSAlphabet", "NRPSGrammar", "NRPS"]

_DEFAULT_RESIDUES = tuple(sorted(_nrps.RESIDUES))  # deterministic enumeration order


@dataclass(frozen=True)
class NRPSAlphabet:
    """The catalytic moves an NRPS cluster's domains permit (the genome-mining input)."""

    residues: tuple[str, ...] = _DEFAULT_RESIDUES
    releases: tuple[Release, ...] = (Release.HYDROLYSIS, Release.MACROLACTAM)


def _score(residues: tuple[str, ...]) -> float:
    """Parsimony prior: prefer shorter peptides (fewer condensation modules)."""
    return -float(len(residues))


def _ranked(best: dict[str, Candidate]) -> list[Candidate]:
    return sorted(best.values(), key=lambda c: -c.score)


class NRPSGrammar:
    """Non-ribosomal peptide synthetase grammar (v0)."""

    name = "nrps"

    def alphabet_from_domains(self, domains: Any, *_args: Any, **_kw: Any) -> NRPSAlphabet:
        d = {str(x).upper() for x in domains}
        releases = [Release.HYDROLYSIS]
        if "TE" in d or "TD" in d:  # thioesterase / reductase release -> macrocyclization
            releases.append(Release.MACROLACTAM)
        return NRPSAlphabet(releases=tuple(releases))

    def enumerate(self, alphabet: NRPSAlphabet, min_len: int, max_len: int,
                  target_cho: tuple[int, ...] | None = None,
                  program_cap: int = 20000) -> GenResult:
        """Enumerate alphabet-allowed peptides, execute, and return distinct ranked structures.

        ``target_cho`` is the ``(C, H, N, O)`` of an MS-measured formula (note the N, unlike the
        PKS triple). When given, the analytic :func:`lpi.executor.nrps.peptide_formula` prefilters
        each sequence BEFORE building a molecule -- the per-module combinatorics never blow up.
        """
        residues = tuple(alphabet.residues)
        target = tuple(target_cho) if target_cho is not None else None
        best: dict[str, Candidate] = {}
        tried = 0
        for n in range(max(1, min_len), max_len + 1):
            for combo in itertools.product(residues, repeat=n):
                for rel in alphabet.releases:
                    if rel is Release.MACROLACTAM and n < 2:
                        continue  # one-residue head-to-tail closure is a strained alpha-lactam
                    if target is not None and _nrps.peptide_formula(combo, rel) != target:
                        continue  # exact formula prefilter (incl. N): skip off-mass sequences
                    tried += 1
                    if tried > program_cap:
                        return GenResult(_ranked(best), tried, capped=True)
                    pep = _nrps.Peptide(combo, rel)
                    try:
                        mol = _nrps.run(pep)
                    except _nrps.NRPSExecError:
                        continue
                    smi = M.canonical_smiles(mol)
                    s = _score(combo)
                    if smi not in best or s > best[smi].score:
                        best[smi] = Candidate(smi, pep, s)
        return GenResult(_ranked(best), tried, capped=False)

    def mass_prefilter_keys(self, neutral_mass: float, ppm: float,
                            max_c: int = 60) -> list[tuple[int, int, int, int]]:
        """Product (C, H, N, O) formulas within ``ppm`` of ``neutral_mass`` -- note the nitrogen,
        which the peptide enumerator's exact prefilter requires."""
        return enumerate_chno(neutral_mass, ppm, with_n=True, max_c=max_c)


#: module singleton.
NRPS = NRPSGrammar()
