"""Verifier-grounded candidate generator for genome mining.

Given a cluster's catalytic *alphabet* (which reduction states, releases, starters, and
C-MeT its domains permit) and a chain-length band, enumerate every program the alphabet
allows, execute it with the sound deterministic executor, and return the distinct producible
core scaffolds ranked by a parsimony prior. This turns the executor into a hypothesis
generator: a short, ranked list of chemically-constructible cores to match against
metabolomics -- useful at today's reconstructibility ceiling because it never needs to pin
the exact per-cycle program, only to enumerate the plausible ones.

Two-walls corollary: for an NR/aromatic alphabet the reduction is fixed (keto), so the
candidate set is tiny and the truth ranks at/near the top; for an HR alphabet it is
combinatorial (up to 4^N reduction patterns) -- the candidate-set size is itself the
quantitative face of the per-cycle-reduction grammar wall.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from rdkit import Chem, RDLogger

from lpi.chem import mol as M
from lpi.chem.program import Cycle, Program, ReductionState, Release
from lpi.executor import core as _core
from lpi.search.beam import _formula_cho, _formula_consistent, linear_acid_formula

RDLogger.DisableLog("rdApp.*")

_STARTER_C = {"acetyl": 2, "propionyl": 3, "butyryl": 4, "hexanoyl": 6}


@dataclass(frozen=True)
class Alphabet:
    """The catalytic moves a cluster's domains permit (the genome-mining input)."""

    reductions: tuple[ReductionState, ...] = (
        ReductionState.KETO, ReductionState.KR, ReductionState.DH, ReductionState.ER)
    releases: tuple[Release, ...] = (Release.HYDROLYSIS,)
    starters: tuple[str, ...] = ("acetyl",)
    allow_cmet: bool = False


@dataclass
class Candidate:
    smiles: str
    program: Program
    score: float


@dataclass
class GenResult:
    candidates: list[Candidate] = field(default_factory=list)
    programs_tried: int = 0
    capped: bool = False


def _score(prog: Program) -> float:
    """Parsimony prior: prefer fewer/lighter reductions, fewer methyls, shorter chains."""
    red = sum(c.reduction.rank for c in prog.cycles)
    me = sum(1 for c in prog.cycles if c.c_methyl)
    return -(red + me + 0.1 * len(prog.cycles))


def generate(alphabet: Alphabet, min_cycles: int, max_cycles: int,
             max_carbons: int = 40, program_cap: int = 20000,
             target_cho: tuple[int, int, int] | None = None) -> GenResult:
    """Enumerate alphabet-allowed programs, execute, and return distinct ranked cores.

    If ``target_cho`` (the C,H,O of an MS-measured molecular formula) is given, programs whose
    analytic released-acid formula cannot reach it are skipped BEFORE execution -- the same
    prefilter the verifier uses (beam.linear_acid_formula / _formula_consistent) -- so the
    per-cycle-reduction combinatorics never blow up: only on-mass programs are executed and
    counted toward ``program_cap``. This is the genome-mining mode (cluster alphabet + MS mass).
    """
    opts = [Cycle(reduction=r, c_methyl=me)
            for r in alphabet.reductions
            for me in ((False, True) if alphabet.allow_cmet else (False,))]
    best: dict[str, Candidate] = {}
    tried = 0
    for starter in alphabet.starters:
        sc = _STARTER_C.get(starter, 2)
        for n in range(min_cycles, max_cycles + 1):
            for combo in itertools.product(opts, repeat=n):
                if sc + sum(3 if c.c_methyl else 2 for c in combo) > max_carbons:
                    continue
                lin = linear_acid_formula(Program(starter, combo)) if target_cho else None
                for rel in alphabet.releases:
                    if target_cho is not None and not _formula_consistent(lin, target_cho, rel):
                        continue  # formula prefilter: never execute an off-mass program
                    tried += 1
                    if tried > program_cap:
                        return GenResult(_ranked(best), tried, capped=True)
                    prog = Program(starter, combo, rel)
                    try:
                        mol = _core.exec(prog)
                    except Exception:  # noqa: BLE001
                        continue
                    if target_cho is not None and _formula_cho(mol) != target_cho:
                        continue  # exact formula (pins the actual waters lost on cyclization)
                    smi = M.canonical_smiles(mol)
                    s = _score(prog)
                    if smi not in best or s > best[smi].score:
                        best[smi] = Candidate(smi, prog, s)
    return GenResult(_ranked(best), tried, capped=False)


def _ranked(best: dict[str, Candidate]) -> list[Candidate]:
    return sorted(best.values(), key=lambda c: -c.score)


def _flat(smi: str) -> str:
    """Stereo-free canonical SMILES -- the executor is achiral by design, so candidate/target
    comparison is at the constitutional (skeleton + connectivity) level."""
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else smi


def rank_of(result: GenResult, target_smiles: str) -> int | None:
    """1-based rank of the target among ranked candidates (stereo-free), or None."""
    tgt = _flat(target_smiles)
    for i, c in enumerate(result.candidates, start=1):
        if _flat(c.smiles) == tgt:
            return i
    return None
