"""Verifier-Grounded Biosynthetic Program Engine (v0).

One observable-constrained inference interface, now grammar-parametric. Given a cluster's
catalytic ALPHABET (the legal program grammar) and a set of OBSERVABLES, it enumerates
producible structures with a family's sound executor + verifier, constrains them by each
observable, ranks the survivors, and returns candidates with a per-cluster three-state
RECONSTRUCTIBILITY verdict. The architectural layers it ties together:

  L1  operators (Theta)  : each grammar's sound executor (lpi.executor / lpi.grammars.*)
  L2  grammar            : lpi.grammars.Grammar -- which programs a cluster's domains allow,
                           and how to enumerate them (pks, nrps, ...)
  L3  observability       : this module -- domain alphabet (always), MS accurate mass (-> formula),
                           MS/MS (-> fragment fingerprint)
  L4  inference engine    : grammar.enumerate (sound enumerate + formula prefilter)
  L5  reconstructibility   : the three-state verdict -- where the residual uncertainty lives

The three states (the goal frame: every BGC/product pair lands in exactly one):
  VERIFIED        -- Z* non-empty and near-unique: these legal programs produce this structure.
  UNDER-OBSERVED  -- several legal programs remain; ``next_observable`` names what would collapse them.
  OUT-OF-GRAMMAR  -- the operator library cannot produce it (add coverage, or it is non-local biology).

L3/L5 read only candidate SMILES, so they are shared verbatim across grammars; ``infer`` defaults to
the polyketide grammar (the paper's case study) but accepts any :class:`lpi.grammars.Grammar`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from rdkit import Chem, RDLogger

from lpi.grammars import PKS, Grammar
# Re-exported for backward compatibility (callers/tests import it from lpi.engine):
from lpi.grammars.pks import alphabet_from_domains  # noqa: F401
from lpi.search.generate import Alphabet, Candidate, rank_of

RDLogger.DisableLog("rdApp.*")
_PT = Chem.GetPeriodicTable()

#: |Z*| at or below this is "near-unique" -> the VERIFIED state; above it, UNDER-OBSERVED.
NEAR_UNIQUE_MAX = 5


# ---- L3 component: MS/MS fragment fingerprint (crude CID proxy) -------------------
def frag_fingerprint(smiles: str) -> frozenset:
    """Set of fragment masses from breaking each acyclic single bond (original H counts kept).
    A rough in-silico CID proxy; a production engine would use CFM-ID + spectral cosine."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return frozenset()
    amass = {a.GetIdx(): _PT.GetAtomicWeight(a.GetAtomicNum()) + a.GetTotalNumHs() * 1.008
             for a in mol.GetAtoms()}
    out = set()
    for b in mol.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        em = Chem.RWMol(mol)
        em.RemoveBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for frag in Chem.GetMolFrags(em.GetMol()):
            out.add(round(sum(amass[i] for i in frag), 1))
    return frozenset(out)


# ---- L3: the observables we condition on -----------------------------------------
@dataclass
class Observables:
    alphabet: Alphabet                          # from the cluster's domains (grammar-specific)
    min_cycles: int = 1                         # min elongation/condensation steps
    max_cycles: int = 10                        # max elongation/condensation steps
    target_cho: tuple[int, int, int] | None = None   # from MS accurate mass
    msms: frozenset | None = None                    # observed MS/MS fragment fingerprint


# ---- L5: the three reconstructibility states -------------------------------------
class State(str, Enum):
    """The goal-frame verdict every BGC/product pair lands in (see module docstring)."""

    VERIFIED = "verified"
    UNDER_OBSERVED = "under-observed"
    OUT_OF_GRAMMAR = "out-of-grammar"


@dataclass
class EngineResult:
    candidates: list[Candidate]
    regime: str               # fine-grained label (reconstructed / near-unique / ...)
    verdict: str              # human-readable explanation
    ladder: dict              # candidate count after each observable rung
    state: State              # the canonical three-state verdict
    next_observable: str | None  # for UNDER-OBSERVED: what would most collapse Z* next


def _msms_filter(cands: list[Candidate], fp: frozenset | None) -> list[Candidate]:
    if fp is None:
        return cands
    keep = [c for c in cands if frag_fingerprint(c.smiles) == fp]
    return keep or cands  # crude fp may miss; never drop everything


def _classify(n_final: int, obs: Observables) -> tuple[str, str]:
    if n_final == 0:
        return ("coverage-limited / out-of-scope",
                "no producible structure under this grammar (operator coverage or non-family)")
    used = "alphabet" + ("+mass" if obs.target_cho is not None else "") \
        + ("+MS/MS" if obs.msms is not None else "")
    if n_final == 1:
        return "reconstructed", f"unique structure given {used}"
    if n_final <= NEAR_UNIQUE_MAX:
        return "near-unique", f"{n_final} candidates given {used}; MS/MS or a prior disambiguates"
    return "underdetermined", f"{n_final} candidates given {used}; add observables or sequence signal"


def _next_observable(obs: Observables) -> str:
    """Name the next orthogonal observable that would collapse an under-observed set
    (the experiment-planner seed; Track 2 quantifies the expected collapse)."""
    if obs.target_cho is None:
        return "accurate mass (molecular formula)"
    if obs.msms is None:
        return "MS/MS fragmentation"
    return "isotope labelling or gene knockout (beyond the implemented observables)"


def _verdict_state(n_final: int, obs: Observables) -> tuple[State, str | None]:
    if n_final == 0:
        return State.OUT_OF_GRAMMAR, None
    if n_final <= NEAR_UNIQUE_MAX:
        return State.VERIFIED, None
    return State.UNDER_OBSERVED, _next_observable(obs)


def infer(obs: Observables, grammar: Grammar = PKS, ladder: bool = True) -> EngineResult:
    """Enumerate producible structures (via ``grammar``) constrained by the observables; rank;
    return the three-state reconstructibility verdict. Defaults to the polyketide grammar."""
    res = grammar.enumerate(obs.alphabet, obs.min_cycles, obs.max_cycles, target_cho=obs.target_cho)
    cands = _msms_filter(res.candidates, obs.msms)
    rungs: dict = {}
    if ladder:
        rungs["alphabet"] = len(
            grammar.enumerate(obs.alphabet, obs.min_cycles, obs.max_cycles).candidates)
        rungs["alphabet+mass"] = len(res.candidates)
        rungs["alphabet+mass+msms"] = len(cands)
    regime, verdict = _classify(len(cands), obs)
    state, nxt = _verdict_state(len(cands), obs)
    return EngineResult(cands, regime, verdict, rungs, state, nxt)


def contains(result: EngineResult, target_smiles: str) -> int | None:
    """Rank of a known target among the engine's candidates (for retrospective evaluation)."""
    return rank_of(result, target_smiles)


def infer_cluster(domains, min_cycles: int, max_cycles: int,
                  target_cho: tuple[int, int, int] | None = None,
                  msms: frozenset | None = None, starters: tuple[str, ...] = ("acetyl",),
                  grammar: Grammar = PKS, ladder: bool = True) -> EngineResult:
    """End-to-end interface: a cluster's domain set + observables -> candidates + three-state verdict.

    Derives the program alphabet from ``domains`` via the grammar (L2), then runs
    observable-constrained inference (L3/L4) and the reconstructibility verdict (L5).
    """
    alpha = grammar.alphabet_from_domains(domains, starters)
    obs = Observables(alpha, min_cycles, max_cycles, target_cho, msms)
    return infer(obs, grammar=grammar, ladder=ladder)
