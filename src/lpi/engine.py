"""Verifier-Grounded Biosynthetic Program Engine (v0 -- polyketide reference).

One observable-constrained inference interface that unifies the components demonstrated in this
work. Given a cluster's catalytic ALPHABET (the legal program grammar) and a set of OBSERVABLES,
it enumerates producible cores with the sound executor + verifier, constrains them by each
observable, ranks the survivors, and returns candidates with a per-cluster RECONSTRUCTIBILITY
verdict. The five architectural layers it ties together:

  L1  operators (Theta)  : lpi.executor (PKS extension/reduction/cyclization) + tailoring edits
  L2  grammar            : lpi.search.generate.Alphabet (which operators a cluster's domains allow)
  L3  observability       : this module -- domain alphabet (always), MS accurate mass (-> formula),
                           MS/MS (-> fragment fingerprint)
  L4  inference engine    : lpi.search.generate / beam (sound enumerate + formula prefilter)
  L5  reconstructibility   : classify_regime -- where the residual uncertainty lives

The signature output is the observable LADDER: how the verified candidate set collapses as
orthogonal observables are added (alphabet -> +mass -> +MS/MS), per cluster.

Scope: the polyketide grammar implemented here. The multi-system engine (NRPS, terpene, RiPP,
non-local pathways) and a calibrated program prior are future work; this is the PKS reference v0.
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger

from lpi.chem.program import ReductionState, Release
from lpi.search.generate import Alphabet, Candidate, generate, rank_of

RDLogger.DisableLog("rdApp.*")
_PT = Chem.GetPeriodicTable()


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
    alphabet: Alphabet                          # from the cluster's domains
    min_cycles: int = 1
    max_cycles: int = 10
    target_cho: tuple[int, int, int] | None = None   # from MS accurate mass
    msms: frozenset | None = None                    # observed MS/MS fragment fingerprint


@dataclass
class EngineResult:
    candidates: list[Candidate]
    regime: str
    verdict: str
    ladder: dict          # candidate count after each observable rung


def _msms_filter(cands: list[Candidate], fp: frozenset | None) -> list[Candidate]:
    if fp is None:
        return cands
    keep = [c for c in cands if frag_fingerprint(c.smiles) == fp]
    return keep or cands  # crude fp may miss; never drop everything


def _classify(n_final: int, obs: Observables) -> tuple[str, str]:
    if n_final == 0:
        return ("coverage-limited / out-of-scope",
                "no producible core under this grammar (operator coverage or non-PKS)")
    used = "alphabet" + ("+mass" if obs.target_cho is not None else "") \
        + ("+MS/MS" if obs.msms is not None else "")
    if n_final == 1:
        return "reconstructed", f"unique core given {used}"
    if n_final <= 5:
        return "near-unique", f"{n_final} candidates given {used}; MS/MS or a prior disambiguates"
    return "underdetermined", f"{n_final} candidates given {used}; add observables or sequence signal"


def infer(obs: Observables, ladder: bool = True) -> EngineResult:
    """Enumerate producible cores constrained by the observables; rank; classify the regime."""
    res = generate(obs.alphabet, obs.min_cycles, obs.max_cycles, target_cho=obs.target_cho)
    cands = _msms_filter(res.candidates, obs.msms)
    rungs: dict = {}
    if ladder:
        rungs["alphabet"] = len(generate(obs.alphabet, obs.min_cycles, obs.max_cycles).candidates)
        rungs["alphabet+mass"] = len(res.candidates)
        rungs["alphabet+mass+msms"] = len(cands)
    regime, verdict = _classify(len(cands), obs)
    return EngineResult(cands, regime, verdict, rungs)


def contains(result: EngineResult, target_smiles: str) -> int | None:
    """Rank of a known target among the engine's candidates (for retrospective evaluation)."""
    return rank_of(result, target_smiles)


# ---- L2 bridge: derive the legal program grammar from a cluster's domain set ------
def alphabet_from_domains(domains, starters: tuple[str, ...] = ("acetyl",)) -> Alphabet:
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


def infer_cluster(domains, min_cycles: int, max_cycles: int,
                  target_cho: tuple[int, int, int] | None = None,
                  msms: frozenset | None = None, starters: tuple[str, ...] = ("acetyl",),
                  ladder: bool = True) -> EngineResult:
    """End-to-end interface: a cluster's domain set + observables -> candidates + verdict.

    Derives the program grammar from ``domains`` (L2), then runs observable-constrained inference
    (L3/L4) and the reconstructibility verdict (L5).
    """
    obs = Observables(alphabet_from_domains(domains, starters), min_cycles, max_cycles,
                      target_cho, msms)
    return infer(obs, ladder=ladder)
