"""Metabolomics observability layer (Track 2): realistic, uncertainty-aware MS observables.

Turns the engine's exact mass/MS-MS masks into the constraints a real untargeted-metabolomics
workflow supplies, and keeps them honest:

* adduct-aware neutral-mass recovery, with an explicit charge state and ionization mode
  (``observed_mz = (neutral_mass + adduct_shift) / |charge|``);
* a ppm-tolerant mass filter that is a *genuine* approximate verifier
  ``Z*_eps(G,O) = { z : mass(Exec(z)) within ppm of m under some adduct, MSMS(z) >= tau }`` --
  candidates are enumerated WITHOUT an exact-formula prefilter and then filtered, so tolerance is
  real, not cosmetic; if enumeration is capped the verdict is flagged ``censored`` (a lower bound);
* a within-grammar molecular-formula ambiguity report (the formulas produced by *legal candidate
  programs* in the current grammar / adduct / tolerance window -- not "all possible formulas");
* a noise-robust modified-cosine MS/MS scorer with a Da tolerance and threshold tau;
* :func:`minimum_sufficient_observables` -- per cluster, the state at each observable rung and the
  observable that would collapse the set, turning the three-state verdict into an experiment planner.

Everything here reads only executed candidate *structures* (SMILES), so the same machinery serves
every grammar (PKS, NRPS, ...); only the program generator is family-specific.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from lpi.engine import NEAR_UNIQUE_MAX, State, frag_fingerprint
from lpi.grammars import PKS, Grammar

RDLogger.DisableLog("rdApp.*")

_ELECTRON = 0.0005485799


@dataclass(frozen=True)
class Adduct:
    """An MS adduct. ``shift`` is added to the neutral monoisotopic mass to give the ion's
    ``m*|z|``; ``charge`` is ``|z|``; ``mode`` is the ionization polarity."""

    name: str
    shift: float
    charge: int = 1
    mode: str = "positive"  # positive / negative / unknown


# Monoisotopic adduct shifts (electron mass included, so a 2 ppm sweep is meaningful). All v0
# adducts are singly charged, but the charge/mode hooks make multiply-charged species a data change,
# not a code change.
ADDUCTS: dict[str, Adduct] = {
    "[M+H]+":      Adduct("[M+H]+",      +1.0072764, 1, "positive"),
    "[M+Na]+":     Adduct("[M+Na]+",     +22.9892207, 1, "positive"),
    "[M+K]+":      Adduct("[M+K]+",      +38.9631583, 1, "positive"),
    "[M+NH4]+":    Adduct("[M+NH4]+",    +18.0338255, 1, "positive"),
    "[M+H-H2O]+":  Adduct("[M+H-H2O]+",  -17.0032881, 1, "positive"),  # in-source water loss
    "[M-H]-":      Adduct("[M-H]-",      -1.0072764, 1, "negative"),
    "[M+Cl]-":     Adduct("[M+Cl]-",     +34.9694013, 1, "negative"),
    "[M+HCOO]-":   Adduct("[M+HCOO]-",   +44.9982028, 1, "negative"),  # formate
}


def ion_mz(neutral_mass: float, adduct: Adduct) -> float:
    """Predicted observed m/z of ``neutral_mass`` under ``adduct``."""
    return (neutral_mass + adduct.shift) / abs(adduct.charge)


def neutral_from_mz(mz: float, adduct: Adduct) -> float:
    """Neutral monoisotopic mass implied by an observed ``mz`` under ``adduct``."""
    return mz * abs(adduct.charge) - adduct.shift


def ppm_error(observed: float, candidate: float) -> float:
    """Mass error of a candidate neutral mass against an observed neutral mass, in ppm."""
    return abs(candidate - observed) / observed * 1e6 if observed else float("inf")


# ---- small SMILES caches (candidate enumeration repeats structures) ----------------
_mass_cache: dict[str, float] = {}
_formula_cache: dict[str, str] = {}


def exact_mass(smiles: str) -> float:
    m = _mass_cache.get(smiles)
    if m is None:
        mol = Chem.MolFromSmiles(smiles)
        m = Descriptors.ExactMolWt(mol) if mol is not None else float("nan")
        _mass_cache[smiles] = m
    return m


def molecular_formula(smiles: str) -> str:
    f = _formula_cache.get(smiles)
    if f is None:
        mol = Chem.MolFromSmiles(smiles)
        f = rdMolDescriptors.CalcMolFormula(mol) if mol is not None else "?"
        _formula_cache[smiles] = f
    return f


# ---- MS/MS: noise-robust modified cosine over fragment masses ----------------------
def msms_score(predicted: frozenset | set, observed, tol_da: float = 0.02) -> float:
    """Binary modified-cosine similarity between predicted and observed fragment-mass sets.

    ``matched / sqrt(|predicted| * |observed|)`` with greedy matching within ``tol_da``. Extra
    (noise) observed peaks lower the score gently rather than breaking the match, and the true
    structure scores highest because its single-bond-cleavage fragments best overlap the spectrum.
    """
    P = sorted({round(float(x), 4) for x in predicted})
    O = sorted(float(x) for x in observed)
    if not P or not O:
        return 0.0
    matched = 0
    j = 0
    used = [False] * len(O)
    for p in P:
        # greedy: first unused observed peak within tolerance
        for k in range(len(O)):
            if not used[k] and abs(O[k] - p) <= tol_da:
                used[k] = True
                matched += 1
                break
    return matched / (len(P) * len(O)) ** 0.5


# ---- the tolerant observable bundle and result -------------------------------------
@dataclass
class MSObservables:
    """Realistic MS observables. Provide ``observed_mz`` (preferred) or a direct ``neutral_mass``."""

    observed_mz: float | None = None
    adducts: tuple[str, ...] = ("[M+H]+", "[M-H]-")
    ppm: float = 5.0
    neutral_mass: float | None = None     # bypass the adduct model if the neutral mass is known
    msms_peaks: tuple[float, ...] | None = None
    msms_tol_da: float = 0.02
    msms_tau: float = 0.5

    def has_mass(self) -> bool:
        return self.observed_mz is not None or self.neutral_mass is not None

    def neutral_targets(self) -> list[float]:
        """Candidate neutral masses consistent with the observation (one per allowed adduct)."""
        if self.neutral_mass is not None:
            return [self.neutral_mass]
        if self.observed_mz is None:
            return []
        return [neutral_from_mz(self.observed_mz, ADDUCTS[a]) for a in self.adducts]


@dataclass
class MSResult:
    candidates: list
    state: State
    reason: str                       # grammar-relative explanation
    ladder: dict                      # {"genome", "+mass", "+msms"} candidate counts
    formulas: list[str] = field(default_factory=list)  # within-grammar surviving formulas
    next_observable: str | None = None
    censored: bool = False            # enumeration capped -> counts are lower bounds


def _next_observable(ms: MSObservables) -> str:
    if not ms.has_mass():
        return "accurate mass (molecular formula)"
    if ms.msms_peaks is None:
        return "MS/MS fragmentation"
    return "isotope labelling or gene knockout (beyond the implemented observables)"


def infer_ms(alphabet, ms: MSObservables, grammar: Grammar = PKS,
             min_len: int = 1, max_len: int = 10, ladder: bool = True) -> MSResult:
    """Tolerant, adduct/MS-MS-aware inference -> a sound Z*_eps + a grammar-relative three-state verdict.

    The mass rung is a genuine ppm-tolerant set built via the mass-window formula prefilter (enumerate
    the formulas within the window, run the grammar's sound exact enumeration once per formula, union),
    so tolerance is real, not cosmetic, and no candidate is dropped by an exact-equality shortcut. The
    genome-only rung is the full program space and may be combinatorial: if its enumeration is capped,
    its *count* is reported as a lower bound (``censored``), but the mass-conditioned verdict is sound.
    """
    # --- genome rung (the full program space; the mass-withheld candidate set) ---
    n_genome: int | None = None
    censored = False
    cands0: list = []
    if ladder or not ms.has_mass():
        gen0 = grammar.enumerate(alphabet, min_len, max_len)
        cands0 = gen0.candidates
        censored = bool(getattr(gen0, "capped", False))
        n_genome = len(cands0)

    # --- mass rung: sound mass-window formula prefilter (union over adducts x in-window formulas) ---
    if ms.has_mass():
        keys: dict[tuple, None] = {}
        for t in ms.neutral_targets():
            for key in grammar.mass_prefilter_keys(t, ms.ppm):
                keys.setdefault(key, None)
        by_smi: dict[str, object] = {}
        for key in keys:
            for c in grammar.enumerate(alphabet, min_len, max_len, target_cho=key).candidates:
                cur = by_smi.get(c.smiles)
                if cur is None or c.score > cur.score:
                    by_smi[c.smiles] = c
        cands_mass = sorted(by_smi.values(), key=lambda c: -c.score)
    else:
        cands_mass = list(cands0)
    formulas = sorted({molecular_formula(c.smiles) for c in cands_mass})

    # --- MS/MS rung (tolerance-scored modified cosine) ---
    # MS/MS can only RULE OUT a candidate that predicts fragments inconsistent with the spectrum.
    # A candidate the crude single-bond fragmenter cannot fragment (no acyclic single bonds, e.g. a
    # rigid diketopiperazine) predicts an empty spectrum; that is missing evidence, not a mismatch,
    # so it is left unconstrained rather than spuriously rejected.
    msms_applied = ms.msms_peaks is not None
    if msms_applied:
        cands_msms = []
        for c in cands_mass:
            pred = frag_fingerprint(c.smiles)
            if not pred or msms_score(pred, ms.msms_peaks, ms.msms_tol_da) >= ms.msms_tau:
                cands_msms.append(c)
    else:
        cands_msms = cands_mass
    final = cands_msms

    ladder_d = {"genome": n_genome, "+mass": len(cands_mass), "+msms": len(final)}
    state, reason, nxt = _verdict(ms, grammar, n_genome, len(cands_mass), len(final),
                                  msms_applied, censored)
    return MSResult(final, state, reason, ladder_d, formulas, nxt, censored)


def _verdict(ms, grammar, n_genome, n_mass, n_final, msms_applied, censored):
    g = grammar.name
    if not ms.has_mass():
        # verdict on the genome-only set (its count may be a censored lower bound)
        lb = " (lower bound; genome enumeration censored at the program cap)" if censored else ""
        if n_genome == 0:
            return (State.OUT_OF_GRAMMAR,
                    f"no producible structure under the {g} grammar for the given step band", None)
        if n_genome <= NEAR_UNIQUE_MAX:
            return (State.VERIFIED,
                    f"{n_genome} candidate(s) from the {g} grammar alone", None)
        return (State.UNDER_OBSERVED,
                f"{n_genome} candidates under the {g} grammar from the genome alone{lb}",
                _next_observable(ms))
    # mass supplied -> the mass rung is sound (per-formula exact enumeration), so is the verdict
    if n_mass == 0:
        adducts = ", ".join(ms.adducts) if ms.neutral_mass is None else "the given neutral mass"
        return (State.OUT_OF_GRAMMAR,
                f"no legal {g} program matches the mass within {ms.ppm:g} ppm under "
                f"{{{adducts}}}; check the adduct/charge assignment or expand the {g} grammar", None)
    if msms_applied and n_final == 0:
        return (State.UNDER_OBSERVED,
                f"{n_mass} mass-consistent {g} candidate(s), but none clear MS/MS tau={ms.msms_tau:g}; "
                f"the true structure may be out-of-grammar, or the spectrum/threshold needs revisiting",
                "higher-quality MS/MS or a tighter grammar")
    if n_final <= NEAR_UNIQUE_MAX:
        return (State.VERIFIED,
                f"{n_final} candidate(s), unique up to the {g} grammar and the supplied observables",
                None)
    return (State.UNDER_OBSERVED,
            f"{n_final} candidates remain under the {g} grammar and current observables",
            _next_observable(ms))


# ---- the experiment planner: minimum sufficient observables ------------------------
@dataclass
class ObservablePlan:
    rungs: dict           # rung name -> (state, count) e.g. {"genome": (State, n), ...}
    verified_at: str | None   # first rung reaching VERIFIED, or None
    recommended: str | None   # next observable to collect if not yet verified
    censored: bool


def minimum_sufficient_observables(alphabet, ms: MSObservables, grammar: Grammar = PKS,
                                   min_len: int = 1, max_len: int = 10) -> ObservablePlan:
    """Run the observable ladder and report which observable first collapses the set to VERIFIED --
    the quantified experiment planner behind the ``next_observable`` seed."""
    genome = infer_ms(alphabet, MSObservables(), grammar, min_len, max_len)
    mass_only = infer_ms(alphabet, MSObservables(observed_mz=ms.observed_mz, adducts=ms.adducts,
                                                 ppm=ms.ppm, neutral_mass=ms.neutral_mass),
                         grammar, min_len, max_len)
    full = infer_ms(alphabet, ms, grammar, min_len, max_len)
    rungs = {
        "genome": (genome.state, genome.ladder["genome"]),
        "+mass": (mass_only.state, mass_only.ladder["+mass"]),
        "+mass+msms": (full.state, full.ladder["+msms"]),
    }
    verified_at = next((name for name, (st, _) in rungs.items() if st is State.VERIFIED), None)
    recommended = None if verified_at else full.next_observable
    return ObservablePlan(rungs, verified_at, recommended, full.censored)
