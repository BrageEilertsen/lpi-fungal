"""Planner item 2: the E[Δ|Z*|] estimator -- grounded in the sound verifier.

|Z*| is the size of the verifier's consistent candidate set (lpi.observe.infer_ms). For a design we know
the intended product y_d = Exec(z_d), so we can compute |Z*| under any subset of the modelled observables
(genome alphabet, accurate mass, MS/MS) and the information gain Δ|Z*| of adding one observable. This is
the within-frame disambiguation quantity the harness scores against; the planner (item 4) ranks
observations by it, weighted by realizability and divided by observation cost (item 3).

v0 honesty (PLANNER_SKETCH Q4 -- the estimator is bounded by the forward model):
  * mass is sound (adduct + ppm + formula prefilter); MS/MS uses the crude single-bond fragmenter;
    isotope / expression / knockout are v1+ (no forward model yet).
  * v0 realizability is deterministic (Q1), so the confirm/refute outcome of observing a KNOWN target is
    near-deterministic: a producible, in-grammar design confirms (its product is mass-consistent with >=1
    legal program); an out-of-grammar design refutes (e.g. a methylmalonyl extender swap -> the malonyl-only
    grammar yields zero mass-consistent programs). The probabilistic transition API (item 1) is exercised
    structurally now and carries genuinely uncertain probabilities only in v1 (observation-updated
    probabilistic realizability, which needs wet-lab build outcomes).
"""
from __future__ import annotations

from dataclasses import dataclass

from rdkit import RDLogger

from lpi.chem import mol as M
from lpi.engine import frag_fingerprint
from lpi.executor import core
from lpi.grammars import PKS, Grammar
from lpi.observe import MSObservables, exact_mass, infer_ms
from lpi.planner.state import DesignState, ForwardModel, Observation
from lpi.realizability import Realizability, _target_domains

RDLogger.DisableLog("rdApp.*")

MASS, MSMS = "mass", "msms"


@dataclass(frozen=True)
class DesignProduct:
    """The design's intended product y_d = Exec(z_d): structure, monoisotopic mass, MS/MS fingerprint."""

    smiles: str
    mass: float
    msms: frozenset


def design_product(realiz: Realizability) -> DesignProduct | None:
    """Execute the design's target program -> y_d, or None if it does not produce a structure (a
    coverage-limited design the executor cannot render)."""
    try:
        smi = M.canonical_smiles(core.exec(realiz.target))
    except Exception:  # noqa: BLE001 - an edited program whose cyclization cannot fire
        return None
    return DesignProduct(smi, exact_mass(smi), frag_fingerprint(smi))


def _alphabet_and_len(realiz: Realizability, grammar: Grammar):
    """The GENOME alphabet for the inference question: the nearest cluster's declared domain set (what the
    genome offers) plus any domains the design's structural edits add -- NOT the program's required domains
    (using those would leak the answer the observation is meant to infer, e.g. assuming an unfired ER is
    absent). Matches the harness's alphabet derivation."""
    prog = realiz.target
    domains = {"KS", "AT", "ACP"} | set(realiz.nearest.domains) | set(_target_domains(prog))
    return grammar.alphabet_from_domains(domains, starters=(prog.starter,)), len(prog.cycles)


def z_star(realiz: Realizability, prod: DesignProduct, use_mass: bool, use_msms: bool,
           grammar: Grammar = PKS) -> int:
    """|Z*|: the number of grammar-legal programs consistent with the genome alphabet and the chosen
    observables of the design's own product. Reuses the sound verifier. Monotone non-increasing as
    observables are added. (The genome-only rung is the full program space and may be censored for large
    alphabets; for the small reachable designs it is exact.)"""
    alpha, n = _alphabet_and_len(realiz, grammar)
    ms = MSObservables(
        neutral_mass=prod.mass if use_mass else None,
        msms_peaks=tuple(sorted(prod.msms)) if (use_msms and prod.msms) else None,
    )
    return len(infer_ms(alpha, ms, grammar, max(1, n - 1), n + 1, ladder=False).candidates)


def delta_z(realiz: Realizability, prod: DesignProduct, current: frozenset[str], obs_kind: str,
            grammar: Grammar = PKS) -> int:
    """Information gain Δ|Z*| of adding observation ``obs_kind`` to the already-incorporated observable
    set ``current`` (a subset of {"mass","msms"}) -- |Z*|(current) - |Z*|(current + obs_kind). The
    observation's value is the design's own product (the known target), so this is deterministic in v0."""
    nxt = current | {obs_kind}
    before = z_star(realiz, prod, MASS in current, MSMS in current, grammar)
    after = z_star(realiz, prod, MASS in nxt, MSMS in nxt, grammar)
    return before - after


def forward_model_confirm(realiz: Realizability, prod: DesignProduct | None,
                          grammar: Grammar = PKS) -> float:
    """v0 P(confirm): a producible, in-grammar design confirms (its product is mass-consistent with >=1
    legal program); a non-producible or out-of-grammar design refutes. Design-level in v0; per-edit and
    probabilistic-build-success forward models are v1."""
    if prod is None:
        return 0.0
    return 1.0 if z_star(realiz, prod, True, False, grammar) >= 1 else 0.0


def make_forward_model(grammar: Grammar = PKS) -> ForwardModel:
    """Adapt the design-level confirm probability to item 1's ForwardModel signature
    ``(DesignState, Observation, edit_index) -> p_confirm`` (product cached per design target)."""
    cache: dict[str, DesignProduct | None] = {}

    def fm(state: DesignState, obs: Observation, edit_index: int) -> float:
        key = repr(state.realiz.target)
        if key not in cache:
            cache[key] = design_product(state.realiz)
        return forward_model_confirm(state.realiz, cache[key], grammar)

    return fm
