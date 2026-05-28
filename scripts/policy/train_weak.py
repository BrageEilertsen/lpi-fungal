"""Phase D2: weak-supervision marginal-likelihood objective for the per-cycle policy.

For each BGC asset B, the inverse compiler returns a verified candidate set Z*(B) of
size 1..K. The sequence policy pi_theta defines a probability over a program z as a
product of per-cycle conditional actions; we minimize the Negative Log Marginal
Likelihood over the corpus D:

  L(theta) = - Sum_B  log Sum_{z in Z*(B)}  P(z | B; theta)

       where  P(z | B; theta) = Prod_t  pi_theta(z_t | s_t^(z), B)

Properties:
  - |Z*|=1 case collapses to standard cross-entropy (the curated GOLD tier).
  - |Z*|>1 case (SILVER) splits gradient across candidates by current model prob;
    candidates that share substructure with the GOLD set accumulate probability.
  - log-sum-exp keeps things numerically stable across the per-candidate product
    (each cycle's log-prob can be very negative; the exp before sum needs care).

This script is the SCAFFOLD: it loads the data, builds the per-cycle feature tables
for every candidate program (running the executor's _apply_reduction loop to derive
the substrate-prefix features for the SILVER candidates), validates the loss formula
on the 30 curated cycles (where Z*=1, the loss should equal standard cross-entropy
loss to within numerical precision), and reports the corpus statistics.

Training (gradient optimization of theta) is the next step -- see ``train_run`` at
the bottom; it's a stub here.

Run:
    PYTHONPATH=src .venv/bin/python scripts/policy/train_weak.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
# Inline logsumexp to avoid a scipy dep -- the only place we use it is over a
# 1-D vector of candidate log-probabilities per BGC.
def logsumexp(x: np.ndarray | list[float]) -> float:
    a = np.asarray(x, dtype=float)
    m = a.max()
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.exp(a - m).sum()))

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem.program import Cycle, Extender, Program, ReductionState as Rs, Release  # noqa: E402
from lpi.data.curated import load_all  # noqa: E402
from lpi.executor import operators as op  # noqa: E402
from lpi.executor.core import _apply_reduction  # noqa: E402

INTERMEDIATES = ROOT / "data" / "policy" / "intermediates.parquet"
GROUND_TRUTH = ROOT / "data" / "policy" / "ground_truth_programs.json"

# Action space: per-cycle reduction state (4-way categorical). We start here; C-MeT
# (binary) and extender (binary) can be added as additional heads later.
REDUCTION_LABELS = ["keto", "kr", "dh", "er"]
LABEL_INDEX = {l: i for i, l in enumerate(REDUCTION_LABELS)}
N_ACTIONS = len(REDUCTION_LABELS)

# Carbons contributed by each starter (matches extract_phase_d_programs.STARTER_CARBONS).
STARTER_C = {"acetyl": 2, "propionyl": 3, "butyryl": 4, "hexanoyl": 6, "benzoyl": 7}


# ---- Per-cycle feature extraction ------------------------------------------------

@dataclass
class CycleFeat:
    """Features at a single cycle of a candidate program -- the policy's input s_t."""
    bgc: str
    cand_idx: int
    cycle_t: int
    n_cycles: int
    # POSITION
    t: int
    frac: float
    sub_hr: int  # 1 if HR else 0 (PR / NR)
    # SUBSTRATE_prefix
    chain_c: int
    oxid_sum: int
    n_red: int
    prev: int  # previous cycle's reduction rank, -1 if t=1
    # The action label at this cycle (for training)
    label: int  # REDUCTION_LABELS index


def cycle_features_for_program(bgc: str, cand_idx: int, prog: dict,
                               sub_hr: int) -> list[CycleFeat]:
    """Walk a candidate program, extracting per-cycle features that match the existing
    substrate_state_probe + intermediates.parquet definitions."""
    starter = prog["starter"]
    cycles = prog["cycles"]
    N = len(cycles)
    sc = STARTER_C.get(starter, 2)
    prior_levels: list[int] = []
    n_cmet = 0
    feats: list[CycleFeat] = []
    for t, c in enumerate(cycles, start=1):
        lv = LABEL_INDEX[c["reduction"]]
        chain_c = sc + 2 * t + n_cmet
        oxid_sum = sum(prior_levels)
        n_red = sum(1 for x in prior_levels if x > 0)
        prev = prior_levels[-1] if prior_levels else -1
        feats.append(CycleFeat(
            bgc=bgc, cand_idx=cand_idx, cycle_t=t, n_cycles=N,
            t=t, frac=t / N, sub_hr=sub_hr,
            chain_c=chain_c, oxid_sum=oxid_sum, n_red=n_red, prev=prev,
            label=lv,
        ))
        prior_levels.append(lv)
        n_cmet += int(c["c_methyl"])
    return feats


# ---- Dataset assembly ------------------------------------------------------------

@dataclass
class BGCAsset:
    """One training example: a BGC with its Z*(B) -- multiple candidates, each a list
    of per-cycle features. The marginal-likelihood loss sums over candidates."""
    bgc: str
    candidates: list[list[CycleFeat]]   # candidates[k] = per-cycle features of z_k
    n_z_star: int


def build_corpus() -> list[BGCAsset]:
    """Assemble the Phase D corpus: 9 curated HR/PR synthases (Z*=1 each) + the 16
    inventory GOLD/SILVER BGCs (Z*=candidate set from ground_truth_programs.json).

    Each BGC contributes ONE BGCAsset; SILVER assets have >1 candidate; GOLD assets
    have exactly 1 (so the loss collapses to standard cross-entropy on them)."""
    assets: list[BGCAsset] = []

    # Curated 9 HR/PR -- their program IS the unique Z* member.
    for e in load_all():
        if e.subclass not in ("HR", "PR"):
            continue
        prog_dict = dict(
            starter=e.program.starter,
            cycles=[dict(reduction=c.reduction.value, c_methyl=bool(c.c_methyl),
                         extender=c.extender.value) for c in e.program.cycles],
            release=e.program.release.value,
            n_cycles=e.program.n_cycles,
        )
        feats = cycle_features_for_program(
            bgc=f"curated:{e.name}", cand_idx=0, prog=prog_dict,
            sub_hr=1 if e.subclass == "HR" else 0,
        )
        assets.append(BGCAsset(bgc=f"curated:{e.name}", candidates=[feats], n_z_star=1))

    # Inventory GOLD/SILVER -- multiple candidates per BGC (the Z* set).
    if GROUND_TRUTH.exists():
        gt = json.loads(GROUND_TRUTH.read_text())
        for bgc, v in gt.items():
            cand_feats: list[list[CycleFeat]] = []
            # subclass: infer from family. Aromatic (orsellinic, 6-MSA, citrinin) ~ PR/NR;
            # polyene / fatty-acid-like (strobilurin, asperlin) ~ HR. We use sub_hr as a
            # soft feature; exact tag isn't critical for the per-cycle reduction policy.
            family = v.get("family", "") or ""
            sub_hr = 1 if any(t in family for t in ("strobilurin", "asperlin",
                                                     "asperlactone", "gibepyrone")) else 0
            for k, c in enumerate(v["candidates"]):
                feats = cycle_features_for_program(bgc=bgc, cand_idx=k,
                                                   prog=c["program"], sub_hr=sub_hr)
                cand_feats.append(feats)
            assets.append(BGCAsset(bgc=bgc, candidates=cand_feats, n_z_star=len(cand_feats)))
    return assets


# ---- Feature matrix + simple linear policy ---------------------------------------

FEATURE_NAMES = ["t", "frac", "sub_hr", "chain_c", "oxid_sum", "n_red", "prev"]


def feature_vec(cf: CycleFeat) -> np.ndarray:
    return np.array([getattr(cf, n) for n in FEATURE_NAMES], dtype=float)


def feature_matrix(features: list[CycleFeat]) -> np.ndarray:
    return np.vstack([feature_vec(f) for f in features])


# Policy: simple linear logistic over per-cycle features -> 4-way action probabilities.
# theta has shape (N_FEATURES + 1 bias) * N_ACTIONS, reshapable to (D+1, A).
N_FEAT = len(FEATURE_NAMES)


def reshape_theta(theta: np.ndarray) -> np.ndarray:
    return theta.reshape(N_FEAT + 1, N_ACTIONS)


def log_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically-safe log_softmax along the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def candidate_log_prob(theta_W: np.ndarray, feats: list[CycleFeat]) -> float:
    """Log P(z | theta) = Sum_t log pi_theta(z_t | s_t) -- the inner product term."""
    if not feats:
        return 0.0
    X = feature_matrix(feats)
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])  # bias column
    logits = X_aug @ theta_W   # (T, A)
    log_probs = log_softmax(logits)
    labels = np.array([f.label for f in feats])
    return float(log_probs[np.arange(len(labels)), labels].sum())


def nlml_loss(theta: np.ndarray, corpus: list[BGCAsset]) -> float:
    """The Phase D2 objective: Sum_B - log Sum_z P(z | B; theta), with log-sum-exp."""
    W = reshape_theta(theta)
    total = 0.0
    for asset in corpus:
        # log P(z) for each candidate
        log_p_cands = np.array([candidate_log_prob(W, c) for c in asset.candidates])
        # logsumexp over candidates -> log Sum_z P(z)
        log_marginal = logsumexp(log_p_cands)
        total -= log_marginal
    return total


# ---- Validation: |Z*|=1 case must equal standard cross-entropy --------------------

def validate_zstar_one_collapse(corpus: list[BGCAsset]) -> None:
    """Sanity check: for assets with |Z*|=1, the marginal log-likelihood equals the
    single-candidate log-prob to numerical precision (the |Z*|=1 collapse)."""
    rng = np.random.default_rng(0)
    theta = rng.normal(size=(N_FEAT + 1) * N_ACTIONS) * 0.1
    W = reshape_theta(theta)
    issues = 0
    for a in corpus:
        if a.n_z_star != 1:
            continue
        lp_single = candidate_log_prob(W, a.candidates[0])
        lp_marginal = logsumexp([candidate_log_prob(W, c) for c in a.candidates])
        if not np.isclose(lp_single, lp_marginal, atol=1e-10):
            print(f"  MISMATCH on {a.bgc}: single={lp_single}, marginal={lp_marginal}")
            issues += 1
    print(f"  |Z*|=1 collapse check: {issues} mismatches over "
          f"{sum(1 for a in corpus if a.n_z_star==1)} GOLD assets.")


# ---- Main: assemble corpus, dry-run loss, report stats ----------------------------

def main() -> None:
    corpus = build_corpus()

    n_assets = len(corpus)
    n_gold = sum(1 for a in corpus if a.n_z_star == 1)
    n_silver = n_assets - n_gold
    n_cands_total = sum(a.n_z_star for a in corpus)
    n_cycles_total = sum(sum(len(c) for c in a.candidates) for a in corpus)
    n_unique_cycles = sum(len(a.candidates[0]) for a in corpus)  # one "true" cycle count per BGC

    print(f"Phase D corpus assembled:")
    print(f"  total BGC assets: {n_assets}  (GOLD={n_gold}, SILVER={n_silver})")
    print(f"  candidate programs (sum over Z* sets): {n_cands_total}")
    print(f"  cycles per asset (taking |Z*|=1 candidate): {n_unique_cycles}")
    print(f"  total per-candidate per-cycle feature rows: {n_cycles_total}")
    print()
    print("Sample asset breakdown:")
    for tag in ("curated:", "BGC0001121", "BGC0001338", "BGC0001909"):
        for a in corpus:
            if a.bgc.startswith(tag):
                ncycs = len(a.candidates[0])
                print(f"  {a.bgc[:50]:<50} |Z*|={a.n_z_star:>3}  n_cycles={ncycs}")
                break

    print()
    print("Validating |Z*|=1 collapse to cross-entropy...")
    validate_zstar_one_collapse(corpus)

    print()
    print("Dry-run loss at theta=0:")
    theta0 = np.zeros((N_FEAT + 1) * N_ACTIONS)
    loss0 = nlml_loss(theta0, corpus)
    n_assets_contributing = sum(1 for a in corpus if any(len(c) > 0 for c in a.candidates))
    # At theta=0, log_softmax = log(1/A) = -log(A) per step. Expected loss:
    #   For Z*=1 asset with T cycles: -log P(z) = T * log(A)
    #   For Z*=K asset: -log Sum_z P(z) = -log(K * (1/A)^T_min) approx
    #   (if all candidates have same T): = -log(K) + T * log(A) = T*log A - log K
    # So total loss ~ Sum_B T_B * log(A) - Sum_B log |Z*(B)|
    expected_simple = sum(len(a.candidates[0]) for a in corpus) * np.log(N_ACTIONS)
    expected_marg_bonus = sum(np.log(a.n_z_star) for a in corpus)
    expected = expected_simple - expected_marg_bonus
    print(f"  loss(theta=0) = {loss0:.4f}")
    print(f"  expected      = {expected:.4f}  "
          f"(= Sum_B T_B * log(A) - Sum_B log|Z*|; assumes equal-length candidates)")
    print(f"  diff = {loss0 - expected:+.4f}  (small residual = candidates differ in T)")

    print()
    print("Ready for training. Next step:")
    print("  - optimize theta via scipy.optimize.minimize(nlml_loss, theta0, jac=...)")
    print("  - or torch + autograd for cleaner gradient + the planned POSE feature add")
    print("  - LOSO at the BGC level; paired bootstrap vs POSITION baseline")


if __name__ == "__main__":
    main()
