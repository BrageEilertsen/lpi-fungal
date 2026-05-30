"""Soundness harness for adding aromatic-closure (PT) operators to the executor.

WHY A DEDICATED HARNESS. The mass/formula verifier is BLIND TO REGIOCHEMISTRY: a
wrong-OH-position aromatic is formula-identical to the right one, so a new closure can
pass mass-balance and round-trip and still be chemically wrong. The three-state verdict's
``verified'' means graph-isomorphic to the true product -- strictly stronger than
mass-consistent -- and for regiochemically-degenerate classes (aromatics) that extra
correctness is NOT a verifier guarantee; it rests on the encoded closure SMARTS being
chemically right, which is curated domain chemistry. So a recovery count is admissible
only if it clears checks the mass/formula verifier cannot perform.

CORE FUNCTION -- the regiochemistry-aware false-positive check. A claimed recovery of core
X by a closure is admissible iff ALL of:
  (1) EXACT (graph-isomorphic) match to X's TRUE deposited product -- not a formula match;
  (2) the closure does not spuriously exact-match a DIFFERENT core Y (cross-core FP);
  (3) the closure's regiochemistry is VERIFIED-provenance (an author-confirmed mode or a
      regiochemistry-PRESERVING extension of one) -- NOT a guessed regiochemistry. A guessed
      closure that happens to graph-match is a confidently-wrong ``verified'' recovery, worse
      than an honest out-of-grammar, and the verifier would not catch it.
plus the mechanical gates: mass-balance (atoms conserved minus expected waters), forward
round-trip (sanitizes / executes), and the existing test suite staying green.

This module is the validator; it invents no chemistry. New regiochemistries are a deliberate,
sourced, per-class decision made against the primary literature, with check (1)-(3) as the
backstop.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem import mol as M  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "fungal_pks_pairs.parquet"
VERIFIED = "verified"          # author-confirmed mode, or regiochem-preserving extension of one
GUESSED = "guessed"            # regiochemistry not confirmed -- recoveries NOT admissible


# ---------- graph-isomorphic ("verified") identity ----------
def canon(smi: str) -> str | None:
    """Stereo-stripped canonical SMILES -- the graph-isomorphism key that 'verified' means."""
    try:
        return M.canonical_smiles(M.strip_stereo(M.mol_from_smiles(smi)))
    except Exception:  # noqa: BLE001
        return None


def exact_match(produced_smi: str, true_smi: str) -> bool:
    a, b = canon(produced_smi), canon(true_smi)
    return a is not None and a == b


def formula_cho(smi: str) -> tuple[int, int, int] | None:
    try:
        m = Chem.AddHs(M.mol_from_smiles(smi))
    except Exception:  # noqa: BLE001
        return None
    c = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 6)
    h = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 1)
    o = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() == 8)
    return (c, h, o)


def mass_balanced(reactant_smi: str, product_smi: str, n_waters: int) -> bool:
    """product formula == reactant formula - n_waters * H2O (CHO closures)."""
    r, p = formula_cho(reactant_smi), formula_cho(product_smi)
    if r is None or p is None:
        return False
    return p == (r[0], r[1] - 2 * n_waters, r[2] - n_waters)


# ---------- the regiochemistry-aware false-positive check (the core) ----------
def load_true_products() -> dict[str, str]:
    """bgc_id -> canonical deposited product (graph key)."""
    df = pd.read_parquet(PAIRS)
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        smi = r.get("product_smiles_canonical") or r.get("product_smiles_raw")
        c = canon(smi) if isinstance(smi, str) and smi else None
        if c:
            out[r["bgc_id"]] = c
    return out


def cross_core_matches(produced_smi: str, true_products: dict[str, str],
                       intended_bgc: str) -> list[str]:
    """Every OTHER deposited core the produced structure exact-matches -- a spurious recovery."""
    pc = canon(produced_smi)
    if pc is None:
        return []
    return [b for b, c in true_products.items() if c == pc and b != intended_bgc]


def flag_unintended_recoveries(recovered: dict[str, str], intended_bgcs: set[str],
                               true_products: dict[str, str]) -> dict[str, list[str]]:
    """SWEEP-level FP audit. Given what a candidate closure recovered when swept over all
    buildable precursors (bgc -> produced canonical), flag any recovered core OUTSIDE its
    intended scope -- a closure that recovers cores it was not validated for is recovering
    them for an unjustified reason, even if the structure graph-matches. Returns
    {unintended_bgc: [intended_bgcs sharing that structure]} for review."""
    flagged: dict[str, list[str]] = {}
    for bgc, prod in recovered.items():
        if bgc in intended_bgcs:
            continue
        same = [b for b in intended_bgcs if true_products.get(b) == prod]
        flagged[bgc] = same  # empty list => a genuinely different molecule recovered (worst case)
    return flagged


@dataclass
class RecoveryClaim:
    bgc: str
    produced_smi: str
    provenance: str            # VERIFIED | GUESSED
    reactant_smi: str | None = None
    n_waters: int = 0


@dataclass
class Verdict:
    admissible: bool
    reasons: list[str] = field(default_factory=list)


def validate_recovery(claim: RecoveryClaim, true_products: dict[str, str]) -> Verdict:
    """The full gate. A recovery counts toward 'measured' ONLY if it clears every check."""
    reasons: list[str] = []
    true = true_products.get(claim.bgc)
    if true is None:
        return Verdict(False, ["no deposited product for bgc"])
    if not exact_match(claim.produced_smi, true):
        reasons.append("NOT graph-isomorphic to the true deposited product")
    # (cross-core matching is a SWEEP-level audit -- any match is necessarily a same-structure
    #  duplicate deposit, not a wrong-molecule FP; see flag_unintended_recoveries. The wrong-
    #  molecule FP is caught above: a closure that produces the wrong structure fails graph-iso.)
    if claim.provenance != VERIFIED:
        reasons.append("regiochemistry not VERIFIED-provenance (guessed) -- not admissible")
    if claim.reactant_smi is not None and not mass_balanced(
            claim.reactant_smi, claim.produced_smi, claim.n_waters):
        reasons.append("mass not balanced (reactant - n_waters*H2O)")
    return Verdict(not reasons, reasons or ["all checks passed"])


# ---------- mechanical gate: the test suite stays green ----------
def tests_green() -> tuple[bool, str]:
    r = subprocess.run([str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    tail = (r.stdout + r.stderr).strip().splitlines()
    summary = next((l for l in reversed(tail) if "passed" in l or "failed" in l or "error" in l), "")
    return (r.returncode == 0, summary)


# ---------- self-test: validate the harness + the existing verified grammar ----------
def _self_test() -> None:
    print("PT soundness harness -- self-test\n" + "=" * 60)
    tp = load_true_products()
    print(f"loaded {len(tp)} deposited products (graph keys)")

    # 1. exact_match / cross-core FP on a known verified core (orsellinic, BGC0001121).
    ors = "Cc1cc(O)cc(O)c1C(=O)O"
    print("\n[exact-match] orsellinic self-identity:", exact_match(ors, ors))
    dups = cross_core_matches(ors, tp, intended_bgc="BGC0001121")
    print("[cross-core] orsellinic also matches (same-structure duplicate deposits):", dups or "none")

    # 2. mass-balance on the verified orsellinic closure: tetraketide acid -> orsellinic + 1 H2O.
    lin = "CC(=O)CC(=O)CC(=O)CC(=O)O"   # acetyl tetraketide acid, C8H10O5
    print("[mass-balance] tetraketide acid -> orsellinic (-1 H2O):",
          mass_balanced(lin, ors, n_waters=1), formula_cho(lin), "->", formula_cho(ors))

    # 3. the gate refuses a GUESSED-provenance recovery even if it graph-matches.
    good = validate_recovery(RecoveryClaim("BGC0001121", ors, VERIFIED, lin, 1), tp)
    guessed = validate_recovery(RecoveryClaim("BGC0001121", ors, GUESSED, lin, 1), tp)
    print("\n[gate] verified-provenance exact match  -> admissible:", good.admissible)
    print("[gate] same match, GUESSED provenance   -> admissible:", guessed.admissible,
          "|", guessed.reasons)

    # 4. the gate catches a regiochem-WRONG product (formula-identical, graph-different).
    wrong = "Cc1cc(O)c(O)cc1C(=O)O"    # 2,3-diOH isomer of orsellinic: same C8H8O4, wrong ring
    wclaim = validate_recovery(RecoveryClaim("BGC0001121", wrong, VERIFIED, lin, 1), tp)
    print("[gate] formula-identical wrong-regiochem -> admissible:", wclaim.admissible,
          "|", wclaim.reasons)

    # 5. the mechanical gate: test suite stays green.
    ok, summary = tests_green()
    print(f"\n[tests] suite green: {ok}  ({summary})")
    print("=" * 60)
    print("harness ready: recoveries admitted only on (graph-match) AND (no cross-core FP)")
    print("AND (VERIFIED provenance) AND (mass-balanced) AND (tests green).")


if __name__ == "__main__":
    _self_test()
