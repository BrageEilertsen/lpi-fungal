"""Duality regression -- Theorem 1 property test: Z*(verifier) == D(y;O), across all enumerable grammars.

Theorem 1 (the engine's soundness+completeness backbone): for an observable set O, the verifier's returned
candidate set Z* equals the set of all alphabet-producible programs whose ACTUAL product is O-consistent:

    Z*(z;O) = [z]_O = D(y;O).

We test the load-bearing half -- the accurate-mass observable -- INDEPENDENTLY of the verifier's own
prefilter, for every enumerable grammar (PKS, NRPS, HYBRID):

  1. F = enumerate(alpha, lo, hi, target_cho=None)         # full producible set, NO prefilter
     -- assert NOT capped. Ground rule 1: a capped set is not exhaustive, so a capped run VOIDS the test
        (we never let |Z*| stand as exhaustive when it was truncated).
  2. Partition F by the ACTUAL product formula -- RDKit element count of each executed SMILES (same H
     convention as beam._formula_cho), projected to the grammar's key elements (PKS: C,H,O; NRPS/HYBRID:
     C,H,N,O). For EVERY distinct formula T present in F:
       Z_pref = enumerate(alpha, lo, hi, target_cho=T)     # the verifier's internal-prefilter path
       D      = { c in F : actual_formula(c) == T }          # forward-enumerate-then-filter, independent
     -- assert NOT capped on Z_pref;
     -- assert SOUNDNESS: every c in Z_pref has actual_formula(c) == T (no off-mass product slips in);
     -- assert EQUALITY:  { canonical SMILES in Z_pref } == { canonical SMILES in D }.
     Equality <=> the mass prefilter induces EXACTLY the partition the real product formula does
                  <=> the prefilter is SOUND (adds nothing off-mass) and COMPLETE (drops nothing on-mass)
                  <=> Z* = D(y;O), the Theorem 1 duality at the mass observable.

  This is independent of the prefilter because D is built from the executed products' real RDKit formulae,
  not from the analytic prefilter (PKS post-exec _formula_cho; NRPS analytic peptide_formula; HYBRID's
  water-corrected PK-formula derivation) -- so the test PITS the analytic prefilter against the actual
  executor output. A failure is a real soundness/completeness gap and is reported as the result, not tuned.

Also asserted:
  * lossless projection: every candidate's element set subseteq the grammar key (else the (C,H,O[,N])
    projection would hide an element -> flagged, never silently passed).
  * out-of-grammar guard: LANTHI.enumerate raises (registered SCAFFOLD, no executor).

Soundness invariants honored: == is canonical-SMILES set equality (graph iso up to canonicalization --
the engine's own dedup key, NOT relaxed to substructure); ring-closure slack untouched; capped runs void
the test rather than being reported as exhaustive.

Single command: `make duality`
"""
from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RDLogger.DisableLog("rdApp.*")

from lpi.chem.program import ReductionState, Release  # noqa: E402
from lpi.grammars import HYBRID, LANTHI, NRPS, PKS  # noqa: E402
from lpi.grammars.hybrid import HybridAlphabet  # noqa: E402
from lpi.grammars.nrps import NRPSAlphabet  # noqa: E402
from lpi.grammars.nrps import Release as NRel  # noqa: E402
from lpi.search.generate import Alphabet  # noqa: E402

OUT_LOG = ROOT / "results" / "duality_regression.log"


def element_counts(smiles: str) -> dict[str, int]:
    """{symbol: count} with H including implicit (same convention as beam._formula_cho)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    counts: dict[str, int] = {}
    h = 0
    for a in mol.GetAtoms():
        h += a.GetTotalNumHs()
        if a.GetAtomicNum() == 1:
            h += 1
        else:
            counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    counts["H"] = counts.get("H", 0) + h
    return counts


def project(counts: dict[str, int], key: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(counts.get(el, 0) for el in key)


def present_elements(counts: dict[str, int]) -> set[str]:
    return {el for el, n in counts.items() if n > 0}


# ---- test cases: (label, grammar, alphabet, lo, hi, key_elements) ----------------

def cases():
    out = []
    # PKS -- full reduction alphabet + C-MeT, acetyl, cycles 1..3 (584 programs, uncapped)
    out.append((
        "PKS full(4-red+cmet), acetyl, n=1..3", PKS,
        Alphabet(reductions=(ReductionState.KETO, ReductionState.KR,
                             ReductionState.DH, ReductionState.ER),
                 releases=(Release.HYDROLYSIS,), starters=("acetyl",), allow_cmet=True),
        1, 3, ("C", "H", "O")))
    # PKS -- two starters, no cmet, cycles 1..3 (tests starter-dependent formulas)
    out.append((
        "PKS 4-red, acetyl+propionyl, n=1..3", PKS,
        Alphabet(reductions=(ReductionState.KETO, ReductionState.KR,
                             ReductionState.DH, ReductionState.ER),
                 releases=(Release.HYDROLYSIS,), starters=("acetyl", "propionyl"),
                 allow_cmet=False),
        1, 3, ("C", "H", "O")))
    # NRPS -- all 8 residues, hydrolysis + macrolactam, peptides len 1..3
    out.append((
        "NRPS 8-residue, hydrolysis+macrolactam, n=1..3", NRPS,
        NRPSAlphabet(releases=(NRel.HYDROLYSIS, NRel.MACROLACTAM)),
        1, 3, ("C", "H", "N", "O")))
    # HYBRID -- default PK alphabet, single-residue handoff, PK cycles 1..2
    out.append((
        "HYBRID default, max_residues=1, PK n=1..2", HYBRID,
        HybridAlphabet(pks=Alphabet(reductions=(ReductionState.KETO, ReductionState.KR,
                                                ReductionState.DH, ReductionState.ER),
                                    releases=(Release.HYDROLYSIS,), starters=("acetyl",),
                                    allow_cmet=False)),
        1, 2, ("C", "H", "N", "O")))
    return out


def run_case(label, grammar, alpha, lo, hi, key, emit):
    emit(f"[{grammar.name}] {label}")
    full = grammar.enumerate(alpha, lo, hi, target_cho=None)
    if full.capped:
        emit(f"    VOID: full enumeration CAPPED at program_cap "
             f"(tried={full.programs_tried}); cannot test exhaustively. SKIP.")
        return dict(grammar=grammar.name, label=label, status="VOID", n_full=0,
                    n_formulas=0, n_pass=0, mismatches=[])
    F = full.candidates
    # lossless-projection guard
    lossy = [c.smiles for c in F if not present_elements(element_counts(c.smiles)) <= set(key)]
    if lossy:
        emit(f"    WARN: {len(lossy)} candidates carry an element outside key {key} "
             f"(projection lossy) e.g. {lossy[0]}")

    by_T: dict[tuple[int, ...], set[str]] = {}
    for c in F:
        T = project(element_counts(c.smiles), key)
        by_T.setdefault(T, set()).add(c.smiles)

    n_formulas = len(by_T)
    n_pass = 0
    mismatches = []
    for T, D_smiles in sorted(by_T.items()):
        pref = grammar.enumerate(alpha, lo, hi, target_cho=T)
        if pref.capped:
            mismatches.append((T, "PREF_CAPPED", 0, 0))
            continue
        pref_smiles = {c.smiles for c in pref.candidates}
        # soundness: nothing off-mass in the prefiltered set
        offmass = {s for s in pref_smiles if project(element_counts(s), key) != T}
        missing = D_smiles - pref_smiles      # completeness failure (verifier dropped a consistent z)
        extra = pref_smiles - D_smiles         # soundness failure (verifier added an inconsistent z)
        if not offmass and not missing and not extra:
            n_pass += 1
        else:
            mismatches.append((T, f"offmass={len(offmass)} missing={len(missing)} extra={len(extra)}",
                               len(missing), len(extra)))

    status = "PASS" if n_pass == n_formulas else "FAIL"
    emit(f"    n_full={len(F)}  distinct_formulas={n_formulas}  formulas_matching_D={n_pass}  -> {status}")
    if mismatches:
        for T, desc, _m, _e in mismatches[:8]:
            emit(f"      MISMATCH T={T}: {desc}")
    return dict(grammar=grammar.name, label=label, status=status, n_full=len(F),
                n_formulas=n_formulas, n_pass=n_pass, mismatches=mismatches)


def main():
    L = []
    def emit(s=""):
        print(s, flush=True)
        L.append(s)

    emit("Duality regression -- Theorem 1: Z*(verifier) == D(y;O) at the accurate-mass observable")
    emit("=" * 84)
    emit("For every distinct ACTUAL product formula in the full producible set, the verifier's mass")
    emit("prefilter must return EXACTLY the producible programs with that formula (sound + complete).")
    emit("Capped enumerations VOID the test (never reported as exhaustive). == is canonical-SMILES")
    emit("set equality (graph iso up to canonicalization), NOT substructure.")
    emit("")

    results = [run_case(*c, emit) for c in cases()]
    emit("")

    # out-of-grammar guard: LANTHI is a registered scaffold; enumerate must raise.
    emit("[lanthi] out-of-grammar guard: enumerate must raise (registered SCAFFOLD, no executor)")
    try:
        LANTHI.enumerate(object(), 1, 2)
        lanthi_ok = False
        emit("    FAIL: LANTHI.enumerate did NOT raise")
    except Exception as e:  # noqa: BLE001
        lanthi_ok = True
        emit(f"    PASS: raised {type(e).__name__}")
    emit("")

    emit("=" * 84)
    all_pass = all(r["status"] in ("PASS", "VOID") for r in results) and lanthi_ok
    any_void = any(r["status"] == "VOID" for r in results)
    n_tested = sum(r["n_formulas"] for r in results)
    n_ok = sum(r["n_pass"] for r in results)
    for r in results:
        emit(f"  {r['grammar']:<7} {r['status']:<5} {r['n_pass']}/{r['n_formulas']} formulas  "
             f"(n_full={r['n_full']})  -- {r['label']}")
    emit(f"  lanthi  {'PASS' if lanthi_ok else 'FAIL':<5} out-of-grammar guard")
    emit("")
    if all_pass and not any_void:
        verdict = (f"SUPPORTED: Theorem-1 duality holds across PKS/NRPS/HYBRID -- {n_ok}/{n_tested} "
                   "product formulas, the verifier's Z* == the independent O-consistent set D(y;O) "
                   "exactly. Mass prefilter is sound AND complete; LANTHI guard holds.")
    elif all_pass and any_void:
        verdict = (f"SUPPORTED (partial): duality holds on every NON-void case ({n_ok}/{n_tested} "
                   "formulas); one or more cases VOIDED by capping -- re-run with a smaller alphabet.")
    else:
        verdict = ("REFUSED: at least one grammar's mass prefilter does NOT reproduce the independent "
                   "O-consistent set -- a verifier soundness/completeness gap (see MISMATCH lines). "
                   "Reported as the result; NOT tuned away.")
    emit(f"VERDICT: {verdict}")

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(L) + "\n")
    emit(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
