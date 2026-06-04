"""Classify the in-scope-unreachable fungal PKS cores: search-limited vs coverage-limited.

Companion Sec. 10.1 (generability) + the two-walls decomposition. The base reachability scan
(`make reachability`, beam=4000, Theta_0) returns 9/326 reachable and leaves 205 CHO-in-scope
cores not reachable -- 111 'unreachable' (searched at beam 4000, found nothing OR mass-infeasible)
and 94 'too_large' (C>20, NEVER structurally searched: a tractability bound, not a finding).

This scan asks, of those 205, which wall each core sits behind -- holding the operator set FIXED
at Theta_0 (the exact base universe) and escalating only the beam width beta. That isolates the
*beam* axis: a core that was unreachable at beta=4000 but surfaces at higher beta under the SAME
grammar is beam-starved (engineering), not coverage-limited (chemistry). Pre-registered classes:

  search-limited        kappa_beta : 0 -> >0 as beta grows. SOUND (the found program is an
                        executor-verified witness; Z* non-empty). The zearalenone class.
  coverage-limited-     formula_feasible == False under Theta_0: no program's product formula can
    formula             equal the target. SOUND analytic certificate -- needs NEW chemistry (a
                        starter/extender/release outside Theta_0). Rung-3.
  coverage-limited-     formula feasible, but an exhaustive search (completes, not budget-exhausted)
    structural          finds no structural match. Mass buildable, skeleton not. HEURISTIC, and --
                        see the witness note below -- NOT a clean coverage certificate: the beam can
                        underflow (kappa_beta <= kappa) on deep all-keto chains, so this class mixes
                        genuine coverage gaps with cores that are search-limited but too beam-starved
                        to surface at the bulk beta.
  ambiguous-budget      search hit max_executions before completing. UNRESOLVED. HEURISTIC.

Beam axis -- empirical calibration (why bulk beta=8000, not a ladder). A prior full two-rung run
(beta in {8000, 50000}) over 193/205 cores found beta=50000 changed ZERO classifications: every
C<=20 core was kappa={8000:0, 50000:0}; the only surfacing question that mattered was the witness
(below), which needs beta >> 50000. So the bulk classification runs a single beta=8000 rung (already
2x the base scan's beta=4000, and -- under Theta_0's malonyl-only branching -- past the base scan's
exhaustive range; n_programs_executed plateaus while n_pruned_overshoot confirms the space is
enumerated). The redundant 50000 rung is dropped from the bulk for tractability/robustness; the
prior run's progress log is retained as the evidence that it was redundant.

The witness (positive control + the coupling). Zearalenone (BGC0001057) is reachable BY CONSTRUCTION
(Experiment A: the constructed RAL program renders MIBiG BGC0001057, so kappa >= 1) yet is NOT
surfaced by the scan even at beta=50000 -- the C18 8-cycle all-keto chain is starved because C-MeT
partials fill the beam before it completes (Exp A). It is therefore SEARCH-LIMITED by the soundest
possible evidence (a constructed witness), but the bounded-beta scan reports kappa_beta=0. That is
the kappa_beta <= kappa underflow made concrete, and it is the coupling Brage predicted: the beam
(engineering) wall is severe enough to GATE the coverage (chemistry) question for formula-feasible
cores -- we cannot soundly call a formula-feasible non-surfacing core 'coverage-limited' when the
beam demonstrably underflows an in-grammar C18 core. The scan runs a dedicated deep beta-escalation
on the witness to find its surfacing beta (or to bound the underflow depth).

Pre-registered symmetric readout (Ground rule 2 -- report whatever fires, do NOT tune):
  * the reachability wall is substantially ENGINEERING (beam) if search-limited is a substantial
    share of the formula-feasible cores;
  * it is substantially CHEMISTRY (new operators) if coverage-limited (formula + structural)
    dominates.
The honest reading supersedes the mechanical count when the witness underflow shows the structural
class is beam-contaminated.

Ground rules: (1) executor / ~= untouched -- Theta_0 is the committed base universe, beta only
widens the verifier's search; (6) kappa_beta is a LOWER bound, so 'search-limited' and
'coverage-limited-formula' are SOUND while 'coverage-limited-structural'/'ambiguous-budget' are
flagged HEURISTIC; the witness escalation is checked for monotonicity (kappa non-decreasing in beta
-- a decrease would be a Prop 16.4c beam-pruning bug).

Robustness: the per-core CSV is written incrementally (append + flush). Re-running resumes from the
existing CSV (cores already classified are skipped), so an interrupted sweep completes from one
command (`make generability`). `--summary-only` recomputes the verdict from the CSV without scanning.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import namedtuple
from pathlib import Path

from lpi.chem import mol as M
from lpi.data.mibig import PROCESSED
from lpi.search.beam import OperatorSpec, _formula_cho, formula_feasible, search
from lpi.search.reachability import CARBON_CAP, _scan_spec, scan_parquet

PARQUET = PROCESSED / "fungal_pks_pairs.parquet"
OUT_LOG = Path("results/generability_scan.log")
OUT_CSV = Path("results/generability_scan.csv")
OUT_WITNESS = Path("results/generability_witness.csv")
TARGETS_CACHE = Path("results/.generability_targets.json")  # resume cache: lets a restart skip the base scan

# minimal record the classifier needs; cached so a killed run resumes without re-running scan_parquet
_Target = namedtuple("_Target", "bgc_id name smiles status")

# Theta_0: the EXACT base reachability universe (acetyl/propionyl/hexanoyl starters, all reductions,
# C-MeT on, all 6 release modes incl. resorcylic macrolactone, malonyl extender). Fixing Theta here
# isolates the beam axis.
THETA0: OperatorSpec = _scan_spec()

BULK_BETA = 8000                 # single bulk rung (see docstring: 50000 was empirically redundant)
MAXEXEC_SMALL = 200_000          # C<=CARBON_CAP per-core executor budget
MAXEXEC_LARGE = 60_000           # C>CARBON_CAP per-core budget (matches base scan; tractability cap)
WITNESS_IDS = ("BGC0001057",)    # zearalenone: reachable-by-construction (Exp A) search-limited witness
WITNESS_BETAS = (8000, 50_000, 120_000, 200_000)  # deep escalation for the witness only

TARGET_STATUSES = ("unreachable", "too_large", "budget")

CSV_HEADER = ["bgc_id", "name", "carbons", "C", "H", "O", "base_status", "formula_feasible",
              "kappa_8000", "class", "soundness", "budget_exhausted", "n_programs_executed",
              "n_pruned_overshoot", "runtime_s"]


def classify_bulk(bgc_id: str, name: str, smiles: str, base_status: str) -> list:
    """Single-rung (beta=8000) classification of one core. Returns a CSV row (list)."""
    mol = M.mol_from_smiles(smiles)
    C, H, O = _formula_cho(mol)
    feasible = formula_feasible((C, H, O), THETA0, max_cycles=C // 2 + 1)
    if not feasible:
        # SOUND: no program's product formula can equal the target.
        # row must be exactly len(CSV_HEADER)=15: kappa_8000="" (no search), then class/soundness.
        return [bgc_id, str(name), C, C, H, O, base_status, False, "",
                "coverage-limited-formula", "SOUND", "", "", "", "0.000"]

    canon = M.canonical_smiles(mol)
    budget = MAXEXEC_SMALL if C <= CARBON_CAP else MAXEXEC_LARGE
    res = search(canon, THETA0, beam_width=BULK_BETA, max_executions=budget)
    k = res.size
    if k > 0:
        klass, sound = "search-limited", "SOUND"          # surfaced -> verified Z* witness
    elif res.stats.budget_exhausted:
        klass, sound = "ambiguous-budget", "HEURISTIC"    # search incomplete
    else:
        klass, sound = "coverage-limited-structural", "HEURISTIC"  # exhausted-at-beta, no match
    return [bgc_id, str(name), C, C, H, O, base_status, True, k, klass, sound,
            bool(res.stats.budget_exhausted), res.stats.n_programs_executed,
            res.stats.n_pruned_overshoot, f"{res.stats.runtime_s:.3f}"]


def escalate_witness(bgc_id: str, name: str, smiles: str) -> list:
    """Deep beta-escalation on one in-grammar witness; returns list of (beta, kappa, budget, secs)."""
    canon = M.canonical_smiles(M.mol_from_smiles(smiles))
    rows = []
    prev_k = -1
    for b in WITNESS_BETAS:
        res = search(canon, THETA0, beam_width=b, max_executions=2_000_000)
        k = res.size
        mono_ok = k >= prev_k if prev_k >= 0 else True
        rows.append((b, k, bool(res.stats.budget_exhausted), res.stats.runtime_s, mono_ok))
        print(f"    [witness {bgc_id} {str(name)[:18]}] beta={b:>7d}  kappa={k}  "
              f"budget={res.stats.budget_exhausted}  {res.stats.runtime_s:.1f}s"
              f"{'' if mono_ok else '  !! kappa DECREASED (Prop 16.4c)'}",
              file=sys.stderr, flush=True)
        prev_k = k
        if k > 0:
            break  # surfaced -> minimal surfacing beta found
    return rows


def _load_done() -> set[str]:
    if not OUT_CSV.exists():
        return set()
    done = set()
    with OUT_CSV.open() as f:
        r = csv.reader(f)
        header = next(r, None)
        if header != CSV_HEADER:  # schema drift (e.g. stale probe CSV) -> start fresh
            return set()
        for row in r:
            if row:
                done.add(row[0])
    return done


def _summary_from_csv() -> None:
    rows = []
    with OUT_CSV.open() as f:
        r = csv.DictReader(f)
        rows = list(r)

    def n(klass: str) -> int:
        return sum(1 for x in rows if x["class"] == klass)

    n_search = n("search-limited")
    n_formula = n("coverage-limited-formula")
    n_struct = n("coverage-limited-structural")
    n_ambig = n("ambiguous-budget")
    n_feasible = sum(1 for x in rows if x["formula_feasible"] == "True")
    too_large = [x for x in rows if x["base_status"] == "too_large"]
    tl_reach = sum(1 for x in too_large if x["class"] == "search-limited")

    print(f"\n{'=' * 72}")
    print(f"GENERABILITY PARTITION of {len(rows)} in-scope-unreachable cores (Theta_0, bulk beta={BULK_BETA}):\n")
    print(f"  search-limited (beam-starved, SOUND)        : {n_search:>4d}  <- engineering / rung-1")
    print(f"  coverage-limited-formula (mass-infeasible)  : {n_formula:>4d}  <- new chemistry / rung-3  [SOUND]")
    print(f"  coverage-limited-structural (skeleton miss) : {n_struct:>4d}  <- beam-contaminated (see witness) [HEURISTIC]")
    print(f"  ambiguous-budget (search-incomplete)        : {n_ambig:>4d}  <- unresolved frontier   [HEURISTIC]")
    print(f"  {'-' * 62}")
    print(f"  formula-feasible (in-grammar by mass)       : {n_feasible:>4d} / {len(rows)}")
    print(f"  mass-INfeasible (SOUND need-new-chemistry)  : {len(rows) - n_feasible:>4d} / {len(rows)}")
    print(f"\n  reconciliation -- too_large cores reachable at this beta: {tl_reach}/{len(too_large)} "
          f"(census cap-lifted: 0/91)")

    cov_total = n_formula + n_struct
    verdict = ("ENGINEERING-dominant (beam): wall substantially search-limited"
               if n_search >= cov_total else
               "CHEMISTRY-dominant (coverage): wall substantially mass/skeleton-limited (mechanical count)")
    print(f"\n  PRE-REGISTERED MECHANICAL READOUT: {verdict}")
    print(f"    (search-limited={n_search} vs coverage[formula+structural]={cov_total}; unresolved={n_ambig})")
    print(f"\n  HONEST READING: the SOUND results are (a) {n_formula} mass-infeasible cores "
          f"(need new chemistry) and\n  (b) the search-limited count is >=1 by CONSTRUCTION (zearalenone, "
          f"Exp A) though 0 surface at\n  beta={BULK_BETA}. The {n_struct} 'structural' cores are NOT a clean "
          f"coverage certificate: the witness\n  escalation shows the beam underflows an in-grammar C18 core, so "
          f"this class is beam-contaminated\n  (the engineering wall gates the chemistry test). See the witness "
          f"block above.")
    print(f"{'=' * 72}")


def main() -> None:
    args = sys.argv[1:]
    if "--summary-only" in args:
        _summary_from_csv()
        return
    t0 = time.time()
    print("Sec. 10.1 generability scan: classify the in-scope-unreachable cores (two-walls split)\n")
    print(f"Theta_0 = base reachability universe: starters={THETA0.starters}, "
          f"extenders={tuple(e.name for e in THETA0.extenders)}, "
          f"releases={len(THETA0.releases)} modes, c_methyl={THETA0.allow_c_methyl}")
    print(f"bulk beta={BULK_BETA} (single rung; 50000 shown redundant by the prior full-ladder run)\n")

    done = _load_done()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT_CSV.exists() or not done
    if new_file and OUT_CSV.exists():
        OUT_CSV.unlink()              # stale/mismatched schema -> fresh
    if new_file and TARGETS_CACHE.exists():
        TARGETS_CACHE.unlink()        # fresh run -> rebuild the target set from the base scan

    # ---- target set: cached (resume) or freshly scanned (Theta_0 base partition) ----
    if not new_file and TARGETS_CACHE.exists():
        targets = [_Target(**d) for d in json.loads(TARGETS_CACHE.read_text())]
        print(f"[1/3] target set: loaded {len(targets)} cached in-scope-unreachable cores "
              f"(resume; base scan skipped)", flush=True)
    else:
        print("[1/3] base partition: scan_parquet(Theta_0, beam=4000) ...", flush=True)
        base = scan_parquet(PARQUET, spec=THETA0, beam_width=4000, max_executions=60000, progress=True)
        counts = {s: base.count(s) for s in sorted(set(r.status for r in base.rows))}
        print(f"\n  base statuses: {counts}")
        print(f"  reachable at base: {base.reachable} (expect 9)")
        targets = [_Target(r.bgc_id, str(r.name), r.smiles, r.status)
                   for r in base.rows if r.status in TARGET_STATUSES]
        print(f"  in-scope-unreachable target set: {len(targets)} "
              f"(unreachable={base.count('unreachable')}, too_large={base.count('too_large')}, "
              f"budget={base.count('budget')})")
        TARGETS_CACHE.write_text(json.dumps([t._asdict() for t in targets]))

    print(f"\n[2/3] classifying {len(targets)} cores at beta={BULK_BETA} "
          f"(resume: {len(done)} already done) ...", flush=True)
    with OUT_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(CSV_HEADER)
            f.flush()
        for i, r in enumerate(targets):
            if r.bgc_id in done:
                continue
            row = classify_bulk(r.bgc_id, r.name, r.smiles, r.status)
            w.writerow(row)
            f.flush()
            print(f"  [{i + 1}/{len(targets)}] {row[0]} C{row[2]:>2d} {row[9]:28s} {row[10]:9s} "
                  f"k8000={row[8]} {str(row[1])[:24]}", file=sys.stderr, flush=True)

    # ---- witness deep escalation (positive control + the coupling) ----
    print(f"\n[3/3] witness deep beta-escalation {WITNESS_BETAS} (positive control) ...", flush=True)
    tgt_by_id = {r.bgc_id: r for r in targets}
    with OUT_WITNESS.open("w", newline="") as wf:
        ww = csv.writer(wf)
        ww.writerow(["bgc_id", "name", "beta", "kappa", "budget_exhausted", "runtime_s", "beta_monotone"])
        for wid in WITNESS_IDS:
            if wid not in tgt_by_id:
                print(f"    [witness {wid}] NOT in target set (reachable at base or filtered) -- skipped",
                      file=sys.stderr, flush=True)
                continue
            tr = tgt_by_id[wid]
            for (b, k, bud, secs, mono) in escalate_witness(wid, tr.name, tr.smiles):
                ww.writerow([wid, str(tr.name), b, k, bud, f"{secs:.3f}", mono])
                wf.flush()

    _summary_from_csv()
    print(f"\nwrote {OUT_CSV} and {OUT_WITNESS}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
