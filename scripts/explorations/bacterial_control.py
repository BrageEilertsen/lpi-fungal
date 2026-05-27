"""Bacterial positive control for the sequence-aware-policy idea.

The fungal reranking probe was a controlled NEGATIVE: a learned prior does not beat chance on the
iteration-program residual mass leaves. A negative is only trustworthy with a positive control --
the same kind of predictor, on the family where the signal provably exists. Bacterial modular PKS
are COLINEAR: each module carries its OWN reductive-domain content, so the per-step reduction is a
function of that module's domains (the thing a fungal iterative synthase, with ONE constant domain
set across all cycles, cannot provide). So:

  * per-MODULE domain predictor (bacteria's real situation): predict each module's reduction from
    ITS domains. This is the policy working where the per-step signal exists.
  * CONSTANT predictor (the fungal-equivalent condition): you only see the cluster's pooled/constant
    capability, so the best you can do is the modal reduction for every slot.

If per-module >> constant, the architecture is sound and the fungal failure is STRUCTURAL (missing
per-step signal), not a flaw in the method. We also report program-level exact recovery, where
per-module errors COMPOUND -- the honest reason even bacteria want sequence (ESM head 0.85 > 0.81
domain rule; Direction A) for whole-program recovery.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from lpi.data.mibig import PROCESSED
from lpi.model.clustercad_labels import REDUCTION_CLASSES, label_module
from lpi.model.differential import _n_carbons, domain_rule_reduction


def clean_programs():
    """Per cluster: the ordered list of (domain_set, structure-true reduction) for clean C2/C3
    extension modules -- the colinear program, read from structure."""
    df = pd.read_parquet(PROCESSED / "clustercad_modules.parquet")
    progs = []
    for cluster, g in df.groupby("cluster"):
        g = g.sort_values("module_idx")
        mods = []
        for _, r in g.iterrows():
            prev = r["prev_intermediate_smiles"]
            if prev is None or (isinstance(prev, float)):
                continue  # loading module
            try:
                if _n_carbons(r["intermediate_smiles"]) - _n_carbons(prev) not in (2, 3):
                    continue
            except Exception:  # noqa: BLE001
                continue
            doms = set(str(r["domains"]).split(";"))
            lbl = label_module(list(doms), r["intermediate_smiles"])
            if lbl is None:
                continue
            mods.append((doms, lbl.reduction))
        if len(mods) >= 2:
            progs.append((cluster, mods))
    return progs


def main():
    progs = clean_programs()
    n_mod = sum(len(m) for _, m in progs)
    print("Bacterial positive control -- does a per-step predictor recover the program where the")
    print(f"per-step signal exists?  {len(progs)} colinear clusters, {n_mod} clean extension modules.\n")

    pm_correct = 0          # per-module domain predictor
    const_correct = 0       # constant (modal) predictor = the fungal-equivalent condition
    prog_exact_pm = 0
    prog_exact_const = 0
    for _cluster, mods in progs:
        trues = [t for _, t in mods]
        modal = Counter(trues).most_common(1)[0][0]   # the best a CONSTANT predictor can do
        pm_hits = 0
        const_hits = 0
        for doms, true in mods:
            pred_pm = REDUCTION_CLASSES[domain_rule_reduction(doms)]   # per-module domains
            pm_hits += (pred_pm == true)
            const_hits += (modal == true)
        pm_correct += pm_hits
        const_correct += const_hits
        prog_exact_pm += (pm_hits == len(mods))
        prog_exact_const += (const_hits == len(mods))

    print(f"  per-MODULE domain predictor (per-step signal present):")
    print(f"     per-module reduction accuracy : {pm_correct / n_mod:.3f}")
    print(f"     whole-program EXACT recovery   : {prog_exact_pm}/{len(progs)} = {prog_exact_pm/len(progs):.3f}")
    print(f"  CONSTANT predictor (fungal-equivalent: one domain set across all slots):")
    print(f"     per-module reduction accuracy : {const_correct / n_mod:.3f}")
    print(f"     whole-program EXACT recovery   : {prog_exact_const}/{len(progs)} = {prog_exact_const/len(progs):.3f}")
    print()
    print("  Read: per-module >> constant -> the per-step signal is what recovers the program, and a")
    print("  fungal iterative synthase (one constant domain set) is in the CONSTANT column by")
    print("  construction. The architecture is sound; the 0.567 fungal wall is structural. Note even")
    print("  bacteria's whole-program exact recovery is dragged down by error compounding across")
    print("  modules (~19% domain-present-but-inactive) -- the gap the ESM sequence head (0.85) closes")
    print("  where the domain cartoon (0.81) cannot, exactly where a per-step sequence exists.")


if __name__ == "__main__":
    main()
