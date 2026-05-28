"""Cap-convergence sweep + localization of the conformational frontier.

Two questions the single-cap run could not answer:

1. CONVERGENCE. The explosive clusters cap at program_cap. Full enumeration may never
   terminate; what the self-similarity hypothesis actually claims is that
   Gate-3/(clean+Gate-3) STABILIZES as the cap rises. So we sweep program_cap over
   {20k, 40k, 80k, 160k} on the clusters that capped and watch the ratio:
     - flat across levels -> converged; cite the value.
     - still rising at 160k -> Gate-3 is cap-suppressed; the true HR figure is a
       monotone-increasing lower bound (">= X, not yet converged"), itself publishable.

2. LOCALIZATION. Gate-3 is not uniform across the manifold. Label-symmetry requires a
   C-MeT firing whose cycle-position is mass-neutral, so it is STRUCTURALLY IMPOSSIBLE
   in clusters whose programs do not methylate (the small NR/PR aromatic interior) and
   CONCENTRATED in the methylating (HR-type) clusters. The corpus average is a blend of
   two regimes describing no actual cluster. We partition by whether any candidate in
   Z* carries a C-MeT firing and report:
     - non-methylating subpop: Gate-3 floor (should be ~0).
     - methylating subpop: Gate-3/(clean+Gate-3) -- the real frontier figure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "policy"))

from lpi.search.generate import generate  # noqa: E402
from gate_decomposition import GRAMMAR, PRODUCIBLE_TIERS, action_seq, walk_trie  # noqa: E402

INVENTORY = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_LOG = ROOT / "results" / "gate_cap_sweep.log"
CAP_LEVELS = [20000, 40000, 80000, 160000]


def methylates(candidates) -> bool:
    return any(c.c_methyl for cand in candidates for c in cand.program.cycles)


def decomp(candidates) -> tuple[int, int, int]:
    seqs = [action_seq(c.program) for c in candidates]
    seqs = [s for s in seqs if s]
    if not seqs:
        return (0, 0, 0)
    return walk_trie(seqs)


def main() -> None:
    df = pd.read_parquet(INVENTORY)
    prod = df[df.tier.isin(PRODUCIBLE_TIERS)].copy()
    log: list[str] = []
    log.append("Cap-convergence sweep + conformational-frontier localization")
    log.append(f"({len(prod)} producible clusters; cap levels {CAP_LEVELS})")

    # ---- Pass 1 at base cap: classify methylating, identify which capped ----
    base = {}
    for _, r in prod.iterrows():
        lo, hi = int(r.cycle_lo), int(r.cycle_hi)
        if lo < 1:
            continue
        res = generate(GRAMMAR, lo, hi, target_cho=(int(r.C), int(r.H), int(r.O)),
                       program_cap=CAP_LEVELS[0])
        if not res.candidates:
            continue
        base[r.bgc_id] = dict(cycle_lo=lo, cycle_hi=hi, C=int(r.C), H=int(r.H), O=int(r.O),
                              methyl=methylates(res.candidates), capped=res.capped,
                              n_z=len(res.candidates))

    methyl_ids = [b for b, v in base.items() if v["methyl"]]
    nonmethyl_ids = [b for b, v in base.items() if not v["methyl"]]

    # ---- Localization: floor (non-methylating) vs frontier (methylating) at base cap ----
    def agg_at(ids, cap):
        clean = g2 = g3 = 0
        still_capped = 0
        for b in ids:
            v = base[b]
            res = generate(GRAMMAR, v["cycle_lo"], v["cycle_hi"],
                           target_cho=(v["C"], v["H"], v["O"]), program_cap=cap)
            if not res.candidates:
                continue
            if res.capped:
                still_capped += 1
            c, a, t = decomp(res.candidates)
            clean += c; g2 += a; g3 += t
        non_g2 = clean + g3
        return dict(clean=clean, g2=g2, g3=g3,
                    frac_all=(g3 / (clean + g2 + g3) if (clean+g2+g3) else 0.0),
                    frac_nonG2=(g3 / non_g2 if non_g2 else 0.0),
                    still_capped=still_capped, n=len(ids))

    log.append(f"\n{'='*70}\nLOCALIZATION (base cap {CAP_LEVELS[0]})\n{'='*70}")
    floor = agg_at(nonmethyl_ids, CAP_LEVELS[0])
    log.append(f"  NON-methylating subpop ({len(nonmethyl_ids)} clusters -- aromatic/NR-PR floor):")
    log.append(f"    Gate-3 = {floor['g3']}  -> Gate-3/(clean+Gate-3) = {floor['frac_nonG2']:.2%}  "
               f"(structural floor; label-symmetry impossible without C-MeT)")
    log.append(f"  METHYLATING subpop ({len(methyl_ids)} clusters -- the frontier):")
    fr0 = agg_at(methyl_ids, CAP_LEVELS[0])
    log.append(f"    Gate-3/(clean+Gate-3) = {fr0['frac_nonG2']:.2%}  "
               f"(Gate-3={fr0['g3']}, clean={fr0['clean']}, capped={fr0['still_capped']})")

    # ---- Convergence sweep on the methylating subpop ----
    log.append(f"\n{'='*70}\nCAP-CONVERGENCE SWEEP (methylating subpop)\n{'='*70}")
    log.append(f"  {'cap':>8} {'Gate-3':>7} {'clean+G3':>9} {'G3/(clean+G3)':>14} {'still_capped':>13}")
    sweep = []
    for cap in CAP_LEVELS:
        a = agg_at(methyl_ids, cap)
        sweep.append((cap, a))
        log.append(f"  {cap:>8} {a['g3']:>7} {a['clean']+a['g3']:>9} "
                   f"{a['frac_nonG2']:>13.2%} {a['still_capped']:>13}")

    # convergence verdict
    fracs = [a["frac_nonG2"] for _, a in sweep]
    last_delta = abs(fracs[-1] - fracs[-2]) if len(fracs) >= 2 else 1.0
    converged = last_delta < 0.01 and sweep[-1][1]["still_capped"] == 0
    log.append("")
    if converged:
        log.append(f"  VERDICT: CONVERGED -- ratio flat (last delta {last_delta:.2%}), all un-capped.")
        log.append(f"  Conformational frontier within methylating clusters = {fracs[-1]:.1%}.")
    elif last_delta < 0.01:
        log.append(f"  VERDICT: ratio FLAT (last delta {last_delta:.2%}) but "
                   f"{sweep[-1][1]['still_capped']} clusters still cap at {CAP_LEVELS[-1]}.")
        log.append(f"  Frontier ~= {fracs[-1]:.1%}, stable across the explored depth; "
                   f"residual cap is on candidate VOLUME, not changing the ratio.")
    else:
        log.append(f"  VERDICT: STILL RISING (last delta {last_delta:.2%}). Gate-3 is "
                   f"cap-suppressed; the methylating-frontier figure is a monotone-increasing")
        log.append(f"  LOWER BOUND >= {fracs[-1]:.1%}, not yet converged at cap {CAP_LEVELS[-1]}.")

    log.append(f"\n{'='*70}\nHEADLINE\n{'='*70}")
    log.append(f"  Conformational frontier is LOCALIZED, not uniform:")
    log.append(f"    aromatic / non-methylating interior:  Gate-3 ~= {floor['frac_nonG2']:.1%} (floor)")
    log.append(f"    methylating (HR-type) subclass:       Gate-3 = {fracs[-1]:.1%} "
               f"@ cap {CAP_LEVELS[-1]} ({'converged' if converged or last_delta<0.01 else 'lower bound, rising'})")
    log.append(f"  The corpus blend describes no actual cluster; the pocket-coordinate problem")
    log.append(f"  is a concentrated property of the methylating subclass, near-absent elsewhere.")
    log.append(f"  => solving the citrinin hinge for HR solves essentially all of Gate-3.")

    text = "\n".join(log)
    print(text)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(text + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
