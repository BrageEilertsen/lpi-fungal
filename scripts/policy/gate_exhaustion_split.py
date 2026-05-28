"""Exhaustion-segregated Gate-3 frontier: exact-where-enumerated, bounded-where-not.

The cap-convergence sweep (gate_cap_sweep.py) reports a single BLENDED
Gate-3/(clean+Gate-3) over the methylating subpopulation at each cap, plus a count
of clusters still capped. That answers the convergence-TREND question but blends a
measurement with a bound: a cluster whose Z* is still truncated at the top cap
contributes a LOWER BOUND, not a measured Gate-3, and must not be averaged into the
headline. This script segregates them.

EXHAUSTION CERTIFICATE. `capped=False` means generate() explored the entire program
space without truncation -- so Z* is COMPLETE, the trie over that cluster is the
COMPLETE trie, every branch node is real, every clean sub-trunk genuinely clean, and
Gate-3 is the EXACT count, not an estimate-modulo-cap.

We escalate the program cap only as far as each cluster needs:
  - uncapped at 20k                 -> Z* complete                      EXACT
  - capped at 20k, uncapped at 160k  -> Z* complete at 160k              EXACT
  - capped even at 160k             -> Z* still truncated; Gate-3 is a
                                       monotone-increasing LOWER BOUND   BOUND
                                       (we also record the 80k point to
                                        exhibit |Z*| still growing)

Per-cluster work is parallelized across cores (the explosive HR clusters cost
~500-1000 s each to exhaust; independent, so they fan out).

REPORT (two certified numbers, exhaustion status attached, never blended):
  - methylating (HR-type) EXACT subpop:  Gate-3/(clean+Gate-3) -- certified frontier
  - methylating BOUND clusters:          listed individually as lower bounds
  - non-methylating (aromatic NR/PR):    Gate-3/(clean+Gate-3) -- exact floor (~0)
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "policy"))

from lpi.search.generate import generate  # noqa: E402
from gate_decomposition import GRAMMAR, PRODUCIBLE_TIERS, action_seq, walk_trie  # noqa: E402

INVENTORY = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_LOG = ROOT / "results" / "gate_exhaustion_split.log"

BASE_CAP = 20000
MID_CAP = 80000
TOP_CAP = 160000
N_WORKERS = 6  # leave a core for the concurrent sweep + OS on an 8-core box


def methylates(candidates) -> bool:
    return any(c.c_methyl for cand in candidates for c in cand.program.cycles)


def decomp(candidates) -> tuple[int, int, int]:
    seqs = [action_seq(c.program) for c in candidates]
    seqs = [s for s in seqs if s]
    if not seqs:
        return (0, 0, 0)
    return walk_trie(seqs)


def _eval_cluster(args):
    """Escalate cap only as far as needed; return per-cluster record (all scalars)."""
    bgc_id, lo, hi, C, H, O = args
    cho = (C, H, O)
    r = generate(GRAMMAR, lo, hi, target_cho=cho, program_cap=BASE_CAP)
    if not r.candidates:
        return None
    if not r.capped:
        c, a, t = decomp(r.candidates)
        return dict(bgc=bgc_id, status="EXACT", cap=BASE_CAP, clean=c, g2=a, g3=t,
                    nz=len(r.candidates), methyl=methylates(r.candidates),
                    nz_mid=None, g3_mid=None, clean_mid=None)
    # capped at base -> escalate straight to top
    rt = generate(GRAMMAR, lo, hi, target_cho=cho, program_cap=TOP_CAP)
    if not rt.capped:
        c, a, t = decomp(rt.candidates)
        return dict(bgc=bgc_id, status="EXACT", cap=TOP_CAP, clean=c, g2=a, g3=t,
                    nz=len(rt.candidates), methyl=methylates(rt.candidates),
                    nz_mid=None, g3_mid=None, clean_mid=None)
    # still capped at top -> BOUND; grab the mid point to show |Z*| still growing
    rm = generate(GRAMMAR, lo, hi, target_cho=cho, program_cap=MID_CAP)
    c, a, t = decomp(rt.candidates)
    cm, am, tm = decomp(rm.candidates)
    return dict(bgc=bgc_id, status="BOUND", cap=TOP_CAP, clean=c, g2=a, g3=t,
                nz=len(rt.candidates), methyl=methylates(rt.candidates),
                nz_mid=len(rm.candidates), g3_mid=tm, clean_mid=cm)


def _ratio(clean, g3):
    d = clean + g3
    return (g3 / d) if d else 0.0


def main() -> None:
    df = pd.read_parquet(INVENTORY)
    prod = df[df.tier.isin(PRODUCIBLE_TIERS)].copy()

    args = []
    for _, r in prod.iterrows():
        lo, hi = int(r.cycle_lo), int(r.cycle_hi)
        if lo < 1:
            continue
        args.append((r.bgc_id, lo, hi, int(r.C), int(r.H), int(r.O)))

    # chunksize=1: explosive clusters are wildly heterogeneous in cost (~1 s vs ~1000 s),
    # so dispatch one at a time to keep the heavy ones spread across all workers.
    with Pool(N_WORKERS) as pool:
        recs = [x for x in pool.map(_eval_cluster, args, chunksize=1) if x is not None]

    methyl = [x for x in recs if x["methyl"]]
    nonmethyl = [x for x in recs if not x["methyl"]]
    m_exact = [x for x in methyl if x["status"] == "EXACT"]
    m_bound = [x for x in methyl if x["status"] == "BOUND"]
    nm_exact = [x for x in nonmethyl if x["status"] == "EXACT"]
    nm_bound = [x for x in nonmethyl if x["status"] == "BOUND"]

    def agg(xs):
        clean = sum(x["clean"] for x in xs)
        g2 = sum(x["g2"] for x in xs)
        g3 = sum(x["g3"] for x in xs)
        return clean, g2, g3

    log: list[str] = []
    log.append("Exhaustion-segregated Gate-3 frontier (exact-where-enumerated)")
    log.append(f"({len(recs)} producible clusters; caps base={BASE_CAP} mid={MID_CAP} top={TOP_CAP})")
    log.append(f"  classification: {len(methyl)} methylating (HR-type), "
               f"{len(nonmethyl)} non-methylating (aromatic NR/PR)")
    log.append(f"  exhaustion: methylating {len(m_exact)} EXACT / {len(m_bound)} BOUND; "
               f"non-methylating {len(nm_exact)} EXACT / {len(nm_bound)} BOUND")

    # ---- certified frontier: methylating EXACT only ----
    log.append(f"\n{'='*72}\nCERTIFIED FRONTIER -- methylating subpop, EXACT (uncapped) clusters\n{'='*72}")
    c, g2, g3 = agg(m_exact)
    log.append(f"  {len(m_exact)} clusters, Z* fully enumerated (capped=False):")
    log.append(f"    clean={c}  Gate-2={g2}  Gate-3={g3}")
    log.append(f"    Gate-3/(clean+Gate-3) = {_ratio(c, g3):.2%}   <-- EXACT, certified by enumeration")
    log.append(f"    Gate-3/all decisions  = {(g3/(c+g2+g3) if (c+g2+g3) else 0):.2%}")

    # ---- bound clusters: listed individually, NOT blended ----
    log.append(f"\n{'='*72}\nNOT YET CERTIFIED -- methylating clusters still capped at {TOP_CAP}\n{'='*72}")
    if not m_bound:
        log.append("  (none -- every methylating cluster exhausted; the frontier is EXACT corpus-wide)")
    else:
        log.append(f"  {len(m_bound)} cluster(s); Gate-3 is a monotone-increasing LOWER BOUND, segregated:")
        log.append(f"    {'bgc':<14}{'|Z*|@80k':>9}{'|Z*|@160k':>10}{'G3@80k':>8}{'G3@160k':>9}"
                   f"{'G3/(cln+G3)@160k':>18}")
        for x in sorted(m_bound, key=lambda z: -z["g3"]):
            log.append(f"    {x['bgc']:<14}{x['nz_mid']:>9}{x['nz']:>10}{x['g3_mid']:>8}{x['g3']:>9}"
                       f"{'>= '+format(_ratio(x['clean'], x['g3']), '.1%'):>18}")
        cb, g2b, g3b = agg(m_bound)
        log.append(f"  blended bound (reported as a bound only): Gate-3/(clean+Gate-3) >= {_ratio(cb, g3b):.1%}")

    # ---- aromatic floor: non-methylating ----
    log.append(f"\n{'='*72}\nAROMATIC FLOOR -- non-methylating subpop\n{'='*72}")
    c0, g20, g30 = agg(nm_exact)
    log.append(f"  {len(nm_exact)} EXACT clusters (aromatic NR/PR interior; no C-MeT firing possible):")
    log.append(f"    clean={c0}  Gate-2={g20}  Gate-3={g30}")
    log.append(f"    Gate-3/(clean+Gate-3) = {_ratio(c0, g30):.2%}   <-- EXACT floor "
               f"(label-symmetry structurally impossible without C-MeT)")
    if nm_bound:
        log.append(f"  WARNING: {len(nm_bound)} non-methylating cluster(s) still capped at {TOP_CAP}: "
                   f"{', '.join(x['bgc'] for x in nm_bound)}")

    # ---- per-cluster methylating table (for the paper, checkable) ----
    log.append(f"\n{'='*72}\nPER-CLUSTER (methylating subpop)\n{'='*72}")
    log.append(f"  {'bgc':<14}{'status':>7}{'cap':>8}{'|Z*|':>6}{'clean':>6}{'G2':>5}{'G3':>4}"
               f"{'G3/(cln+G3)':>13}")
    for x in sorted(methyl, key=lambda z: (z["status"], -z["g3"])):
        log.append(f"  {x['bgc']:<14}{x['status']:>7}{x['cap']:>8}{x['nz']:>6}{x['clean']:>6}"
                   f"{x['g2']:>5}{x['g3']:>4}{_ratio(x['clean'], x['g3']):>12.1%}")

    # ---- headline ----
    log.append(f"\n{'='*72}\nHEADLINE\n{'='*72}")
    ce, _, g3e = agg(m_exact)
    frontier_exact = _ratio(ce, g3e)
    c0h, _, g30h = agg(nm_exact)
    floor_exact = _ratio(c0h, g30h)
    if not m_bound:
        log.append(f"  The conformational frontier is characterized EXACTLY on both ends, by")
        log.append(f"  complete enumeration -- no cap asterisk:")
        log.append(f"    aromatic / non-methylating interior:  Gate-3 = {floor_exact:.1%}  (exact floor)")
        log.append(f"    methylating (HR-type) subclass:       Gate-3 = {frontier_exact:.1%}  (exact)")
        log.append(f"  Gate-3 is a LOCALIZED property of the highly-reducing subclass, near-absent")
        log.append(f"  over the aromatic cores -- both numbers measured, not bounded.")
    else:
        log.append(f"  Frontier characterized EXACTLY where enumerable, BOUNDED where not:")
        log.append(f"    aromatic / non-methylating interior:  Gate-3 = {floor_exact:.1%}  (exact floor)")
        log.append(f"    methylating EXACT subclass ({len(m_exact)} clusters): Gate-3 = {frontier_exact:.1%}  (exact)")
        log.append(f"    methylating BOUND ({len(m_bound)} clusters): Gate-3 >= bound, segregated above, NOT blended")
        log.append(f"  A still-growing Z* at cap {TOP_CAP} is a finding, not a nuisance: that cluster's")
        log.append(f"  true Gate-3 cannot be certified by exhaustion and is reported as a lower bound.")

    text = "\n".join(log)
    print(text)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(text + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
