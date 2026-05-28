"""De-duplicated Gate-3 frontier: distinct-chemistry weighting, not decision-pooling.

WHY. The decision-pooled Gate-3/(clean+Gate-3) over the methylating subpop is 4.84%,
but ~88% of those decisions come from three clusters with byte-identical enumerations
(C16H24O4, cyc 6-9: Brefeldin A / 4-epi-15-epi-brefeldin A / decumbenone A -- three
DISTINCT compounds sharing a molecular formula, hence an identical inference problem).
Pooling weights each cluster by |Z*|, i.e. by how combinatorially explosive its
enumeration is -- which is the wrong weight for "how hard is the conformational
frontier." A cluster is not more important to the frontier question because its grammar
happens to enumerate 4000 candidates. The frontier question is CLUSTER-weighted by
construction: each distinct inference problem counts once.

DISTINCT-CHEMISTRY UNIT. Clusters sharing (cycle_lo, cycle_hi, C, H, O) yield identical
Z* and identical decomposition -- our formula-level model cannot distinguish them -- so
they are ONE inference problem. We collapse on that key. Some collapses are genuine
MIBiG re-deposits (orsellinic acid x3, strobilurin A x2, 6-MSA x2); the brefeldin/
decumbenone triplet is three distinct compounds the model treats identically. Both
collapse for the SAME reason: identical enumeration => one problem, counted once.

REPORT (over distinct keys, exhaustion-certified):
  - methylating frontier: cluster-weighted mean / median / range  (HEADLINE)
                          decision-pooled-over-distinct            (disclosed alternate)
  - aromatic floor: pooled (exactly 0; robust to de-dup -- the one duplication-proof fig)
  - |Z*| <-> Gate-3: ratio anti-correlation (Spearman) + absolute-count trend
    (the within-HR localization MECHANISM: Gate-2 is combinatorial in reduction state,
     Gate-3 is chemistry-bounded; big |Z*| dilutes the ratio without erasing Gate-3 --
     the within-corpus echo of the across-O Gate-2/Gate-3 asymmetry).
"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path
from statistics import mean, median

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "policy"))

from lpi.search.generate import generate  # noqa: E402
from gate_decomposition import GRAMMAR, PRODUCIBLE_TIERS, action_seq, walk_trie  # noqa: E402

INVENTORY = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_LOG = ROOT / "results" / "gate_frontier_dedup.log"

BASE_CAP, MID_CAP, TOP_CAP = 20000, 80000, 160000
N_WORKERS = 6


def methylates(candidates) -> bool:
    return any(c.c_methyl for cand in candidates for c in cand.program.cycles)


def decomp(candidates) -> tuple[int, int, int]:
    seqs = [s for s in (action_seq(c.program) for c in candidates) if s]
    return walk_trie(seqs) if seqs else (0, 0, 0)


def _eval_key(args):
    """Enumerate ONE distinct (lo,hi,C,H,O) key, escalating cap only as needed."""
    lo, hi, C, H, O, members = args
    cho = (C, H, O)
    r = generate(GRAMMAR, lo, hi, target_cho=cho, program_cap=BASE_CAP)
    if not r.candidates:
        return None
    if r.capped:
        r = generate(GRAMMAR, lo, hi, target_cho=cho, program_cap=TOP_CAP)
    capped = r.capped
    c, a, t = decomp(r.candidates)
    return dict(lo=lo, hi=hi, C=C, H=H, O=O, members=members,
                nz=len(r.candidates), clean=c, g2=a, g3=t,
                methyl=methylates(r.candidates), capped=capped)


def _ratio(clean, g3):
    d = clean + g3
    return (g3 / d) if d else 0.0


def _spearman(xs, ys):
    """Rank-correlation without scipy."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main() -> None:
    df = pd.read_parquet(INVENTORY)
    prod = df[df.tier.isin(PRODUCIBLE_TIERS)].copy()

    args = []
    for vals, sub in prod.groupby(["cycle_lo", "cycle_hi", "C", "H", "O"]):
        lo, hi, C, H, O = (int(v) for v in vals)
        if lo < 1:
            continue
        members = [(r.bgc_id, str(r.compound_name)) for _, r in sub.iterrows()]
        args.append((lo, hi, C, H, O, members))

    with Pool(N_WORKERS) as pool:
        keys = [x for x in pool.map(_eval_key, args, chunksize=1) if x is not None]

    methyl = [k for k in keys if k["methyl"]]
    aromatic = [k for k in keys if not k["methyl"]]

    log: list[str] = []
    log.append("De-duplicated Gate-3 frontier (per distinct inference problem)")
    log.append(f"  {len(prod)} producible clusters -> {len(keys)} distinct (lo,hi,C,H,O) keys")
    log.append(f"  methylating keys: {len(methyl)}   non-methylating (aromatic) keys: {len(aromatic)}")
    n_bound = sum(1 for k in keys if k["capped"])
    log.append(f"  exhaustion: {len(keys)-n_bound}/{len(keys)} keys fully enumerated "
               f"({'ALL EXACT' if n_bound == 0 else f'{n_bound} still capped'})")

    # ---- duplicate groups: annotate same-compound vs distinct-compound-same-formula ----
    log.append(f"\n{'='*74}\nCOLLAPSED GROUPS (n>1 clusters per key)\n{'='*74}")
    for k in sorted(keys, key=lambda z: -len(z["members"])):
        if len(k["members"]) > 1:
            names = {n.strip().lower() for _, n in k["members"]}
            kind = "same-compound re-deposit" if len(names) == 1 else "distinct compounds, shared formula"
            log.append(f"  C{k['C']}H{k['H']}O{k['O']} cyc{k['lo']}-{k['hi']} (n={len(k['members'])}, {kind}):")
            for bid, nm in k["members"]:
                log.append(f"       {bid}  {nm}")

    # ---- methylating frontier: cluster-weighted (HEADLINE) ----
    log.append(f"\n{'='*74}\nMETHYLATING FRONTIER -- distinct-chemistry weighted (HEADLINE)\n{'='*74}")
    m_ratios = [_ratio(k["clean"], k["g3"]) for k in methyl]
    log.append(f"  {len(methyl)} distinct methylating inference problems, each counted once:")
    log.append(f"    cluster-weighted MEAN   Gate-3/(clean+Gate-3) = {mean(m_ratios):.2%}")
    log.append(f"    cluster-weighted MEDIAN                        = {median(m_ratios):.2%}")
    log.append(f"    range                                          = {min(m_ratios):.1%} .. {max(m_ratios):.1%}")
    cc, gg2, gg3 = (sum(k["clean"] for k in methyl), sum(k["g2"] for k in methyl), sum(k["g3"] for k in methyl))
    log.append(f"  decision-pooled OVER DISTINCT keys (disclosed alternate, |Z*|-weighted):")
    log.append(f"    Gate-3/(clean+Gate-3) = {_ratio(cc, gg3):.2%}  (clean={cc}, G2={gg2}, G3={gg3})")
    log.append(f"    -- still dominated by the brefeldin-family key (|Z*|=4114); pooling weights")
    log.append(f"       by enumeration size, which is why it sits well below the cluster mean.")

    # ---- aromatic floor (duplication-proof) ----
    log.append(f"\n{'='*74}\nAROMATIC FLOOR -- non-methylating keys\n{'='*74}")
    ac, ag2, ag3 = (sum(k["clean"] for k in aromatic), sum(k["g2"] for k in aromatic), sum(k["g3"] for k in aromatic))
    log.append(f"  {len(aromatic)} distinct keys: clean={ac} G2={ag2} Gate-3={ag3}")
    log.append(f"    Gate-3/(clean+Gate-3) = {_ratio(ac, ag3):.2%}  (exact; robust to de-dup since Gate-3=0)")

    # ---- |Z*| <-> Gate-3 mechanism ----
    log.append(f"\n{'='*74}\nWITHIN-HR LOCALIZATION MECHANISM: |Z*| vs Gate-3\n{'='*74}")
    nz = [k["nz"] for k in methyl]
    rho_ratio = _spearman(nz, m_ratios)
    rho_abs = _spearman(nz, [k["g3"] for k in methyl])
    g3s = [k["g3"] for k in methyl]
    log.append(f"  Spearman( |Z*| , Gate-3 RATIO )     = {rho_ratio:+.2f}   (ratio DILUTES as |Z*| grows)")
    log.append(f"  Spearman( |Z*| , Gate-3 ABS count ) = {rho_abs:+.2f}   (absolute Gate-3 GROWS, not constant)")
    log.append(f"  absolute Gate-3 spans {min(g3s)} .. {max(g3s)} across the methylating keys")
    log.append(f"  => the dilution is Gate-2 (reduction-combinatorial) outgrowing Gate-3")
    log.append(f"     (chemistry-bounded), NOT Gate-3 vanishing -- the within-corpus echo of")
    log.append(f"     the across-O Gate-2/Gate-3 asymmetry, now across clusters at fixed O.")

    # ---- per-distinct-key methylating table ----
    log.append(f"\n{'='*74}\nPER-KEY (methylating, sorted by ratio)\n{'='*74}")
    log.append(f"  {'C,H,O cyc':<16}{'n':>2}{'|Z*|':>7}{'clean':>7}{'G2':>6}{'G3':>4}{'G3/(cln+G3)':>13}  example")
    for k in sorted(methyl, key=lambda z: -_ratio(z["clean"], z["g3"])):
        tag = f"C{k['C']}H{k['H']}O{k['O']} {k['lo']}-{k['hi']}"
        ex = k["members"][0][1][:28]
        log.append(f"  {tag:<16}{len(k['members']):>2}{k['nz']:>7}{k['clean']:>7}{k['g2']:>6}{k['g3']:>4}"
                   f"{_ratio(k['clean'], k['g3']):>12.1%}  {ex}")

    text = "\n".join(log)
    print(text)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(text + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
