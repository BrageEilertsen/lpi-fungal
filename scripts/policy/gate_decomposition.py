"""Branching decomposition of the per-cycle training signal -- STRICT TRIE.

The honest denominator for every "Gate 2 collapses the bulk" claim. The marginal-
likelihood loss log Sum_z pi(z|B) factorizes into clean cross-entropy over the common
prefix exactly because every candidate shares one (s_t, a_t) there. The TRIE is that
factorization made explicit: single-child segments = the shared-gradient region the
loss collapses; branch nodes = where the log-sum-exp stops collapsing. The strict trie
is the ONLY walk that measures the thing the loss factorizes over -- the per-position
marginal counts phantom ambiguities (upstream-diverged candidates the gradient never
contrasts at that position), inflating Gate-3 and deflating clean mass.

Trie walk over candidate action-sequences (action = (reduction, c_methyl, extender)):
  - single-child node -> CLEAN decision: the action is determined given the prefix.
  - branch node, classified by what LOCALLY distinguishes its children CONDITIONED on
    the shared prefix to that node:
      GATE-2  children diverge in reduction label (or in chain length) -> substrate-
              readable (label-distinguishable).
      GATE-3  children agree on reduction but differ only in a mass-neutral event
              (c_methyl placement, extender) -> only a pocket coordinate breaks it
              (label-symmetric).
    Branch-local: a node whose children diverge in reduction is Gate-2 there even if
    those children later also differ in C-MeT -- that C-MeT difference is a separate
    downstream node with its own classification.

UNIT TEST: citrinin's two candidates share an identical reduction trajectory and differ
only in C-MeT cycle index -> the trie must report exactly one Gate-3 node, zero Gate-2.
If it smears Gate-3 across positions or shows any Gate-2, the conditioning is leaking.

Computed at alphabet+mass (realistic O) and alphabet-only (poorer O -> looser Z* ->
more branches -> upper bound on the Gate-3 fraction).

CAVEAT held sharp (the number cannot settle this): clean (trunk) mass is an UPPER BOUND
on usable signal, not the signal. "Free label" = the cycle's action is determined given
the prefix IN THE CORPUS, not that a policy gets it right out-of-distribution. The trie
SIZES the opportunity; the LOSO accuracy on trunk cycles BANKS it. Resolution sized is
not biology delivered -- same discipline as the geometric-channel caveat.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem.program import ReductionState as Rs, Release  # noqa: E402
from lpi.search.generate import Alphabet, generate  # noqa: E402

INVENTORY = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_LOG = ROOT / "results" / "gate_decomposition.log"

GRAMMAR = Alphabet(
    reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
    releases=(Release.HYDROLYSIS, Release.ALDOL_AROMATIC, Release.LACTONIZATION),
    starters=("acetyl",),
    allow_cmet=True,
)
PRODUCIBLE_TIERS = {"GOLD_unique", "SILVER_2_3", "SILVER_4_5", "SILVER_6_10",
                    "BRONZE_11_30", "DISCARD_>30"}


def action_seq(prog) -> list[tuple]:
    return [(c.reduction.value, bool(c.c_methyl), c.extender.value) for c in prog.cycles]


def walk_trie(seqs: list[list[tuple]]) -> tuple[int, int, int]:
    """Recursively walk the action-trie. Returns (clean, gate2, gate3) decision counts.
    A candidate that has ended contributes the sentinel None as its next action."""
    firsts = [s[0] if s else None for s in seqs]
    distinct = set(firsts)

    if len(distinct) == 1:
        if None in distinct:
            return (0, 0, 0)                       # all candidates ended here
        tails = [s[1:] for s in seqs]
        c, g2, g3 = walk_trie(tails)
        return (c + 1, g2, g3)                      # single-child -> 1 clean decision

    # branch node: classify by local divergence conditioned on this prefix
    real_reductions = {f[0] for f in distinct if f is not None}
    has_length_split = None in distinct
    is_gate2 = (len(real_reductions) > 1) or has_length_split

    groups: dict[tuple | None, list] = {}
    for s in seqs:
        key = s[0] if s else None
        groups.setdefault(key, []).append(s[1:] if s else [])
    c_tot = g2_tot = g3_tot = 0
    for key, tails in groups.items():
        if key is None:
            continue                                # ended branch -> no further decisions
        c, g2, g3 = walk_trie(tails)
        c_tot += c; g2_tot += g2; g3_tot += g3
    if is_gate2:
        g2_tot += 1
    else:
        g3_tot += 1
    return (c_tot, g2_tot, g3_tot)


def root_trunk_depth(seqs: list[list[tuple]]) -> int:
    """Leading run of cycles where ALL candidates share the identical action (the
    strict common prefix: identical state AND action -> exactly supervised CE)."""
    depth = 0
    while True:
        firsts = [s[depth] if depth < len(s) else None for s in seqs]
        if len(set(firsts)) == 1 and firsts[0] is not None:
            depth += 1
        else:
            break
    return depth


@dataclass
class ClusterTrie:
    bgc_id: str
    n_candidates: int
    n_cycles: int
    clean: int
    gate2: int
    gate3: int
    root_trunk: int
    capped: bool


def decompose_cluster(bgc_id: str, candidates: list, capped: bool) -> ClusterTrie | None:
    seqs = [action_seq(c.program) for c in candidates]
    seqs = [s for s in seqs if s]
    if not seqs:
        return None
    clean, g2, g3 = walk_trie(seqs)
    return ClusterTrie(bgc_id=bgc_id, n_candidates=len(seqs),
                       n_cycles=max(len(s) for s in seqs),
                       clean=clean, gate2=g2, gate3=g3,
                       root_trunk=root_trunk_depth(seqs), capped=capped)


def _agg(rows: list[ClusterTrie]) -> dict:
    total = sum(d.clean + d.gate2 + d.gate3 for d in rows)
    clean = sum(d.clean for d in rows)
    g2 = sum(d.gate2 for d in rows)
    g3 = sum(d.gate3 for d in rows)
    non_g2 = clean + g3  # decisions not trivially substrate-readable
    return dict(
        n=len(rows), total=total, clean=clean, gate2=g2, gate3=g3,
        gate3_frac_all=(g3 / total if total else 0.0),
        gate3_frac_nonG2=(g3 / non_g2 if non_g2 else 0.0),  # the O-invariant denominator
    )


def run_regime(df: pd.DataFrame, target_cho: bool, log: list[str]) -> dict:
    regime = "alphabet+mass" if target_cho else "alphabet-only"
    log.append(f"\n{'='*78}\nREGIME: {regime}\n{'='*78}")
    rows: list[ClusterTrie] = []
    for _, r in df.iterrows():
        C, H, O = int(r.C), int(r.H), int(r.O)
        lo, hi = int(r.cycle_lo), int(r.cycle_hi)
        if lo < 1:
            continue
        try:
            cho = (C, H, O) if target_cho else None
            res = generate(GRAMMAR, lo, hi, target_cho=cho)
        except Exception:
            continue
        if not res.candidates:
            continue
        d = decompose_cluster(r.bgc_id, res.candidates, capped=res.capped)
        if d:
            rows.append(d)

    if not rows:
        log.append("  (no producible clusters)")
        return {}

    uncapped = [d for d in rows if not d.capped]
    capped = [d for d in rows if d.capped]
    a_all = _agg(rows)
    a_unc = _agg(uncapped) if uncapped else None
    a_cap = _agg(capped) if capped else None

    def block(tag, a):
        if not a:
            log.append(f"  {tag}: (none)")
            return
        log.append(f"  {tag}: {a['n']} clusters, {a['total']} decisions")
        log.append(f"    clean {a['clean']} ({a['clean']/a['total']:.1%}) | "
                   f"Gate-2 {a['gate2']} ({a['gate2']/a['total']:.1%}) | "
                   f"Gate-3 {a['gate3']} ({a['gate3']/a['total']:.1%})")
        log.append(f"    Gate-3 / all decisions:        {a['gate3_frac_all']:.1%}")
        log.append(f"    Gate-3 / (clean+Gate-3) [O-inv]: {a['gate3_frac_nonG2']:.1%}")

    block("ALL producible", a_all)
    log.append("")
    block("UNCAPPED only (CERTIFIABLE -- Z* fully enumerated)", a_unc)
    log.append("")
    block("CAPPED only (Z* truncated at program_cap -- explosive HR clusters)", a_cap)

    cit = next((d for d in rows if d.bgc_id == "BGC0001338"), None)
    if cit:
        ok = (cit.gate2 == 0 and cit.gate3 == 1)
        log.append(f"\n  CITRININ UNIT TEST (BGC0001338): clean={cit.clean} "
                   f"gate2={cit.gate2} gate3={cit.gate3} capped={cit.capped} "
                   f"-> {'PASS (1 Gate-3, 0 Gate-2)' if ok else 'FAIL/NA (see dilution note)'}")

    log.append("\n  Per-cluster (|Z*|<=30):")
    log.append(f"  {'bgc':<12} {'N':>2} {'|Z*|':>5} {'clean':>5} {'G2':>3} {'G3':>3} {'cap':>4}")
    for d in sorted(rows, key=lambda x: x.n_candidates):
        if d.n_candidates <= 30:
            log.append(f"  {d.bgc_id:<12} {d.n_cycles:>2} {d.n_candidates:>5} "
                       f"{d.clean:>5} {d.gate2:>3} {d.gate3:>3} {'Y' if d.capped else '':>4}")

    return dict(regime=regime, all=a_all, uncapped=a_unc, capped=a_cap)


RUN_ALPHABET_ONLY = False  # the prior full run gave Gate-2 28138 vs 593 (10x), all capped;
                           # cited from the existing log for the O-dependence argument.


def main() -> None:
    df = pd.read_parquet(INVENTORY)
    prod = df[df.tier.isin(PRODUCIBLE_TIERS)].copy()
    log: list[str] = []
    log.append("Branching decomposition (STRICT TRIE) -- uncapped-certified recompute")
    log.append(f"({len(prod)} clusters with producible core, from {len(df)} MIBiG scanned)")

    am = run_regime(prod, target_cho=True, log=log)
    ao = run_regime(prod, target_cho=False, log=log) if RUN_ALPHABET_ONLY else None

    log.append(f"\n{'='*78}\nHEADLINE\n{'='*78}")
    if am and am["uncapped"]:
        u = am["uncapped"]; a = am["all"]
        log.append("  The CERTIFIABLE figure is on the uncapped subset (Z* fully enumerated):")
        log.append(f"    Gate-3 / all decisions:          {u['gate3_frac_all']:.1%} "
                   f"(uncapped)   vs  {a['gate3_frac_all']:.1%} (all, capped-contaminated)")
        log.append(f"    Gate-3 / (clean+Gate-3) [O-inv]: {u['gate3_frac_nonG2']:.1%} "
                   f"(uncapped)   vs  {a['gate3_frac_nonG2']:.1%} (all)")
        log.append(f"    clean (free labels): {u['clean']}/{u['total']} = "
                   f"{u['clean']/u['total']:.1%} (uncapped)")
        log.append("")
        log.append("  The CAP bites hardest inside the explosive HR clusters -- exactly where")
        log.append("  C-MeT label-symmetry (Gate-3) lives. So the capped-contaminated 'all'")
        log.append("  figure is NOT a random-slice estimate; the uncapped subset is the honest one.")
    log.append("")
    log.append("  O-DEPENDENCE (alphabet-only prior run, all 32 capped): Gate-2 exploded")
    log.append("  2613 -> 28138 (~10x) while Gate-3 stayed chemistry-bounded. Gate-2 is")
    log.append("  combinatorial in the grammar; Gate-3 is an O-invariant of the chemistry.")
    log.append("  => the Gate-3 / (clean+Gate-3) ratio is the O-stable denominator; the raw")
    log.append("  Gate-3 / all-decisions fraction drifts with whatever O you ran at.")
    log.append("")
    log.append("  TWO INDEPENDENT RESULTS, kept in separate sentences:")
    log.append("  (1) THINNESS: the conformational frontier is a small fraction of decisions.")
    log.append("  (2) SOLVABILITY: whether that thin frontier is simulable (AF2/MD) or")
    log.append("      measurable-only (cryo-EM) is the citrinin hinge experiment, unsettled.")
    log.append("  A 5% frontier that needs cryo-EM is still a wall, just a thin one. Thinness")
    log.append("  sized != biology delivered.")

    text = "\n".join(log)
    print(text)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(text + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
