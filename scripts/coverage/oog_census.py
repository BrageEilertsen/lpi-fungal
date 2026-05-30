"""Phase-0 census: classify the out-of-grammar fungal PKS cores by minimal missing-operator class.

Decision gate: is the executor's coverage gap CONCENTRATED (a few operator classes recover most
out-of-grammar cores -> a finite engineering grind, worth doing) or ASYMPTOTIC (a long flat tail of
bespoke transformations -> not a solo project)? And -- the sharper question -- of the concentrated
part, how much is SOUNDLY buildable versus entangled with transformations we cannot write soundly?

THREE SEGREGATED CURVES (blending them describes no actual roadmap):
  1. CHO-unreachable  -- the IN-SCOPE coverage curve: "add a PKS tailoring operator to this grammar"
                         (the aromatic-cyclization grind we can actually do). This is the headline.
  2. heteroatom       -- SCOPE BOUNDARY, not failure: non-CHO cores need NRPS-hybrid / halogenase /
                         heteroatom machinery -- a different grammar, possibly a different paper.
  3. too_large (C>20) -- a SCAN CAP, not a gap. Re-scanned with a lifted carbon cap FIRST; whatever
                         is still unreachable is folded into curve 1/2 by its actual missing class.
                         (The C>20 cap is the same kind of tractability artifact as the program_cap on
                         |Z*|: don't let it masquerade as a coverage gap.)

THE CURVE IS A STRUCTURAL-FEATURE PROXY AND AN UPPER BOUND ON REAL RECOVERY. A core counts as
proxy-recovered when ALL its tagged operator classes are covered -- which is NECESSARY, NOT SUFFICIENT:
the structural tag does not guarantee the new operator can be written *soundly*. The bound is loosest
exactly on the multi-class (entangled) cores -- so we emit the entanglement distribution alongside the
curve, because that, not the steepness, says whether "build the aromatic operators" clears a clean
majority or stalls on oxidative-rearrangement entanglement (the citrinin-tell logic: the structurally
hardest chemistry co-occurs with the operators we cannot write soundly).

Classes are COARSE on purpose (a decision gate needs a curve whose shape is robust to tagging choices,
not Phase-1 fusion-mode prioritization detail). Class assignment is anchored to the REAL grammar via
the beam's formula_feasible, not ad-hoc SMARTS:
  - formula-infeasible, feasible after removing k oxygens  -> oxidative_tailoring  (O-budget gap)
  - formula-infeasible otherwise                            -> other_formula       (H/C / skeleton gap)
  - formula-feasible but structurally unreachable           -> topology gap:
        carbocyclic aromatic ring  -> aromatic_cyclization
        non-aromatic carbocycle    -> non_aromatic_ring_rearrangement
        neither                    -> other_topology      (release / linear-topology gap)
  - prenyl/isoprenoid SMARTS       -> + prenylation         (co-occurs; least-robust tag, flagged)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem import mol as M  # noqa: E402
import lpi.search.reachability as RCH  # noqa: E402
from lpi.search.beam import OperatorSpec, formula_feasible, _formula_cho  # noqa: E402

PAIRS = ROOT / "data" / "processed" / "fungal_pks_pairs.parquet"
REACH = ROOT / "results" / "reachability.csv"
OUT_LOG = ROOT / "results" / "oog_census.log"
OUT_CSV = ROOT / "results" / "oog_census_per_core.csv"

SPEC = OperatorSpec(starters=("acetyl", "propionyl", "hexanoyl"))
LIFTED_CAP = 80
HALOGENS = {"F", "Cl", "Br", "I"}
# strict prenyl/isoprenoid: a dimethylallyl unit (CH3)2C=CH-CH2- -- specific enough to avoid
# tagging ordinary methylmalonyl-derived methyl branches (which over-tag a loose C=C(C)C SMARTS).
_PRENYL = [Chem.MolFromSmarts(s) for s in ("[CH3][CX3]([CH3])=[CH][CH2]", "[CH2]C=[CX3]([CH3])[CH3]")]


# ---------- structural detectors ----------
def _symbols(mol) -> set[str]:
    return {a.GetSymbol() for a in mol.GetAtoms()}


def _carbocyclic_aromatic(mol) -> bool:
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
               for i in ring):
            return True
    return False


def _has_nonaromatic_ring(mol) -> bool:
    """Any ring that is NOT a fully-aromatic carbocycle -- includes non-aromatic carbocycles
    AND O/N-heterocyclic rings (lactones, macrolactones, pyranones). The earlier carbocycle-only
    check missed lactone rings entirely, which mis-tagged aromatic+macrolactone cores (zearalenone,
    hypothemycin, the resorcylic-acid-lactones) as clean single-class aromatic."""
    ri = mol.GetRingInfo()
    for ring in ri.AtomRings():
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() and mol.GetAtomWithIdx(i).GetAtomicNum() == 6
                   for i in ring):
            return True
    return False


def _prenyl(mol) -> bool:
    return any(p is not None and mol.HasSubstructMatch(p) for p in _PRENYL)


def _oxidative_excess(C: int, H: int, O: int) -> bool:
    """Formula-infeasible, but feasible after stripping <=6 oxygens -> an O-budget (oxidation) gap."""
    for j in range(1, 7):
        if O - j >= 0 and formula_feasible((C, H, O - j), SPEC, max_cycles=C // 2 + 1):
            return True
    return False


def hetero_classes(mol) -> set[str]:
    s = _symbols(mol)
    out: set[str] = set()
    if "N" in s:
        out.add("N_incorporation")
    if s & HALOGENS:
        out.add("halogenation")
    if "S" in s:
        out.add("S_incorporation")
    rest = s - {"C", "H", "O", "N", "S"} - HALOGENS
    if rest:
        out.add("other_heteroatom")
    return out or {"other_heteroatom"}


def cho_classes(mol) -> set[str]:
    C, H, O = _formula_cho(mol)
    feasible = formula_feasible((C, H, O), SPEC, max_cycles=C // 2 + 1)
    cls: set[str] = set()
    if _carbocyclic_aromatic(mol):
        cls.add("aromatic_cyclization")
    if _has_nonaromatic_ring(mol):   # incl. lactone/macrolactone rings, REGARDLESS of aromaticity
        cls.add("non_aromatic_ring_rearrangement")
    if _prenyl(mol):
        cls.add("prenylation")
    if not feasible:
        cls.add("oxidative_tailoring" if _oxidative_excess(C, H, O) else "other_formula")
    return cls or {"other_topology"}


# ---------- greedy set-cover recovery curve ----------
def greedy_curve(core_sets: list[set[str]]) -> list[dict]:
    """A core is recovered only when ALL its classes are covered. At each step add the class that
    newly fully-recovers the most cores; if none is one-away, add the most-needed class (progress)."""
    remaining = [set(c) for c in core_sets]
    covered: set[str] = set()
    n_total = len(remaining)
    recovered = 0
    curve: list[dict] = []
    while remaining:
        one_away = Counter()
        for c in remaining:
            need = c - covered
            if len(need) == 1:
                one_away[next(iter(need))] += 1
        if one_away:
            pick, _ = one_away.most_common(1)[0]
        else:
            allneed = Counter(x for c in remaining for x in (c - covered))
            if not allneed:
                break
            pick, _ = allneed.most_common(1)[0]
        covered.add(pick)
        newly = [c for c in remaining if c <= covered]
        recovered += len(newly)
        remaining = [c for c in remaining if not c <= covered]
        curve.append(dict(op=pick, n_ops=len(covered), recovered=recovered,
                          frac=recovered / n_total if n_total else 0.0))
    return curve


def fmt_curve(title: str, curve: list[dict], n: int) -> list[str]:
    out = [f"  {title}  ({n} cores)"]
    out.append(f"    {'#ops':>4} {'+operator class':<32} {'cumulative recovered':>22}")
    for row in curve:
        out.append(f"    {row['n_ops']:>4} {row['op']:<32} "
                   f"{row['recovered']:>4}/{n}  ({row['frac']:.0%})")
    return out


def main() -> None:
    pairs = pd.read_parquet(PAIRS)[["bgc_id", "compound_name", "product_smiles_canonical"]]
    reach = pd.read_csv(REACH)[["bgc_id", "status"]]
    df = reach.merge(pairs, on="bgc_id", how="left")

    log: list[str] = ["Phase-0 out-of-grammar census: missing-operator classes + recovery curve",
                      f"({len(df)} fungal PKS in MIBiG 4.0; baseline statuses from reachability.csv)"]
    base = Counter(df.status)
    log.append("  baseline: " + ", ".join(f"{k}={v}" for k, v in base.most_common()))

    # ---- 1. re-scan too_large with a lifted cap (don't let the scan cap masquerade as a gap) ----
    RCH.CARBON_CAP = LIFTED_CAP
    big = df[df.status == "too_large"]
    log.append(f"\n{'='*74}\nRE-SCAN too_large (C>20) with lifted cap (C{LIFTED_CAP})\n{'='*74}")
    rescued = Counter()
    new_status: dict[str, str] = {}
    for _, r in big.iterrows():
        smi = r.product_smiles_canonical
        if not isinstance(smi, str) or not smi:
            new_status[r.bgc_id] = "parse_error"
            continue
        # Reduced budget: a REACHABLE core is found quickly; only unreachable-but-formula-feasible
        # large cores burn the full budget, and they return "unreachable"/"budget" either way and get
        # classified structurally below -- so a smaller cap loses almost nothing and avoids the tail.
        row = RCH.scan_target(r.bgc_id, str(r.compound_name), smi,
                              beam_width=2000, max_executions=8000)
        new_status[r.bgc_id] = row.status
        rescued[row.status] += 1
    log.append("  re-scan outcome: " + ", ".join(f"{k}={v}" for k, v in rescued.most_common()))
    log.append(f"  -> {rescued.get('reachable', 0)} were reachable-but-untested (scan-cap artifacts, "
               f"NOT coverage gaps, excluded); {rescued.get('budget', 0)} returned 'budget' "
               f"(reachability not exhaustively confirmed at the reduced re-scan budget -- classified "
               f"structurally below as a lower bound); the rest are confirmed unreachable. All "
               f"non-reachable cores fold into the curves by actual missing class.")
    df["status2"] = df.apply(lambda r: new_status.get(r.bgc_id, r.status), axis=1)

    # ---- 2. classify every non-reachable core ----
    cho_cores: list[tuple[str, str, set[str]]] = []   # (bgc, name, classes)
    het_cores: list[tuple[str, str, set[str]]] = []
    audit: list[dict] = []
    for _, r in df.iterrows():
        st = r.status2
        if st in ("reachable",):
            continue
        smi = r.product_smiles_canonical
        if not isinstance(smi, str) or not smi:
            continue
        try:
            mol = M.mol_from_smiles(smi)
        except Exception:  # noqa: BLE001
            continue
        # "budget" large cores aren't confirmed reachable; classify them structurally like the
        # unreachable ones (classification is structure/formula-based, not verdict-based), and flag
        # the count separately so their gap status is read as a lower bound.
        if _symbols(mol) - {"C", "H", "O"}:
            curve_grp, cls = "heteroatom", hetero_classes(mol)
            het_cores.append((r.bgc_id, str(r.compound_name), cls))
        else:
            curve_grp, cls = "CHO", cho_classes(mol)
            cho_cores.append((r.bgc_id, str(r.compound_name), cls))
        audit.append(dict(bgc_id=r.bgc_id, name=str(r.compound_name), baseline=r.status,
                          status_rescan=st, curve=curve_grp, classes="|".join(sorted(cls)),
                          n_classes=len(cls)))

    # ---- 3a. CHO in-scope curve (HEADLINE) ----
    log.append(f"\n{'='*74}\nCURVE 1 -- CHO-unreachable (IN-SCOPE: add a PKS tailoring operator)\n{'='*74}")
    log.append("  *** structural-feature PROXY: cores 'recovered' = all tagged classes covered.")
    log.append("  *** This is an UPPER BOUND on real recovery (tag != sound operator); loosest on")
    log.append("  *** the multi-class cores -- see the entanglement distribution below.")
    cho_sets = [c for _, _, c in cho_cores]
    log += fmt_curve("CHO recovery (greedy set-cover)", greedy_curve(cho_sets), len(cho_sets))
    log.append("  per-class core counts (a core may carry several -> sums exceed the total):")
    cc = Counter(x for c in cho_sets for x in c)
    for k, v in cc.most_common():
        log.append(f"    {k:<34} {v:>4}")

    # ---- 3b. entanglement distribution (the decisive number) ----
    log.append(f"\n{'='*74}\nENTANGLEMENT (CHO) -- how much of the recoverable-looking pile is clean\n{'='*74}")
    n = len(cho_sets)
    multi = [c for c in cho_sets if len(c) >= 2]
    hard = {"oxidative_tailoring", "non_aromatic_ring_rearrangement", "other_formula"}
    single_arom = [c for c in cho_sets if c == {"aromatic_cyclization"}]
    arom_entangled = [c for c in cho_sets if "aromatic_cyclization" in c and (c & hard)]
    touch_hard = [c for c in cho_sets if c & hard]
    dist = Counter(len(c) for c in cho_sets)
    log.append(f"  class-count distribution: " + ", ".join(f"{k}-class={v}" for k, v in sorted(dist.items())))
    if n:
        log.append(f"  need >1 operator class (entangled):            {len(multi)}/{n}  ({len(multi)/n:.0%})")
        log.append(f"  touch a hard/maybe-unsound class (ox/rearr):   {len(touch_hard)}/{n}  ({len(touch_hard)/n:.0%})")
        log.append(f"  CLEAN single-class aromatic (the doable pile): {len(single_arom)}/{n}  ({len(single_arom)/n:.0%})")
        log.append(f"  aromatic BUT entangled with a hard class:      {len(arom_entangled)}/{n}  ({len(arom_entangled)/n:.0%})")
    log.append("  => 'build the aromatic operators' soundly recovers AT MOST the clean single-class")
    log.append("     aromatic pile; the aromatic-but-entangled cores need a hard operator too and may")
    log.append("     not be soundly expressible -- the proxy curve overcounts recovery exactly there.")

    # ---- 3c. heteroatom scope-boundary curve ----
    log.append(f"\n{'='*74}\nCURVE 2 -- heteroatom (SCOPE BOUNDARY: not a PKS-grammar gap)\n{'='*74}")
    het_sets = [c for _, _, c in het_cores]
    log += fmt_curve("heteroatom recovery (greedy set-cover)", greedy_curve(het_sets), len(het_sets))
    hc = Counter(x for c in het_sets for x in c)
    for k, v in hc.most_common():
        log.append(f"    {k:<34} {v:>4}")
    log.append("  these need NRPS-hybrid / halogenase / heteroatom machinery -- a different grammar,")
    log.append("  reported as the scope boundary, not as a coverage failure.")

    # ---- verdict ----
    log.append(f"\n{'='*74}\nVERDICT\n{'='*74}")
    cho_curve = greedy_curve(cho_sets)
    if cho_curve:
        top3 = next((row["frac"] for row in cho_curve if row["n_ops"] == 3), cho_curve[-1]["frac"])
        log.append(f"  CHO coverage shape: top-3 operator classes proxy-cover {top3:.0%} of the {len(cho_sets)} "
                   f"in-scope cores (UPPER BOUND).")
    log.append(f"  Clean single-class-aromatic floor (soundly buildable lower bound): "
               f"{len(single_arom)}/{len(cho_sets)} = {len(single_arom)/len(cho_sets):.0%}" if cho_sets else "  (no CHO cores)")
    log.append("  Concentrated-vs-asymptotic is read off Curve 1's steepness; how much of the")
    log.append("  concentrated part is real is read off the clean-vs-entangled split. Both reported;")
    log.append("  the entanglement fraction -- not the steepness -- gates whether 'solve the PKS core'")
    log.append("  is a finite grind or stalls on oxidative-rearrangement entanglement.")

    text = "\n".join(log)
    print(text)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text(text + "\n")
    pd.DataFrame(audit).sort_values(["curve", "n_classes", "classes"]).to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)} and {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
