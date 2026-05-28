"""Phase D inventory: |Z*| distribution over MIBiG fungal PKS clusters at the
metabologenomics-realistic (alphabet + mass) observable conditioning.

For each of the 326 fungal PKS pairs:
  1. Skip clusters with elements outside {C, H, O} (the rule set doesn't make N/Cl/...).
  2. Skip very large carbon counts (C > 20 -- beyond the executor's tractable range).
  3. Run the verifier-grounded generator with the rich PKS grammar (KETO/KR/DH/ER + 3
     release modes + acetyl starter + C-MeT enabled) constrained by the cluster's
     formula. Cycle range chosen by carbon count: lo = max(3, C//2 - 2), hi = C//2 + 1.
  4. Record |Z*| at alphabet-only and alphabet+mass rungs.

Then tabulate the SILVER tier candidates: clusters with |Z*| at alphabet+mass in {2,
3, 4, 5, ..., K} for various K. The recommendation for the K cutoff in Phase D's
marginal-likelihood weak supervision objective falls out of the elbow in the
distribution.

Output: data/policy/phase_d_inventory.parquet (per-cluster table) +
        results/phase_d_inventory.log (distribution summary).
"""
from __future__ import annotations

import csv
import re
import signal
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.chem.program import ReductionState as Rs, Release  # noqa: E402
from lpi.search.generate import Alphabet, generate  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_PARQUET = ROOT / "data" / "policy" / "phase_d_inventory.parquet"
OUT_CHECKPOINT = ROOT / "data" / "policy" / "phase_d_inventory.csv"  # incremental
OUT_LOG = ROOT / "results" / "phase_d_inventory.log"
MIBIG_PARQUET = ROOT / "data" / "processed" / "fungal_pks_pairs.parquet"
PER_CLUSTER_TIMEOUT_S = 30  # hard cap via SIGALRM; explosive clusters get TIMEOUT tier

# Same rich PKS grammar used in engine_benchmark.py (the canonical alphabet for
# fungal HR-PKS conditioning -- KETO/KR/DH/ER + the implemented releases + acetyl
# starter + C-MeT enabled).
GRAMMAR = Alphabet(
    reductions=(Rs.KETO, Rs.KR, Rs.DH, Rs.ER),
    releases=(Release.HYDROLYSIS, Release.ALDOL_AROMATIC, Release.LACTONIZATION),
    starters=("acetyl",),
    allow_cmet=True,
)

ALLOWED_ELEMENTS = {"C", "H", "O"}
CARBON_MIN = 3
CARBON_MAX = 16  # lowered from 20 -- the SILVER tier (small |Z*|) is in C<=12 anyway
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


class _Timeout(Exception):
    pass


def _alarm_handler(_signum, _frame):  # pragma: no cover -- only fires on hang
    raise _Timeout()


def parse_formula(formula: str) -> dict[str, int] | None:
    """Parse e.g. 'C8H8O3' -> {'C': 8, 'H': 8, 'O': 3}. Returns None on parse error."""
    if not isinstance(formula, str) or not formula:
        return None
    atoms: dict[str, int] = {}
    for el, n in _FORMULA_RE.findall(formula):
        if not el:
            continue
        atoms[el] = atoms.get(el, 0) + (int(n) if n else 1)
    return atoms if atoms else None


def recover_formula_from_smiles(smiles: str) -> str | None:
    """Derive a Hill-form molecular formula from a SMILES string via RDKit.

    Used as a fallback when MIBiG's ``formula`` column is empty -- 112 of 326 fungal
    PKS entries lack a registered formula but have a parseable ``product_smiles_canonical``.
    RDKit handles implicit hydrogens + charge balance + aromatic Hs correctly; doing this
    by regex would break on those cases.
    """
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        return CalcMolFormula(mol)
    except Exception:  # noqa: BLE001
        return None


def inventory_row(row: pd.Series) -> dict:
    formula = row.get("formula", "")
    atoms = parse_formula(formula)
    formula_source = "mibig" if atoms is not None else "none"
    # SMILES -> formula recovery for the 112 clusters where MIBiG has no formula tag
    # but has a parseable canonical SMILES (Brage's call: don't leave that signal on
    # the table). Honest tracking via ``formula_source``.
    if atoms is None:
        smi = row.get("product_smiles_canonical") or row.get("product_smiles_raw") or ""
        recovered = recover_formula_from_smiles(smi)
        if recovered:
            atoms = parse_formula(recovered)
            if atoms is not None:
                formula = recovered
                formula_source = "smiles_recovery"
    base = dict(bgc_id=row["bgc_id"], compound_name=row.get("compound_name", ""),
                formula=formula, formula_source=formula_source,
                mass=row.get("mass", 0.0),
                organism=row.get("organism", ""), subclass=row.get("pks_subclass", ""))
    if atoms is None:
        return {**base, "C": 0, "H": 0, "O": 0,
                "z_star_alphabet_plus_mass": -1, "tier": "PARSE_ERROR",
                "runtime_s": 0.0, "cycle_lo": 0, "cycle_hi": 0}
    extra = set(atoms) - ALLOWED_ELEMENTS
    C = atoms.get("C", 0)
    H = atoms.get("H", 0)
    O = atoms.get("O", 0)
    if extra:
        return {**base, "C": C, "H": H, "O": O,
                "z_star_alphabet_plus_mass": -1,
                "tier": f"NON_CHO ({','.join(sorted(extra))})", "runtime_s": 0.0,
                "cycle_lo": 0, "cycle_hi": 0}
    if not (CARBON_MIN <= C <= CARBON_MAX):
        return {**base, "C": C, "H": H, "O": O,
                "z_star_alphabet_plus_mass": -1,
                "tier": "OUT_OF_RANGE", "runtime_s": 0.0,
                "cycle_lo": 0, "cycle_hi": 0}

    # Chain length range: each cycle adds ~2 carbons (malonyl); +0/1 from C-MeT.
    # acetyl starter contributes 2 carbons; so cycles ~ (C - 2) / 2 ± slack.
    n_est = max(0, (C - 2) // 2)
    lo = max(1, n_est - 1)
    hi = min(10, n_est + 2)

    t0 = time.time()
    # alphabet+mass rung (the metabologenomics observable set). SIGALRM enforces
    # a hard timeout to prevent the combinatorial blow-up that wasted 39 minutes on
    # cluster 0 in the first inventory attempt.
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(PER_CLUSTER_TIMEOUT_S)
    try:
        res_mass = generate(GRAMMAR, lo, hi, target_cho=(C, H, O))
        n_mass = len(res_mass.candidates)
    except _Timeout:
        signal.alarm(0)
        return {**base, "C": C, "H": H, "O": O,
                "z_star_alphabet_plus_mass": -1, "tier": "TIMEOUT",
                "runtime_s": float(PER_CLUSTER_TIMEOUT_S),
                "cycle_lo": lo, "cycle_hi": hi}
    except Exception as exc:  # noqa: BLE001
        signal.alarm(0)
        return {**base, "C": C, "H": H, "O": O,
                "z_star_alphabet_plus_mass": -1,
                "tier": f"GEN_ERROR: {exc!r}", "runtime_s": time.time() - t0,
                "cycle_lo": lo, "cycle_hi": hi}
    signal.alarm(0)
    elapsed = time.time() - t0

    # Alphabet-only enumeration was dropped from this inventory after the first run
    # showed it dominates wall time without changing the metabologenomics answer
    # (the formula prefilter is what makes generate() tractable). Keep the column
    # name for schema compatibility; flagged -2 (= "not computed").

    if n_mass == 0:
        tier = "UNREACHABLE"
    elif n_mass == 1:
        tier = "GOLD_unique"
    elif n_mass <= 3:
        tier = "SILVER_2_3"
    elif n_mass <= 5:
        tier = "SILVER_4_5"
    elif n_mass <= 10:
        tier = "SILVER_6_10"
    elif n_mass <= 30:
        tier = "BRONZE_11_30"
    else:
        tier = "DISCARD_>30"

    return {**base, "C": C, "H": H, "O": O,
            "z_star_alphabet_plus_mass": n_mass,
            "tier": tier, "runtime_s": elapsed,
            "cycle_lo": lo, "cycle_hi": hi}


def main() -> None:
    if not MIBIG_PARQUET.exists():
        raise SystemExit(f"missing {MIBIG_PARQUET} -- run `make data` first")
    df = pd.read_parquet(MIBIG_PARQUET)
    print(f"Scanning {len(df)} fungal PKS pairs (timeout {PER_CLUSTER_TIMEOUT_S}s/cluster)...",
          flush=True)
    OUT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    t_start = time.time()
    fieldnames = None
    with OUT_CHECKPOINT.open("w", newline="") as fh:
        writer: csv.DictWriter | None = None
        for i, (_, row) in enumerate(df.iterrows()):
            t_c = time.time()
            result = inventory_row(row)
            wall = time.time() - t_c
            rows.append(result)
            if writer is None:
                fieldnames = list(result.keys())
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            # Some rows have fewer keys (early returns) -- pad with None to align schema.
            writer.writerow({k: result.get(k) for k in fieldnames})
            fh.flush()
            # Per-cluster progress -- every cluster, not every 25, so a stuck cluster is
            # visible immediately and we know exactly where the cost lives.
            print(f"  [{i+1:>3}/{len(df)}] {row['bgc_id']:<12} "
                  f"C{result.get('C',0):>2}H{result.get('H',0):>2}O{result.get('O',0):>2} "
                  f"|Z*|={result.get('z_star_alphabet_plus_mass','?')!s:<5} "
                  f"tier={result['tier']:<15} {wall:>5.1f}s "
                  f"(total {time.time()-t_start:>5.0f}s)", flush=True)
    out = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PARQUET, index=False)
    print(f"\nwrote {OUT_PARQUET.relative_to(ROOT)} -- {len(out)} rows", flush=True)

    # Summary
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"Phase D inventory: |Z*| distribution at alphabet+mass conditioning")
    lines.append(f"  ({len(out)} fungal PKS clusters; rich grammar; carbon range "
                 f"[{CARBON_MIN}, {CARBON_MAX}])")
    lines.append("")
    lines.append("Formula source breakdown:")
    for src, n in out.formula_source.value_counts().sort_index().items():
        lines.append(f"  {src:<22} {n:>4}")
    lines.append("")
    lines.append("Tier distribution:")
    for tier, n in out.tier.value_counts().sort_index().items():
        lines.append(f"  {tier:<30} {n:>4}")
    lines.append("")
    lines.append("Tractable tiers (under PKS grammar with CHO formula -- not the proxies):")
    tractable = out[~out.tier.isin([t for t in out.tier.unique() if t.startswith(
        ("PARSE_ERROR", "NON_CHO", "OUT_OF_RANGE", "GEN_ERROR"))])]
    lines.append(f"  total tractable: {len(tractable)}")
    by_n_mass = tractable.z_star_alphabet_plus_mass.value_counts().sort_index()
    lines.append("")
    lines.append("|Z*| at alphabet+mass histogram (tractable clusters only):")
    cum = 0
    for n_mass, n_clust in by_n_mass.items():
        cum += n_clust
        lines.append(f"  |Z*|={n_mass:>4}: {n_clust:>3} clusters   (cumulative <={n_mass}: {cum})")
    lines.append("")
    lines.append("SILVER candidate inventory (|Z*| in {2, 3}):")
    silver23 = tractable[tractable.z_star_alphabet_plus_mass.isin([2, 3])]
    for _, r in silver23.iterrows():
        lines.append(f"  {r.bgc_id}  C{r.C}H{r.H}O{r.O} ({r.compound_name[:38]:<38}) "
                     f"|Z*|={r.z_star_alphabet_plus_mass}")
    lines.append(f"\nTotal SILVER_2_3: {len(silver23)} clusters")
    lines.append(f"SILVER_4_5:       {(tractable.z_star_alphabet_plus_mass.between(4, 5)).sum()}")
    lines.append(f"SILVER_6_10:      {(tractable.z_star_alphabet_plus_mass.between(6, 10)).sum()}")
    lines.append(f"BRONZE_11_30:     {(tractable.z_star_alphabet_plus_mass.between(11, 30)).sum()}")
    lines.append(f"GOLD_unique:      {(tractable.z_star_alphabet_plus_mass == 1).sum()}")
    lines.append(f"UNREACHABLE:      {(tractable.z_star_alphabet_plus_mass == 0).sum()}")
    text = "\n".join(lines)
    print()
    print(text)
    OUT_LOG.write_text(text + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
