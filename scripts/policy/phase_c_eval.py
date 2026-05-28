"""Phase C5: POSE-augmented LOSO eval (the answer).

Extends scripts/explorations/substrate_state_probe.py with a POSE feature variant
fed from data/policy/descriptors.parquet. Reports LOSO accuracy across feature
sets, with a stratified bootstrap CI over synthase folds (decision 3 from Phase A).

Scopes reported (HONESTLY):
  full-30      all 9 HR+PR synthases (24 conformational + 6 substrate-only with
               NaN pose features -- substrate-only cycles can ONLY use substrate /
               position features; POSE variants only score on cycles that have it).
  conf-23      the 5 synthases with reliable ACP-Ser anchors (6-OH-mellein, 6-MSA,
               LovB, mellein, TenS). 23 cycles; the conformational hypothesis test.

Feature sets:
  majority             constant per-fold (the n=30 wall analogue at LOSO).
  POSITION             (t, frac, sub)
  SUBSTRATE_prefix     POSITION + (chain_c, oxid_sum, n_red, prev) -- substrate_state_probe.
  POSE                 POSITION + 5 pocket descriptors (contact_count, polar/hydrophobic,
                       sasa_buried_fraction, pocket_volume).
  POSE_plus_SUBSTRATE  union of POSE and SUBSTRATE_prefix.

The KEY comparison: POSE vs POSITION at the conf-23 scope. If POSE - POSITION's
95% bootstrap CI excludes zero, the conformational bridge is empirically reachable
with static pocket descriptors at n=23. If it straddles zero, n is underpowered
and the answer is "scale, then retry."
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.data.curated import load_all  # noqa: E402

INTER_PARQUET = ROOT / "data" / "policy" / "intermediates.parquet"
DESC_PARQUET = ROOT / "data" / "policy" / "descriptors.parquet"
RNG = np.random.default_rng(0xC0FFEE)
N_BOOT = 1000
K = 5  # k-NN for the eval; matches substrate_state_probe

LEVEL = {"keto": 0, "kr": 1, "dh": 2, "er": 3}
START_C = {"acetyl": 2, "propionyl": 3, "butyryl": 4, "hexanoyl": 6}

POSE_FEATS = [
    "pocket_contact_count_5A",
    "pocket_polar_contacts",
    "pocket_hydrophobic_contacts",
    "ligand_sasa_buried_fraction",
    "pocket_volume_A3",
]


def build_rows() -> pd.DataFrame:
    """Join intermediates + descriptors. Compute the substrate-prefix features inline
    (same definitions as substrate_state_probe.py)."""
    inter = pd.read_parquet(INTER_PARQUET)
    desc = pd.read_parquet(DESC_PARQUET)

    # Substrate-prefix features per (synthase, cycle): chain_c, oxid_sum, n_red, prev
    rows = []
    for e in load_all():
        if e.subclass not in ("HR", "PR"):
            continue
        sc = START_C.get(e.program.starter, 2)
        prior_levels: list[int] = []
        ncmet = 0
        for t, c in enumerate(e.program.cycles, start=1):
            lv = LEVEL[c.reduction.value]
            chain_c = sc + 2 * t + ncmet
            oxid_sum = sum(prior_levels)
            n_red = sum(1 for x in prior_levels if x > 0)
            prev = prior_levels[-1] if prior_levels else -1
            rows.append(dict(synthase=e.name, sub=1 if e.subclass == "HR" else 0,
                             y=lv, t=t, n_cycles=len(e.program.cycles), frac=t / len(e.program.cycles),
                             chain_c=chain_c, oxid_sum=oxid_sum, n_red=n_red, prev=prev))
            prior_levels.append(lv)
            ncmet += int(c.c_methyl)
    sub_df = pd.DataFrame(rows)

    # Join the conformational descriptors. cycle_t and synthase already shared.
    pose_cols = ["synthase", "cycle_t", "arm"] + POSE_FEATS
    pose_df = desc[pose_cols].rename(columns={"cycle_t": "t"})
    merged = sub_df.merge(pose_df, on=["synthase", "t"], how="left")
    return merged


def constant_loso(data: pd.DataFrame) -> float:
    """Majority-by-fold baseline (the LOSO analogue of the 0.567 wall)."""
    correct = 0
    for s in data.synthase.unique():
        tr, te = data[data.synthase != s], data[data.synthase == s]
        if len(tr) == 0 or len(te) == 0:
            continue
        vals, cnts = np.unique(tr.y, return_counts=True)
        maj = vals[np.argmax(cnts)]
        correct += int((te.y == maj).sum())
    return correct / len(data)


def knn_loso(data: pd.DataFrame, feats: list[str], k: int = K) -> float:
    """Leave-one-synthase-out k-NN accuracy. Z-score features per fold (train stats)."""
    correct = 0
    total = 0
    for s in data.synthase.unique():
        tr, te = data[data.synthase != s], data[data.synthase == s]
        if tr.empty or te.empty:
            continue
        Xtr = tr[feats].astype(float).to_numpy()
        Xte = te[feats].astype(float).to_numpy()
        # Drop test rows with any NaN feature (substrate-only cycles when scoring POSE)
        mask = ~np.isnan(Xte).any(axis=1)
        if mask.sum() == 0:
            continue
        Xte_clean = Xte[mask]
        ytr = tr.y.to_numpy()
        yte = te.y.to_numpy()[mask]
        # Same for train -- only use rows with full features
        tr_mask = ~np.isnan(Xtr).any(axis=1)
        if tr_mask.sum() < k:
            continue
        Xtr_clean = Xtr[tr_mask]
        ytr_clean = ytr[tr_mask]
        mu, sd = Xtr_clean.mean(0), Xtr_clean.std(0)
        sd[sd == 0] = 1.0
        Xtr_s = (Xtr_clean - mu) / sd
        Xte_s = (Xte_clean - mu) / sd
        for i in range(Xte_s.shape[0]):
            d = np.sqrt(((Xtr_s - Xte_s[i]) ** 2).sum(1))
            nn = ytr_clean[np.argsort(d)[:min(k, len(d))]]
            vals, cnts = np.unique(nn, return_counts=True)
            correct += int(vals[np.argmax(cnts)] == yte[i])
            total += 1
    return correct / total if total else float("nan")


def bootstrap_loso(data: pd.DataFrame, score_fn, n_boot: int = N_BOOT,
                   ) -> tuple[float, float, float]:
    """Stratified bootstrap by synthase fold -- resample synthases with replacement.

    Returns (mean, lo95, hi95) of the score across resamples.
    """
    synthases = list(data.synthase.unique())
    scores: list[float] = []
    for _ in range(n_boot):
        sample = RNG.choice(synthases, size=len(synthases), replace=True)
        # Build the bootstrap data by concatenating each sampled fold (with replacement
        # introduces duplicates -- intended; this samples the synthase distribution).
        # For LOSO inside the bootstrap, we hold out each sampled synthase ONCE; with
        # duplicates the held-out subset just gets evaluated more often.
        parts = [data[data.synthase == s].copy() for s in sample]
        # Rename duplicates so LOSO holds them out distinctly.
        seen: dict[str, int] = {}
        for i, s in enumerate(sample):
            seen[s] = seen.get(s, -1) + 1
            if seen[s] > 0:
                parts[i] = parts[i].copy()
                parts[i]["synthase"] = f"{s}__b{seen[s]}"
        boot_df = pd.concat(parts, ignore_index=True)
        score = score_fn(boot_df)
        if not np.isnan(score):
            scores.append(score)
    arr = np.array(scores)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    df = build_rows()
    print(f"Loaded {len(df)} HR+PR cycles across {df.synthase.nunique()} synthases.")
    print(f"  conformational (POSE features non-null): {df[POSE_FEATS[0]].notna().sum()}")
    print(f"  substrate-only (POSE NaN): {df[POSE_FEATS[0]].isna().sum()}")
    print()

    feature_sets = {
        "POSITION":            ["t", "frac", "sub"],
        "SUBSTRATE_prefix":    ["t", "frac", "sub", "chain_c", "oxid_sum", "n_red", "prev"],
        "POSE":                ["t", "frac", "sub", *POSE_FEATS],
        "POSE_plus_SUBSTRATE": ["t", "frac", "sub", "chain_c", "oxid_sum", "n_red", "prev",
                                *POSE_FEATS],
    }

    for scope_name, scope_df in [
        ("full-30 (all 9 HR+PR synthases)",  df),
        ("conf-23 (5 ACP-anchored synthases)", df[df[POSE_FEATS[0]].notna()]),
    ]:
        print(f"=== {scope_name}  (n={len(scope_df)}, k_synth={scope_df.synthase.nunique()}) ===")
        # majority baseline (single-pass; bootstrap separately for comparable CIs)
        maj = constant_loso(scope_df)
        maj_mean, maj_lo, maj_hi = bootstrap_loso(scope_df, constant_loso)
        print(f"  {'majority':<22} {maj:.3f}  [boot mean {maj_mean:.3f}, 95% CI {maj_lo:.3f}-{maj_hi:.3f}]")

        for fname, feats in feature_sets.items():
            try:
                point = knn_loso(scope_df, feats)
                if np.isnan(point):
                    print(f"  {fname:<22}  --skipped (no scorable cycles)")
                    continue
                mean, lo, hi = bootstrap_loso(scope_df, lambda d, f=feats: knn_loso(d, f))
                lift = point - maj
                print(f"  {fname:<22} {point:.3f}  [boot mean {mean:.3f}, 95% CI {lo:.3f}-{hi:.3f}]  "
                      f"lift vs majority {lift:+.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"  {fname:<22} ERROR: {exc!r}")
        print()


if __name__ == "__main__":
    main()
