"""Bacterial ESM demo (Direction A follow-up): does a sequence policy + verifier(mass) rerank recover
MORE whole programs than the domain cartoon, on bacterial modular PKS -- the family where a per-step
sequence EXISTS? This is the positive direction of the structural-wall story (the companion seed).

Task: recover each cluster's per-module INACTIVE-DOMAIN pattern -- did each present KR actually fire?
This is precisely the cartoon's blind spot: a present KR LOOKS active to gene content, but ~7% are
silent (KETO despite a KR domain). Recovering a whole cluster's program needs EVERY module right, so
per-module errors compound across the assembly line -- which is why per-module accuracy and
whole-program recovery diverge.

Three conditions, leave-cluster-out over 118 clusters / 1039 KR-bearing modules:
  (1) DOMAIN RULE  -- KR present => fired (all-active). The gene-content cartoon.
  (2) ESM policy   -- frozen ESM-2 + linear head on the KR sequence, argmax P(active)>0.5.
  (3) ESM -> VERIFIER(mass) rerank -- accurate mass pins the cluster's active count k (total reductions);
      label the top-k modules by ESM P(active), the rest inactive. The neuro-symbolic loop: learned
      proposer, sound count constraint.

SOUNDNESS: the ESM head is the unsound learned layer; the mass/count constraint is sound. The boundary
is typed and not smeared. Reported as-is, including null.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lpi.data.mibig import PROCESSED
from lpi.model.esm_head import _leave_cluster_out_folds, _train_linear, embed_sequences


def oof_p_active(df, X, groups):
    """Out-of-fold P(active) for every module (leave-cluster-out; every cluster is test once)."""
    ya = df["kr_active"].to_numpy()
    p = np.full(len(ya), np.nan)
    for tr, te in _leave_cluster_out_folds(groups):
        if te.sum() == 0:
            continue
        if len(np.unique(ya[tr])) < 2:       # no inactive in train -> cartoon fallback (all active)
            p[te] = 1.0
            continue
        n0, n1 = (ya[tr] == 0).sum(), (ya[tr] == 1).sum()
        cw = [len(ya[tr]) / (2 * max(n0, 1)), len(ya[tr]) / (2 * max(n1, 1))]
        _pred, proba = _train_linear(X[tr], ya[tr], X[te], 2, class_weight=cw)
        p[te] = proba[:, 1]
    return ya, p


def main():
    df = pd.read_parquet(PROCESSED / "clustercad_kr_examples.parquet")
    df = df[df["kr_sequence"].notna()].reset_index(drop=True)
    emb = embed_sequences(dict(zip(df["kr_domainid"], df["kr_sequence"])))
    df = df[df["kr_domainid"].isin(emb)].reset_index(drop=True)
    X = np.array([emb[d] for d in df["kr_domainid"]], dtype=np.float32)
    groups = df["cluster"].to_numpy()
    ya, p = oof_p_active(df, X, groups)

    clusters = pd.unique(df["cluster"])
    conds = ("domain", "esm", "rerank")
    permod_hit = {c: 0 for c in conds}
    prog_ok = {c: 0 for c in conds}
    n_mod = len(ya)
    inact_clusters = 0
    inact_rec = {c: 0 for c in conds}
    for cl in clusters:
        m = groups == cl
        yt, pp = ya[m], p[m]
        preds = {
            "domain": np.ones_like(yt),                              # KR present => fired
            "esm": (pp > 0.5).astype(int),                           # ESM argmax
            "rerank": np.zeros_like(yt),                             # ESM + mass-count
        }
        k = int(yt.sum())  # ORACLE active count (true #active). Mass + per-module domains CONSTRAIN
        if k > 0:           # this count but do NOT uniquely pin it -> this is an upper bound, not mass.
            preds["rerank"][np.argsort(-pp)[:k]] = 1
        for c in conds:
            permod_hit[c] += int((preds[c] == yt).sum())
            prog_ok[c] += int((preds[c] == yt).all())
        if (yt == 0).any():                                          # clusters with >=1 silent KR
            inact_clusters += 1
            for c in conds:
                inact_rec[c] += int((preds[c] == yt).all())

    N = len(clusters)
    print("Bacterial ESM demo -- sequence policy vs the domain cartoon (+ active-count oracle upper bound)")
    print(f"({N} clusters, {n_mod} KR-bearing modules; {int((ya==0).sum())} silent KRs; leave-cluster-out)\n")
    print(f"  {'condition':28s} {'per-module acc':>14s} {'whole-program recovery':>24s}")
    labels = {"domain": "(1) domain rule (cartoon)", "esm": "(2) ESM policy alone",
              "rerank": "(3) ESM + active-count ORACLE"}
    for c in conds:
        print(f"  {labels[c]:30s} {permod_hit[c]/n_mod:>12.3f} {prog_ok[c]:>17d}/{N} = {prog_ok[c]/N:.3f}")
    honest = prog_ok["esm"] - prog_ok["domain"]
    print(f"\n  HONEST headline -- ESM policy ALONE (no oracle, leave-cluster-out) lifts whole-program")
    print(f"  recovery {honest:+d}: {prog_ok['domain']}/{N} -> {prog_ok['esm']}/{N} "
          f"({prog_ok['domain']/N:.3f} -> {prog_ok['esm']/N:.3f}); on the {inact_clusters} silent-KR clusters")
    print(f"  (cartoon = {inact_rec['domain']}/{inact_clusters}) it recovers {inact_rec['esm']}/{inact_clusters} from SEQUENCE alone.")
    print(f"\n  UPPER BOUND -- adding an active-count constraint (here the TRUE count; an idealization of")
    print(f"  what accurate mass + per-module domains constrain but do NOT uniquely pin): "
          f"{prog_ok['rerank']}/{N} = {prog_ok['rerank']/N:.3f},")
    print(f"  {inact_rec['rerank']}/{inact_clusters} on silent-KR clusters -- headroom for the loop if an observable pins")
    print(f"  the count; NOT a mass result.")
    print("\n  Read: where a per-step SEQUENCE exists (bacterial/modular), the sequence policy recovers")
    print("  whole programs the gene-content cartoon provably cannot -- the positive direction of the")
    print("  structural-wall story. Caveat: leave-cluster-out (not homology-partitioned); the per-step")
    print("  signal survived homology-clean eval at 70% identity (0.85) in the head's own evaluation.")


if __name__ == "__main__":
    main()
