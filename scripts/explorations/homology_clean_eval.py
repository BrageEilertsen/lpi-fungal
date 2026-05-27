"""EXPLORATORY (Direction A, Stage 3b): homology-partitioned re-evaluation of the ESM-2 head.

Leave-cluster-out by BGC cluster (Stage 3) does NOT control sequence homology across clusters, so the
0.860/0.849 are upper bounds. Here we re-evaluate with folds that are disjoint by SEQUENCE-IDENTITY
cluster, so no homolog sits in both train and test. Standard protein-ML practice.

Method note (honest): MMseqs2/CD-HIT are not available here, so clustering is CD-HIT-style GREEDY
clustering on the KR-domain sequences using real %identity from a BLOSUM62 global alignment
(Biopython PairwiseAligner); identity = identical_aligned_positions / length-of-shorter-sequence. We run
two thresholds and report both, 30% (strict: generalize across the family) as the headline, 70%
(generous: remove near-duplicates). Embeddings are reused from the Stage-3 cache (no re-embedding).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices

from lpi.model.esm_head import (_balanced_acc, _leave_cluster_out_folds, _roc_auc, _train_linear,
                                embed_sequences)

_aligner = PairwiseAligner()
_aligner.mode = "global"
_aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
_aligner.open_gap_score = -11.0
_aligner.extend_gap_score = -1.0


def pct_identity(a: str, b: str) -> float:
    aln = _aligner.align(a, b)[0]
    ids = aln.counts().identities
    return ids / min(len(a), len(b))


def greedy_clusters(seqs: list[str], threshold: float) -> list[int]:
    """CD-HIT-style: longest-first; assign to the first representative within `threshold` identity, else
    open a new cluster. Returns a cluster id per input sequence."""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    reps: list[int] = []           # representative indices
    cluster_of = [-1] * len(seqs)
    for i in order:
        for cid, rep in enumerate(reps):
            if pct_identity(seqs[i], seqs[rep]) >= threshold:
                cluster_of[i] = cid
                break
        else:
            cluster_of[i] = len(reps)
            reps.append(i)
    return cluster_of


def eval_with_groups(X, y, groups, binary_auc=False):
    """Leave-cluster-out balanced accuracy (+AUC if binary) over homology-cluster folds."""
    n_clusters = len(set(groups))
    if n_clusters < 2:
        return None, None, n_clusters
    bals, aucs = [], []
    for tr, te in _leave_cluster_out_folds(groups, n_folds=min(5, n_clusters)):
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        n0, n1 = (y[tr] == 0).sum(), (y[tr] == 1).sum()
        cw = [len(y[tr]) / (2 * max(n0, 1)), len(y[tr]) / (2 * max(n1, 1))]
        pred, proba = _train_linear(X[tr], y[tr], X[te], 2, class_weight=cw)
        bals.append(_balanced_acc(y[te], pred))
        if binary_auc and len(np.unique(y[te])) == 2:
            aucs.append(_roc_auc(y[te], proba[:, 1]))
    return (float(np.mean(bals)) if bals else float("nan"),
            float(np.mean(aucs)) if aucs else float("nan"), n_clusters)


def main() -> None:
    df = pd.read_parquet("data/processed/clustercad_kr_examples.parquet")
    df = df[df["kr_sequence"].notna()].reset_index(drop=True)
    emb = embed_sequences(dict(zip(df["kr_domainid"], df["kr_sequence"])))  # cached from Stage 3
    df = df[df["kr_domainid"].isin(emb)].reset_index(drop=True)
    X = np.array([emb[d] for d in df["kr_domainid"]], dtype=np.float32)
    seqs = df["kr_sequence"].tolist()
    print(f"n={len(df)} sequences; BGC clusters={df['cluster'].nunique()}\n")

    for T in (0.70, 0.30):
        t0 = time.time()
        hclust = np.array(greedy_clusters(seqs, T))
        nC = len(set(hclust))
        print(f"== identity threshold {int(T*100)}%: {nC} homology clusters "
              f"(clustering {time.time()-t0:.0f}s) ==")

        ya = df["kr_active"].to_numpy()
        bal, auc, _ = eval_with_groups(X, ya, hclust, binary_auc=True)
        maj_a = max((ya == 0).mean(), (ya == 1).mean())
        print(f"  inactive-KR detection: head bal-acc={bal}  AUC={auc}  (majority/rule=0.50)")

        st = df[df["stereo"].isin(["R", "S"])].reset_index(drop=True)
        Xs = np.array([emb[d] for d in st["kr_domainid"]], dtype=np.float32)
        ys = (st["stereo"] == "S").astype(int).to_numpy()
        hs = np.array(greedy_clusters(st["kr_sequence"].tolist(), T))
        bal_s, _a, nCs = eval_with_groups(Xs, ys, hs)
        maj_s = max(ys.mean(), 1 - ys.mean())
        print(f"  KR stereo (R/S): head acc={bal_s}  ({nCs} clusters; majority={maj_s:.3f})\n")


if __name__ == "__main__":
    main()
