"""Track B end-to-end: reconstruct KR sequences -> fill parquet -> run the ESM-2 head.

Runs once scripts/fetch_kr_sources.py has cached the CDS FASTAs + the KR HMM. Answers the
paper's open question: does SEQUENCE carry the chemistry signal gene content/position lack?
(targets: inactive-KR detection, KR stereochemistry; vs domain-rule/majority baselines).
"""
from __future__ import annotations

from lpi.model.esm_data import build_kr_examples, write_examples
from lpi.model.esm_head import evaluate_head
from lpi.model.kr_reconstruct import reconstruct


def main():
    seqs, stats = reconstruct()
    print("reconstruct stats:", dict(stats))
    print("KR sequences recovered:", len(seqs))
    out = write_examples(build_kr_examples(), seqs)
    print("wrote", out)
    if len(seqs) < 30:
        print("too few sequences for a meaningful leave-cluster-out head; stopping honestly.")
        return
    print("\nESM-2 head (frozen embed + linear, leave-cluster-out):")
    for r in evaluate_head():
        print(f"  {r.target}: n={r.n}  head={r.head_metric:.3f}  auc={r.head_auc:.3f}  "
              f"majority={r.baseline_majority:.3f}  rule={r.baseline_domain_rule:.3f}")
        print(f"      ({r.metric_name})")


if __name__ == "__main__":
    main()
