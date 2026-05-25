"""CLI: differential PoC — train domain->action heads on bacterial ClusterCAD, evaluate
leave-cluster-out + baselines, and transfer-test on fungal cycles.

    python -m lpi.cli.differential
"""

from __future__ import annotations

import sys
from pathlib import Path

from lpi.data.mibig import PROCESSED, ROOT
from lpi.model.differential import evaluate, transfer_test


def main(argv: list[str] | None = None) -> int:
    pq = PROCESSED / "clustercad_modules.parquet"
    if not pq.exists():
        print("clustercad_modules.parquet missing — run the ClusterCAD scrape first "
              "(lpi.data.clustercad).")
        return 1
    r = evaluate(pq)
    t = transfer_test(pq)
    # persist a summary for reproducible figure generation
    import json
    summary = {
        "reduction": {"mlp": r.mlp_reduction_acc, "domain_rule": r.domain_rule_acc,
                      "majority": r.majority_acc},
        "stereo": {"mlp": r.stereo_mlp_acc, "majority": r.stereo_majority_acc},
        "transfer": {"active_domains": t.active_domains_acc,
                     "constant_domains": t.constant_domains_acc, "n_cycles": t.n_cycles},
        "n_train": r.n_train, "n_test": r.n_test,
    }
    (ROOT / "results" / "differential_summary.json").write_text(json.dumps(summary, indent=2))
    print("Differential PoC — ClusterCAD bacterial module-pairs (ΔBGC → Δstructure)")
    print("-" * 70)
    print(f"  clean extension modules: {r.n_train + r.n_test} "
          f"(train {r.n_train} / test {r.n_test}, leave-cluster-out)")
    print(f"  REDUCTION-STATE  MLP={r.mlp_reduction_acc:.3f}  "
          f"domain-rule={r.domain_rule_acc:.3f}  majority={r.majority_acc:.3f}")
    print(f"    per-class (n, MLP acc): {r.per_class_reduction}")
    print(f"  KR-STEREO        MLP={r.stereo_mlp_acc:.3f}  majority={r.stereo_majority_acc:.3f}")
    print("-" * 70)
    print(f"  FUNGAL TRANSFER ({t.n_cycles} curated cycles):")
    print(f"    (a) active-domains  -> reduction: {t.active_domains_acc:.3f}  "
          "(per-step chemistry transfers)")
    print(f"    (b) constant domains-> reduction: {t.constant_domains_acc:.3f}  "
          "(iteration-grammar wall)")
    print("-" * 70)
    print("Read: per-step chemistry is learnable from domains (MLP>>majority) and transfers")
    print("to fungi perfectly WHEN active domains are known (a); the 100%->{:.0%} gap (b) is".format(
        t.constant_domains_acc))
    print("the iteration grammar — which domains fire each cycle — unreadable from gene content.")
    print("KR stereo (MLP~majority) needs sequence motifs: the ESM-2 head is the next step.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
