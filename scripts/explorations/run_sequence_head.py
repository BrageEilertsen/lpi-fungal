"""EXPLORATORY (Direction A, Stage 3): train + evaluate the ESM-2 sequence head on the bacterial KR set.

Now that sequences are joined (Stage 2), run the existing frozen-ESM-2 + linear head, leave-cluster-out,
on the two bacterial targets the domain cartoon provably cannot do: KR stereochemistry (R/S) and
inactive-domain detection (did the present KR fire). Compares to majority / domain-rule baselines.

SOUNDNESS: the sequence head is a LEARNED predictor with NO soundness guarantee -- the unsound predictive
layer that complements the sound executor. The executor remains the verifier; the head adds predictive
power over the sound type system. The boundary is typed and is not smeared here.
"""
from __future__ import annotations

import json
from pathlib import Path

from lpi.model.esm_head import evaluate_head

_OUT = Path("results/sequence_head_eval.json")


def main() -> None:
    print("Stage 3 -- ESM-2 head, bacterial KR (embeds 1039 seqs then trains leave-cluster-out)\n")
    results = evaluate_head()
    payload = []
    for r in results:
        print(f"{r.target}: n={r.n}  head={r.head_metric:.3f}  auc={r.head_auc:.3f}  "
              f"majority={r.baseline_majority:.3f}  domain_rule={r.baseline_domain_rule:.3f}  [{r.metric_name}]")
        payload.append(r.__dict__)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps({"bacterial": payload}, indent=2))
    print(f"\nwrote {_OUT}")
    print("\nReading: head > domain_rule on inactive-KR detection = sequence carries the 'did it fire'")
    print("signal gene content discards. head ~ majority on KR stereo = the previously-shown null, now")
    print("with sequence given its chance. Either way, reported as-is.")


if __name__ == "__main__":
    main()
