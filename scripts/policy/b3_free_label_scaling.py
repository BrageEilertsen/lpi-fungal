"""B3: free-label scaling law -- does per-cycle DECISION count (not CLUSTER count) drive accuracy?

The weak-supervision (NLML) objective manufactures a per-cycle label for every cycle of every
BGC asset, straight from the executor's Z* (no manual annotation). So the data resource is the
number of DECISIONS (cycles), and each cluster (BGC) donates several for free. B3 measures the
held-out learning curve and reads it against BOTH axes:

  * vs CLUSTERS  (n training BGCs)   -- the naive axis
  * vs DECISIONS (n training cycles) -- the "free-label" axis the objective actually feeds on

Protocol (standard learning curve, leave-cluster-out -- ground rule 4):
  Shuffle the BGCs (seeded); reserve a FIXED held-out TEST set of clusters that is NEVER trained on.
  From the remaining train pool, draw R random size-k subsets for each k in K_GRID, train the
  POSITION (deployable) and SUBSTRATE (oracle) policies, and score per-cycle accuracy on the fixed
  test clusters. Report, per k: mean test accuracy over the R draws and the 2.5/97.5 percentile over
  draws (the training-subsample variance -- the uncertainty relevant to "does more data help"),
  plus the mean #training-decisions. A test-cluster bootstrap CI on the full-pool point is also given.

PRE-COMMITTED reading (verify-before-freeze):
  * SUPPORTED (free labels are a lever): mean test accuracy rises with training decisions and the
    largest-k draw-percentile-lo sits above the smallest-k draw-percentile-hi (curve still climbing).
  * REFUSED: the learning curve is FLAT within the draw spread across the full range -> at this
    feature set the ceiling is method/feature-bound, not label-bound.
  * INCONCLUSIVE: positive slope whose draw-percentiles overlap end-to-end (n=25 underpowered).
  Honest expectation on n=25 (max ~17 train clusters, ~8 test): the curve is short and noisy; a
  flat-within-spread result is a first-class negative, reported as such, NOT tuned away.

Single command: `make b3`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.policy.b1_substrate_wall import (  # noqa: E402
    CONDITIONS, apply_std, feat_tensor, label_tensor, train,
)
from scripts.policy.train_weak import build_corpus  # noqa: E402

OUT_LOG = ROOT / "results" / "b3_free_label_scaling.log"
N_TEST = 8                       # fixed held-out clusters (leave-cluster-out)
K_GRID = [3, 6, 9, 12, 15, 17]
R_DRAWS = 20
N_STEPS = 300
N_BOOT = 2000
SEED = 0xC0FFEE


def test_cycle_outcomes(policy, std, asset, names):
    feats = asset.candidates[asset.parsimony_best_idx]
    if not feats:
        return []
    with torch.no_grad():
        preds = policy(apply_std(feat_tensor(feats, names), std, names)).argmax(-1)
        labels = label_tensor(feats)
    return [bool(p == l) for p, l in zip(preds.tolist(), labels.tolist())]


def n_decisions(assets):
    return sum(len(a.candidates[a.parsimony_best_idx]) for a in assets)


def test_cluster_bootstrap_ci(per_test_outcomes, rng, n_boot=N_BOOT):
    n = len(per_test_outcomes)
    if n == 0:
        return (float("nan"), float("nan"))
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        flat = [b for j in idx for b in per_test_outcomes[j]]
        if flat:
            accs.append(sum(flat) / len(flat))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def run_condition(train_pool, test_set, names, rng):
    """Returns {k: dict(mean, p_lo, p_hi, mean_dec, boot_lo, boot_hi, test_n)}."""
    out = {}
    test_n = sum(len(a.candidates[a.parsimony_best_idx]) for a in test_set)
    for k in K_GRID:
        if k > len(train_pool):
            continue
        draw_accs = []
        dec_counts = []
        last_per_test = None
        for _ in range(R_DRAWS):
            sub_idx = rng.choice(len(train_pool), size=k, replace=False)
            tr = [train_pool[j] for j in sub_idx]
            dec_counts.append(n_decisions(tr))
            policy, std = train(tr, names, n_steps=N_STEPS)
            per_test = [test_cycle_outcomes(policy, std, a, names) for a in test_set]
            flat = [b for o in per_test for b in o]
            if flat:
                draw_accs.append(sum(flat) / len(flat))
            last_per_test = per_test
        mean = float(np.mean(draw_accs)) if draw_accs else float("nan")
        p_lo = float(np.percentile(draw_accs, 2.5)) if draw_accs else float("nan")
        p_hi = float(np.percentile(draw_accs, 97.5)) if draw_accs else float("nan")
        boot_lo, boot_hi = test_cluster_bootstrap_ci(last_per_test, rng)
        out[k] = dict(mean=mean, p_lo=p_lo, p_hi=p_hi, mean_dec=float(np.mean(dec_counts)),
                      boot_lo=boot_lo, boot_hi=boot_hi, test_n=test_n)
    return out


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    corpus = build_corpus()

    # fixed leave-cluster-out split
    order = rng.permutation(len(corpus))
    test_set = [corpus[i] for i in order[:N_TEST]]
    train_pool = [corpus[i] for i in order[N_TEST:]]

    L = []
    def emit(s=""):
        print(s, flush=True)
        L.append(s)

    emit("B3 -- free-label scaling law (decisions, not clusters)")
    emit("=" * 78)
    emit(f"corpus: {len(corpus)} BGCs, {n_decisions(corpus)} canonical decisions")
    emit(f"fixed held-out TEST: {len(test_set)} clusters ({n_decisions(test_set)} cycles); "
         f"train pool: {len(train_pool)} clusters ({n_decisions(train_pool)} cycles)")
    emit(f"sweep k={K_GRID}; R={R_DRAWS} draws/k; n_steps={N_STEPS}; seed={hex(SEED)}")
    emit(f"leave-cluster-out: the {len(test_set)} test clusters are NEVER in any training subset.")
    emit("")
    emit("PRE-COMMITTED: SUPPORTED if mean test acc rises with decisions and largest-k draw-pct-lo >")
    emit("smallest-k draw-pct-hi (climbing); REFUSED if flat within draw spread; INCONCLUSIVE if")
    emit("positive-but-overlapping. n=25 is short/noisy -> a flat result is a first-class negative.")
    emit("")

    summary = {}
    for cond in ("POSITION", "SUBSTRATE"):
        names = CONDITIONS[cond]
        emit(f"[{cond}] features={names}")
        emit(f"  {'k':>3} {'mean_dec':>9} {'test_acc':>9} {'draw[2.5,97.5]':>20} {'testboot[2.5,97.5]':>22}")
        res = run_condition(train_pool, test_set, names, rng)
        summary[cond] = res
        for k in sorted(res):
            d = res[k]
            emit(f"  {k:>3} {d['mean_dec']:>9.1f} {d['mean']:>9.4f}   "
                 f"[{d['p_lo']:.3f}, {d['p_hi']:.3f}]      [{d['boot_lo']:.3f}, {d['boot_hi']:.3f}]")
        ks = sorted(res)
        lo_pt, hi_pt = res[ks[0]], res[ks[-1]]
        delta = hi_pt["mean"] - lo_pt["mean"]
        climbing = hi_pt["p_lo"] > lo_pt["p_hi"]
        emit(f"  -> delta(mean test acc, k={ks[0]}->{ks[-1]}) = {delta:+.4f}; "
             f"climbing(draw-pct-separated)={climbing}")
        emit("")

    # verdict on POSITION (deployable)
    emit("-" * 78)
    pos = summary["POSITION"]; ks = sorted(pos)
    lo_pt, hi_pt = pos[ks[0]], pos[ks[-1]]
    delta = hi_pt["mean"] - lo_pt["mean"]
    climbing = hi_pt["p_lo"] > lo_pt["p_hi"]
    if climbing and delta > 0:
        verdict = ("SUPPORTED: held-out accuracy climbs draw-pct-separated with training decisions -> "
                   "free labels are a lever; the bridge is (partly) data-gated and recoverable by scale.")
    elif abs(delta) < 0.03 and not climbing:
        verdict = ("REFUSED: the learning curve is flat within the draw spread across the full range -> "
                   "at this feature set the ceiling is method/feature-bound, not label-bound.")
    else:
        verdict = (f"INCONCLUSIVE: slope delta={delta:+.4f} but draw spreads overlap end-to-end "
                   "(n=25 underpowered); free-label scaling is suggested, not demonstrated.")
    emit(f"VERDICT (POSITION, deployable): {verdict}")
    emit("Note: SUBSTRATE is an oracle; its curve bounds the formalism, POSITION the deployable lever.")

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(L) + "\n")
    emit(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
