"""B1 (decisive): does conditioning the per-cycle policy on substrate state s_t clear the 0.567 wall?

Faithful (a)-vs-(b) harness. SAME corpus, SAME marginal-likelihood (NLML) loss, SAME
leave-one-BGC-out protocol as scripts/policy/train_weak.py; the ONLY variable changed is the
feature set, so any accuracy lift is attributable to s_t and not to a method difference.

Three NESTED conditions (each strictly adds information to the one above):

  CONSTANT  : intercept only                         -> learns the label marginal; under LOSO this
                                                        is the per-synthase-majority predictor =
                                                        the LOCAL ANALOGUE of the 0.567 cross-taxa wall.
  POSITION  : [t, frac, sub_hr]                       -> cheap, genuinely OBSERVABLE state (you know the
                                                        cycle index and whether the synthase is HR/PR).
                                                        This is the DEPLOYABLE condition.
  SUBSTRATE : POSITION + [chain_c, oxid_sum, n_red, prev] -> + running substrate state. oxid_sum/n_red/
                                                        prev are computed from PRIOR ACTIONS in the same
                                                        program, so this is an ORACLE: at inference on a
                                                        held-out synthase you would not have them. It tests
                                                        the FORMALISM (is a_t a function of s_t that
                                                        generalizes across synthases?), NOT a deployable
                                                        predictor.

Held-out: leave-one-BGC-out (n_BGC folds). Uncertainty: CLUSTER bootstrap that resamples whole
held-out BGCs with replacement (respects the LOSO clustering; resampling cycles would understate
the CI). Per-subclass HR/PR breakdown (curated-only split is exact; inventory subclass tags are a
family heuristic and flagged as such).

PRE-COMMITTED reading (verify-before-freeze -- fixed BEFORE the run):
  * SUPPORTED  iff SUBSTRATE clears 0.567 (bootstrap CI lower bound > 0.567) AND beats POSITION with
               CI-separated margin.
  * REFUSED    if SUBSTRATE does not clear 0.567 (the wall stands), OR if SUBSTRATE <= POSITION (the
               extra substrate features do not help -> leans toward the policy being enzyme-specific,
               not substrate-universal -- the probe's standing null).
  * INCONCLUSIVE if SUBSTRATE beats POSITION but its CI straddles 0.567 (signal present, underpowered).
  * Because SUBSTRATE is an ORACLE, even SUPPORTED here would be a statement about the FORMALISM; the
    deployable wall-clearance comparison is CONSTANT vs POSITION.

Single command: `make b1`  (or: PYTHONPATH=src .venv/bin/python scripts/policy/b1_substrate_wall.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.policy.train_weak import (  # noqa: E402
    BGCAsset, CycleFeat, N_ACTIONS, REDUCTION_LABELS, build_corpus,
)

OUT_LOG = ROOT / "results" / "b1_substrate_wall.log"
WALL = 0.567  # constant-domain cross-taxa LOSO baseline (differential_summary.json)
N_BOOT = 2000
SEED = 0xC0FFEE

CONDITIONS = {
    "CONSTANT":  [],
    "POSITION":  ["t", "frac", "sub_hr"],
    "SUBSTRATE": ["t", "frac", "sub_hr", "chain_c", "oxid_sum", "n_red", "prev"],
}


# ---- feature tensors parametrized by an explicit feature list --------------------

def feat_tensor(feats: list[CycleFeat], names: list[str]) -> torch.Tensor:
    if not names:  # CONSTANT: a single constant input -> pure bias / label marginal
        return torch.ones((len(feats), 1), dtype=torch.float32)
    return torch.tensor([[getattr(f, n) for n in names] for f in feats], dtype=torch.float32)


def label_tensor(feats: list[CycleFeat]) -> torch.Tensor:
    return torch.tensor([f.label for f in feats], dtype=torch.long)


class CfgPolicy(nn.Module):
    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---- standardization (fit on train cycles only -> no leakage) --------------------

def fit_standardizer(assets: list[BGCAsset], names: list[str]):
    if not names:
        return None
    X = np.array([[getattr(f, n) for n in names]
                  for a in assets for f in a.candidates[a.parsimony_best_idx]], dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    return mu, sd


def apply_std(t: torch.Tensor, std, names: list[str]) -> torch.Tensor:
    if std is None or not names:
        return t
    mu, sd = std
    return (t - torch.tensor(mu, dtype=torch.float32)) / torch.tensor(sd, dtype=torch.float32)


# ---- NLML loss (identical objective to train_weak, parametrized feature set) -----

def candidate_logprob(policy, feats, names, std) -> torch.Tensor:
    if not feats:
        return torch.tensor(0.0)
    X = apply_std(feat_tensor(feats, names), std, names)
    y = label_tensor(feats)
    logp = torch.log_softmax(policy(X), dim=-1)
    return logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum()


def nlml_loss(policy, assets, names, std) -> torch.Tensor:
    terms = []
    for a in assets:
        lp = torch.stack([candidate_logprob(policy, c, names, std) for c in a.candidates])
        terms.append(-torch.logsumexp(lp, dim=0))
    return torch.stack(terms).sum()


def train(assets: list[BGCAsset], names: list[str], n_steps=600, lr=0.05):
    std = fit_standardizer(assets, names)
    n_in = max(1, len(names))
    policy = CfgPolicy(n_in)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(n_steps):
        opt.zero_grad()
        nlml_loss(policy, assets, names, std).backward()
        opt.step()
    return policy, std


# ---- LOSO returning per-cycle records (for bootstrap + subclass split) -----------

def loso_records(assets: list[BGCAsset], names: list[str]):
    """Returns per-held-BGC list of dicts: {bgc, sub_hr, canon: [bool...], agreed: [bool...]}."""
    out = []
    for i, held in enumerate(assets):
        train_set = [a for j, a in enumerate(assets) if j != i]
        policy, std = train(train_set, names)
        feats = held.candidates[held.parsimony_best_idx]
        canon = []
        if feats:
            with torch.no_grad():
                preds = policy(apply_std(feat_tensor(feats, names), std, names)).argmax(-1)
                labels = label_tensor(feats)
                canon = [bool(p == l) for p, l in zip(preds.tolist(), labels.tolist())]
        # agreed-cycles only (all candidates share the label at t)
        agreed = []
        if held.n_z_star >= 1 and held.candidates[0]:
            T = min(len(c) for c in held.candidates)
            for t in range(T):
                if len({c[t].label for c in held.candidates}) == 1:
                    ft = held.candidates[held.parsimony_best_idx][t]
                    with torch.no_grad():
                        pred = int(policy(apply_std(feat_tensor([ft], names), std, names)).argmax(-1).item())
                    agreed.append(bool(pred == ft.label))
        sub_hr = feats[0].sub_hr if feats else None
        out.append(dict(bgc=held.bgc, sub_hr=sub_hr, canon=canon, agreed=agreed))
    return out


def pooled_acc(records, key) -> tuple[float, int]:
    flat = [b for r in records for b in r[key]]
    return (sum(flat) / len(flat) if flat else float("nan")), len(flat)


def cluster_bootstrap_ci(records, key, rng, n_boot=N_BOOT):
    """Resample whole held-out BGCs with replacement; pool their cycle outcomes."""
    n = len(records)
    accs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        flat = [b for j in idx for b in records[j][key]]
        if flat:
            accs.append(sum(flat) / len(flat))
    if not accs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5)))


def subclass_acc(records, key, sub_hr_val, only_curated=False):
    flat = []
    for r in records:
        if r["sub_hr"] != sub_hr_val:
            continue
        if only_curated and not r["bgc"].startswith("curated:"):
            continue
        flat.extend(r[key])
    return (sum(flat) / len(flat) if flat else float("nan")), len(flat)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    corpus = build_corpus()
    n_gold = sum(1 for a in corpus if a.n_z_star == 1)
    n_silver = sum(1 for a in corpus if a.n_z_star > 1)

    L = []
    def emit(s=""):
        print(s, flush=True)
        L.append(s)

    emit("B1 -- substrate-state conditioning vs the 0.567 wall (faithful (a)-vs-(b) harness)")
    emit("=" * 78)
    emit(f"corpus: {len(corpus)} BGC assets (GOLD |Z*|=1: {n_gold}, SILVER |Z*|>1: {n_silver})")
    emit(f"loss: NLML (identical to train_weak); eval: leave-one-BGC-out ({len(corpus)} folds)")
    emit(f"CI: cluster bootstrap over held-out BGCs, {N_BOOT} reps, 95% percentile; seed={hex(SEED)}")
    emit(f"wall (constant-domain cross-taxa LOSO baseline): {WALL}")
    emit("")
    emit("PRE-COMMITTED reading: SUPPORTED iff SUBSTRATE CI-lower > 0.567 AND SUBSTRATE > POSITION")
    emit("(CI-separated); REFUSED if SUBSTRATE <= POSITION or fails to clear 0.567; SUBSTRATE is an")
    emit("ORACLE (prior actions known) so it tests the FORMALISM, not a deployable predictor.")
    emit("")

    results = {}
    for name, feats in CONDITIONS.items():
        recs = loso_records(corpus, feats)
        ac, n_c = pooled_acc(recs, "canon")
        aa, n_a = pooled_acc(recs, "agreed")
        ci_c = cluster_bootstrap_ci(recs, "canon", rng)
        ci_a = cluster_bootstrap_ci(recs, "agreed", rng)
        results[name] = dict(recs=recs, ac=ac, aa=aa, ci_c=ci_c, ci_a=ci_a, n_c=n_c, n_a=n_a)
        emit(f"[{name:<9}] features={feats if feats else '(intercept only)'}")
        emit(f"            canonical acc = {ac:.4f}  95%CI [{ci_c[0]:.4f}, {ci_c[1]:.4f}]  (n={n_c} cycles)")
        emit(f"            agreed    acc = {aa:.4f}  95%CI [{ci_a[0]:.4f}, {ci_a[1]:.4f}]  (n={n_a} cycles)")
        emit("")

    # ---- the decisive contrasts ----
    sub, pos, con = results["SUBSTRATE"], results["POSITION"], results["CONSTANT"]
    emit("-" * 78)
    emit("DECISIVE CONTRASTS (canonical per-cycle accuracy):")
    emit(f"  SUBSTRATE - POSITION lift = {sub['ac'] - pos['ac']:+.4f}   "
         f"(SUBSTRATE {sub['ac']:.4f} vs POSITION {pos['ac']:.4f})")
    emit(f"  POSITION  - CONSTANT lift = {pos['ac'] - con['ac']:+.4f}   "
         f"(POSITION {pos['ac']:.4f} vs CONSTANT {con['ac']:.4f})")
    emit(f"  SUBSTRATE clears wall {WALL}? CI-lower {sub['ci_c'][0]:.4f} > {WALL}: "
         f"{sub['ci_c'][0] > WALL}")
    emit("")

    # ---- per-subclass (HR vs PR) breakdown ----
    emit("PER-SUBCLASS canonical accuracy (HR=1, PR=0):")
    for name in CONDITIONS:
        recs = results[name]["recs"]
        hr_all = subclass_acc(recs, "canon", 1)
        pr_all = subclass_acc(recs, "canon", 0)
        hr_cur = subclass_acc(recs, "canon", 1, only_curated=True)
        pr_cur = subclass_acc(recs, "canon", 0, only_curated=True)
        emit(f"  [{name:<9}] all: HR={hr_all[0]:.3f}(n={hr_all[1]}) PR={pr_all[0]:.3f}(n={pr_all[1]})  "
             f"| curated-only(exact): HR={hr_cur[0]:.3f}(n={hr_cur[1]}) PR={pr_cur[0]:.3f}(n={pr_cur[1]})")
    emit("  (inventory subclass tags are a family heuristic; the curated-only split is exact.)")
    emit("")

    # ---- verdict (mechanical from the pre-committed rule) ----
    cleared = sub["ci_c"][0] > WALL
    beats_pos = sub["ci_c"][0] > pos["ci_c"][1]  # CI-separated above POSITION
    sub_le_pos = sub["ac"] <= pos["ac"]
    emit("=" * 78)
    if cleared and beats_pos:
        verdict = "SUPPORTED (formalism): substrate-state oracle clears the wall AND beats position"
    elif sub_le_pos:
        verdict = ("REFUSED: SUBSTRATE <= POSITION -- the extra substrate features do not help; "
                   "leans toward an enzyme-specific (not substrate-universal) policy.")
    elif not cleared:
        verdict = ("REFUSED: SUBSTRATE does not clear 0.567 (CI straddles/below the wall) -- the "
                   "wall stands even with the oracle substrate state.")
    else:
        verdict = "INCONCLUSIVE: SUBSTRATE > POSITION but CI straddles 0.567 (signal, underpowered)."
    emit(f"VERDICT: {verdict}")
    emit("Caveat (load-bearing): SUBSTRATE is an ORACLE (oxid_sum/n_red/prev = true prior actions).")
    emit("Even a SUPPORTED here is a statement about the FORMALISM a_t=f(s_t); the deployable")
    emit("wall-clearance question is CONSTANT-vs-POSITION, both of which use only observable state.")

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(L) + "\n")
    emit(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
