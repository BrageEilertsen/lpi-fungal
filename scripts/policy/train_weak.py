"""Phase D2: weak-supervision marginal-likelihood objective for the per-cycle policy.

For each BGC asset B, the inverse compiler returns a verified candidate set Z*(B) of
size 1..K. The sequence policy pi_theta defines a probability over a program z as a
product of per-cycle conditional actions; we minimize the Negative Log Marginal
Likelihood over the corpus D:

  L(theta) = - Sum_B  log Sum_{z in Z*(B)}  P(z | B; theta)

       where  P(z | B; theta) = Prod_t  pi_theta(z_t | s_t^(z), B)

Properties:
  - |Z*|=1 case collapses to standard cross-entropy (the curated GOLD tier).
  - |Z*|>1 case (SILVER) splits gradient across candidates by current model prob;
    candidates that share substructure with the GOLD set accumulate probability.
  - logsumexp keeps the marginal numerically stable across the per-candidate product.

PyTorch backbone (Brage's call): keeps the loss function + data pipelines invariant
when we swap the linear policy for a structural neural network head in Phase D3.

Phase D2 baseline: POSITION + SUBSTRATE features only, no POSE (those come in D3).
This isolates sample-size as the single variable changed from the n=23 Phase C run.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lpi.data.curated import load_all  # noqa: E402

GROUND_TRUTH = ROOT / "data" / "policy" / "ground_truth_programs.json"
OUT_LOG = ROOT / "results" / "phase_d_train_weak.log"

torch.manual_seed(0xC0FFEE)

# Action space: per-cycle reduction state (4-way categorical).
REDUCTION_LABELS = ["keto", "kr", "dh", "er"]
LABEL_INDEX = {l: i for i, l in enumerate(REDUCTION_LABELS)}
N_ACTIONS = len(REDUCTION_LABELS)

STARTER_C = {"acetyl": 2, "propionyl": 3, "butyryl": 4, "hexanoyl": 6, "benzoyl": 7}


# ---- Per-cycle feature extraction ------------------------------------------------

FEATURE_NAMES = ["t", "frac", "sub_hr", "chain_c", "oxid_sum", "n_red", "prev"]
N_FEAT = len(FEATURE_NAMES)


@dataclass
class CycleFeat:
    bgc: str
    cand_idx: int
    cycle_t: int
    n_cycles: int
    t: int
    frac: float
    sub_hr: int
    chain_c: int
    oxid_sum: int
    n_red: int
    prev: int
    label: int


def cycle_features_for_program(bgc: str, cand_idx: int, prog: dict, sub_hr: int) -> list[CycleFeat]:
    starter = prog["starter"]
    cycles = prog["cycles"]
    N = len(cycles)
    sc = STARTER_C.get(starter, 2)
    prior_levels: list[int] = []
    n_cmet = 0
    feats: list[CycleFeat] = []
    for t, c in enumerate(cycles, start=1):
        lv = LABEL_INDEX[c["reduction"]]
        chain_c = sc + 2 * t + n_cmet
        oxid_sum = sum(prior_levels)
        n_red = sum(1 for x in prior_levels if x > 0)
        prev = prior_levels[-1] if prior_levels else -1
        feats.append(CycleFeat(
            bgc=bgc, cand_idx=cand_idx, cycle_t=t, n_cycles=N,
            t=t, frac=t / N, sub_hr=sub_hr,
            chain_c=chain_c, oxid_sum=oxid_sum, n_red=n_red, prev=prev,
            label=lv,
        ))
        prior_levels.append(lv)
        n_cmet += int(c["c_methyl"])
    return feats


@dataclass
class BGCAsset:
    bgc: str
    candidates: list[list[CycleFeat]]
    n_z_star: int
    parsimony_best_idx: int  # the candidate with the most-parsimonious score


def build_corpus() -> list[BGCAsset]:
    assets: list[BGCAsset] = []

    # Curated 9 HR/PR -- |Z*|=1, the unique curated program.
    for e in load_all():
        if e.subclass not in ("HR", "PR"):
            continue
        prog_dict = dict(
            starter=e.program.starter,
            cycles=[dict(reduction=c.reduction.value, c_methyl=bool(c.c_methyl),
                         extender=c.extender.value) for c in e.program.cycles],
            release=e.program.release.value,
            n_cycles=e.program.n_cycles,
        )
        feats = cycle_features_for_program(
            bgc=f"curated:{e.name}", cand_idx=0, prog=prog_dict,
            sub_hr=1 if e.subclass == "HR" else 0,
        )
        assets.append(BGCAsset(bgc=f"curated:{e.name}", candidates=[feats],
                               n_z_star=1, parsimony_best_idx=0))

    # Inventory GOLD/SILVER from ground_truth_programs.json.
    gt = json.loads(GROUND_TRUTH.read_text())
    for bgc, v in gt.items():
        family = (v.get("family") or "").lower()
        sub_hr = 1 if any(t in family for t in (
            "strobilurin", "asperlin", "asperlactone", "gibepyrone")) else 0
        cand_feats: list[list[CycleFeat]] = []
        scores: list[float] = []
        for k, c in enumerate(v["candidates"]):
            feats = cycle_features_for_program(bgc=bgc, cand_idx=k,
                                               prog=c["program"], sub_hr=sub_hr)
            cand_feats.append(feats)
            scores.append(c.get("score", -np.inf))
        best = int(np.argmax(scores)) if scores else 0
        assets.append(BGCAsset(bgc=bgc, candidates=cand_feats,
                               n_z_star=len(cand_feats), parsimony_best_idx=best))
    return assets


def feat_tensor(feats: list[CycleFeat]) -> torch.Tensor:
    return torch.tensor([[getattr(f, n) for n in FEATURE_NAMES] for f in feats],
                        dtype=torch.float32)


def label_tensor(feats: list[CycleFeat]) -> torch.Tensor:
    return torch.tensor([f.label for f in feats], dtype=torch.long)


# ---- The policy ------------------------------------------------------------------

class LinearPolicy(nn.Module):
    """Per-cycle linear logistic: feature_vec -> 4 reduction-action logits."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_FEAT, N_ACTIONS)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.linear(feats)  # (T, A) logits


# ---- The marginal-likelihood loss ------------------------------------------------

def candidate_logprob(policy: LinearPolicy, feats: list[CycleFeat]) -> torch.Tensor:
    """log P(z | theta) = Sum_t log_softmax(logits)[label_t]. Scalar tensor."""
    if not feats:
        return torch.tensor(0.0)
    X = feat_tensor(feats)
    y = label_tensor(feats)
    logits = policy(X)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum()


def nlml_loss_one(policy: LinearPolicy, asset: BGCAsset) -> torch.Tensor:
    """-log Sum_z P(z | theta) for a single BGC asset. Scalar tensor."""
    log_p_cands = torch.stack([candidate_logprob(policy, c) for c in asset.candidates])
    return -torch.logsumexp(log_p_cands, dim=0)


def nlml_loss(policy: LinearPolicy, assets: list[BGCAsset]) -> torch.Tensor:
    return torch.stack([nlml_loss_one(policy, a) for a in assets]).sum()


# ---- Diagnostics: candidate posterior entropy on SILVER assets -------------------

def silver_candidate_entropy(policy: LinearPolicy, asset: BGCAsset) -> float:
    """Shannon entropy (nats) of the posterior over Z*(B) under current policy."""
    if asset.n_z_star <= 1:
        return 0.0
    with torch.no_grad():
        log_p = torch.stack([candidate_logprob(policy, c) for c in asset.candidates])
        log_post = log_p - torch.logsumexp(log_p, dim=0)
        p = log_post.exp()
        # Numerically safe entropy: sum -p log p, treat 0*log0 = 0
        ent = -(p * log_post).sum().item()
    return ent


# ---- Training loop ---------------------------------------------------------------

def train_full_batch(assets: list[BGCAsset], n_steps: int = 600, lr: float = 0.05,
                     log_lines: list[str] | None = None) -> LinearPolicy:
    """Full-batch AdamW. With 32 params + 88-cycle corpus, the gradient is
    deterministic and fast; stochastic mini-batching adds noise without benefit."""
    policy = LinearPolicy()
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    silver_assets = [a for a in assets if a.n_z_star > 1]

    for step in range(n_steps + 1):
        opt.zero_grad()
        loss = nlml_loss(policy, assets)
        if step < n_steps:
            loss.backward()
            opt.step()
        # Diagnostics every 100 steps + at end
        if step % 100 == 0 or step == n_steps:
            ents = {a.bgc: silver_candidate_entropy(policy, a) for a in silver_assets}
            mean_ent = float(np.mean(list(ents.values()))) if ents else 0.0
            citrinin = next((v for k, v in ents.items() if "0001338" in k), float("nan"))
            strob = next((v for k, v in ents.items() if "0001909" in k), float("nan"))
            asperlin = next((v for k, v in ents.items() if "0002180" in k), float("nan"))
            line = (f"  step={step:>4}  NLML={loss.item():>8.3f}  "
                    f"mean_silver_ent={mean_ent:.3f}  "
                    f"citrinin_H={citrinin:.3f}  strob_H={strob:.3f}  "
                    f"asperlin_H={asperlin:.3f}")
            print(line, flush=True)
            if log_lines is not None:
                log_lines.append(line)
    return policy


# ---- LOSO eval -------------------------------------------------------------------

def per_cycle_accuracy(policy: LinearPolicy, assets: list[BGCAsset]
                       ) -> tuple[float, float, int, int]:
    """Returns (acc_canonical, acc_agreed, n_canonical, n_agreed) at the per-cycle level.

    acc_canonical: per-cycle accuracy using the parsimony-best candidate per asset as
                   the ground-truth program (the "best biological guess" for SILVER).
    acc_agreed:    per-cycle accuracy restricted to cycles where ALL candidates in
                   Z*(B) agree on the reduction label (drops the ambiguous cycles).
                   GOLD assets contribute all their cycles; SILVER assets contribute
                   only the consensus cycles.
    """
    pred_canon = 0; total_canon = 0
    pred_agreed = 0; total_agreed = 0
    for a in assets:
        # canonical
        feats = a.candidates[a.parsimony_best_idx]
        if feats:
            with torch.no_grad():
                logits = policy(feat_tensor(feats))
                preds = logits.argmax(dim=-1)
                labels = label_tensor(feats)
                pred_canon += int((preds == labels).sum().item())
                total_canon += int(labels.numel())
        # agreed: only cycles where all candidates have the same label at that t
        if a.n_z_star == 0 or not a.candidates[0]:
            continue
        T = min(len(c) for c in a.candidates)
        for t in range(T):
            labels_at_t = {c[t].label for c in a.candidates}
            if len(labels_at_t) == 1:  # consensus
                feats_canon_t = a.candidates[a.parsimony_best_idx][t]
                with torch.no_grad():
                    logit = policy(feat_tensor([feats_canon_t]))
                    pred = int(logit.argmax(dim=-1).item())
                pred_agreed += int(pred == feats_canon_t.label)
                total_agreed += 1
    acc_c = pred_canon / total_canon if total_canon else float("nan")
    acc_a = pred_agreed / total_agreed if total_agreed else float("nan")
    return acc_c, acc_a, total_canon, total_agreed


def loso_eval(assets: list[BGCAsset], n_steps: int = 600, lr: float = 0.05,
              log_lines: list[str] | None = None) -> dict:
    """Leave-one-BGC-out: 25 folds. Returns canonical + agreed-cycle accuracies."""
    canon_correct = 0; canon_total = 0
    agreed_correct = 0; agreed_total = 0
    per_fold = []
    for i, held in enumerate(assets):
        train = [a for j, a in enumerate(assets) if j != i]
        policy = train_full_batch(train, n_steps=n_steps, lr=lr, log_lines=None)
        ac, aa, nc, na = per_cycle_accuracy(policy, [held])
        canon_correct += int(ac * nc) if not np.isnan(ac) else 0
        canon_total += nc
        agreed_correct += int(aa * na) if not np.isnan(aa) else 0
        agreed_total += na
        per_fold.append((held.bgc, ac, aa, nc, na))
        if log_lines is not None:
            log_lines.append(f"  LOSO[{i+1:>2}/{len(assets)}] {held.bgc[:40]:<40} "
                             f"canon={ac if not np.isnan(ac) else 0:.3f}({nc:>2}) "
                             f"agreed={aa if not np.isnan(aa) else 0:.3f}({na:>2})")
    acc_canon = canon_correct / canon_total if canon_total else float("nan")
    acc_agreed = agreed_correct / agreed_total if agreed_total else float("nan")
    return dict(acc_canonical=acc_canon, acc_agreed=acc_agreed,
                n_canonical=canon_total, n_agreed=agreed_total,
                per_fold=per_fold)


# ---- Main ------------------------------------------------------------------------

def main() -> None:
    log_lines: list[str] = []

    log_lines.append("Phase D2: weak-supervision marginal-likelihood training")
    log_lines.append("=" * 70)

    corpus = build_corpus()
    n_gold = sum(1 for a in corpus if a.n_z_star == 1)
    n_silver = sum(1 for a in corpus if a.n_z_star > 1)
    n_cycles_canon = sum(len(a.candidates[a.parsimony_best_idx]) for a in corpus)
    n_cands_total = sum(a.n_z_star for a in corpus)

    log_lines.append(f"corpus: {len(corpus)} BGC assets (GOLD={n_gold}, SILVER={n_silver})")
    log_lines.append(f"  cycles (parsimony-best per BGC): {n_cycles_canon}")
    log_lines.append(f"  candidate programs (sum over Z*): {n_cands_total}")
    log_lines.append("")

    # ---- Full-corpus training -------------------------------------------
    log_lines.append("[A] FULL-CORPUS TRAINING (sanity: NLML should drop monotonically)")
    log_lines.append(f"    AdamW, lr=0.05, weight_decay=1e-4, full-batch, 600 steps")
    log_lines.append("")
    for line in log_lines:
        print(line)

    policy_full = train_full_batch(corpus, log_lines=log_lines)
    ac_full, aa_full, nc_full, na_full = per_cycle_accuracy(policy_full, corpus)
    line = (f"\n  on full corpus -- canonical acc={ac_full:.3f} ({nc_full} cycles), "
            f"agreed acc={aa_full:.3f} ({na_full} cycles)")
    print(line)
    log_lines.append(line)

    # ---- LOSO eval -------------------------------------------------------
    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append("[B] LEAVE-ONE-BGC-OUT EVAL (25 folds)")
    log_lines.append("")
    print(log_lines[-3]); print(log_lines[-2]); print(log_lines[-1])
    loso = loso_eval(corpus, log_lines=log_lines)

    log_lines.append("")
    log_lines.append("=" * 70)
    summary = (f"PHASE D2 BASELINE (POSITION+SUBSTRATE only, n_BGC=25, "
               f"n_cycles_canonical={loso['n_canonical']}, "
               f"n_cycles_agreed={loso['n_agreed']}):")
    log_lines.append(summary)
    log_lines.append(f"  LOSO accuracy (canonical / parsimony-best ground truth): "
                     f"{loso['acc_canonical']:.4f}")
    log_lines.append(f"  LOSO accuracy (agreed-cycles only):                       "
                     f"{loso['acc_agreed']:.4f}")
    log_lines.append("")
    log_lines.append("Reference comparisons:")
    log_lines.append("  Phase C n=23 paired bootstrap mean: POSITION=0.532, SUBSTRATE=0.503")
    log_lines.append("  substrate_state_probe n=30 LOSO:    POSITION=0.60,  SUBSTRATE=0.47")
    log_lines.append("  the 0.567 wall:                     constant-domain LOSO baseline")
    print()
    for line in log_lines[-9:]:
        print(line)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
