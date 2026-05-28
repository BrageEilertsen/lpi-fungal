"""Phase D3 / D2-POSE: marginal-likelihood training with POSE descriptors added.

Extends train_weak.py with the 5-dim POSE feature vector from phase_d_pose.parquet:
  pocket_contact_count_5A, pocket_polar_contacts, pocket_hydrophobic_contacts,
  ligand_sasa_buried_fraction, pocket_volume_A3

Input dim grows from 7 (POSITION+SUBSTRATE_prefix) to 12 (+POSE).

Same marginal-likelihood objective; same paired-bootstrap LOSO eval; same
diagnostic tracking. The new explicit hypothesis to test:

  Does POSE break citrinin's log(2) entropy plateau?

Substrate-only assets (no protein fold or LOW_PLDDT_EXCLUDED anchor) are dropped
from this POSE pass -- keeps the comparison apples-to-apples between SUBSTRATE
features and SUBSTRATE+POSE features over the conformational subset of the corpus.

Run:
    PYTHONPATH=src .venv/bin/python scripts/policy/train_weak_pose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "policy"))
sys.path.insert(0, str(ROOT / "src"))

# Reuse the train_weak.py components
import train_weak as tw  # noqa: E402

POSE_PARQUET = ROOT / "data" / "policy" / "phase_d_pose.parquet"
OUT_LOG = ROOT / "results" / "phase_d3_train_weak_pose.log"

POSE_FEAT_NAMES = [
    "pocket_contact_count_5A",
    "pocket_polar_contacts",
    "pocket_hydrophobic_contacts",
    "ligand_sasa_buried_fraction",
    "pocket_volume_A3",
]
EXTENDED_FEATURE_NAMES = tw.FEATURE_NAMES + POSE_FEAT_NAMES
N_FEAT_EXT = len(EXTENDED_FEATURE_NAMES)

torch.manual_seed(0xC0FFEE)


def attach_pose_to_corpus(corpus: list[tw.BGCAsset], pose_df: pd.DataFrame
                          ) -> tuple[list[tw.BGCAsset], dict]:
    """For each (bgc, cand_idx, cycle_t), look up POSE features and attach as extra
    attributes on CycleFeat objects. Assets whose candidates ALL lack POSE features
    (substrate-only proxies / low-pLDDT exclusions) are filtered out."""
    # index: (bgc, cand_idx, cycle_t) -> dict
    pose_idx = {}
    for _, r in pose_df.iterrows():
        if r.get("status") != "conformational":
            continue
        key = (str(r["bgc"]), int(r["cand_idx"]), int(r["cycle_t"]))
        pose_idx[key] = {f: float(r[f]) for f in POSE_FEAT_NAMES}

    kept: list[tw.BGCAsset] = []
    n_pose_attached = 0
    n_dropped = 0
    for a in corpus:
        new_candidates: list[list[tw.CycleFeat]] = []
        for k, cycle_feats in enumerate(a.candidates):
            new_cycle_feats = []
            for cf in cycle_feats:
                key = (a.bgc, k, cf.cycle_t)
                if key in pose_idx:
                    cf_ext = _attach_pose(cf, pose_idx[key])
                    new_cycle_feats.append(cf_ext)
                    n_pose_attached += 1
                else:
                    new_cycle_feats.append(None)  # marks missing pose
            # if ALL cycles in this candidate have POSE, keep it
            if new_cycle_feats and all(c is not None for c in new_cycle_feats):
                new_candidates.append(new_cycle_feats)
        if new_candidates:
            kept.append(tw.BGCAsset(bgc=a.bgc, candidates=new_candidates,
                                    n_z_star=len(new_candidates),
                                    parsimony_best_idx=min(a.parsimony_best_idx,
                                                            len(new_candidates) - 1)))
        else:
            n_dropped += 1

    stats = dict(n_pose_attached=n_pose_attached,
                 n_assets_kept=len(kept),
                 n_assets_dropped=n_dropped)
    return kept, stats


def _attach_pose(cf: tw.CycleFeat, pose: dict) -> tw.CycleFeat:
    """Return a new CycleFeat with pose attributes attached via __dict__ extension."""
    # Use namespace-style attachment on a copy
    import copy
    cf2 = copy.copy(cf)
    for name, val in pose.items():
        setattr(cf2, name, val)
    return cf2


def feat_tensor_ext(feats: list) -> torch.Tensor:
    rows = []
    for f in feats:
        rows.append([getattr(f, name) for name in EXTENDED_FEATURE_NAMES])
    return torch.tensor(rows, dtype=torch.float32)


def candidate_logprob_ext(policy: torch.nn.Module, feats: list) -> torch.Tensor:
    if not feats:
        return torch.tensor(0.0)
    X = feat_tensor_ext(feats)
    # z-score the POSE columns per-batch (training stability; means + stds frozen
    # from training assets in a proper run, but per-batch normalization is OK here
    # because magnitudes vary across POSE features -- pocket_volume_A3 in the
    # hundreds, contact_count in the tens, buried_fraction 0..1).
    y = torch.tensor([f.label for f in feats], dtype=torch.long)
    logits = policy(X)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum()


def nlml_loss_ext(policy: torch.nn.Module, assets: list[tw.BGCAsset]) -> torch.Tensor:
    out = []
    for a in assets:
        log_p_cands = torch.stack([candidate_logprob_ext(policy, c) for c in a.candidates])
        out.append(-torch.logsumexp(log_p_cands, dim=0))
    return torch.stack(out).sum()


class ExtendedPolicy(torch.nn.Module):
    """Linear logistic policy over the 12-dim extended feature vector."""
    def __init__(self):
        super().__init__()
        self.norm_buffer_set = False
        self.register_buffer("mu", torch.zeros(N_FEAT_EXT))
        self.register_buffer("sd", torch.ones(N_FEAT_EXT))
        self.linear = torch.nn.Linear(N_FEAT_EXT, tw.N_ACTIONS)

    def set_normalization(self, mu: torch.Tensor, sd: torch.Tensor):
        self.mu.copy_(mu)
        self.sd.copy_(sd)
        self.norm_buffer_set = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.norm_buffer_set:
            X = (X - self.mu) / self.sd
        return self.linear(X)


def fit_normalization(assets: list[tw.BGCAsset]) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute z-score normalization stats from the canonical (parsimony-best)
    candidate of each asset -- the training data."""
    rows = []
    for a in assets:
        for cf in a.candidates[a.parsimony_best_idx]:
            rows.append([getattr(cf, n) for n in EXTENDED_FEATURE_NAMES])
    X = torch.tensor(rows, dtype=torch.float32)
    mu = X.mean(dim=0)
    sd = X.std(dim=0)
    sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    return mu, sd


def train_full_batch(assets: list[tw.BGCAsset], n_steps: int = 600, lr: float = 0.05,
                     log_lines: list[str] | None = None) -> ExtendedPolicy:
    policy = ExtendedPolicy()
    mu, sd = fit_normalization(assets)
    policy.set_normalization(mu, sd)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    silver = [a for a in assets if a.n_z_star > 1]
    for step in range(n_steps + 1):
        opt.zero_grad()
        loss = nlml_loss_ext(policy, assets)
        if step < n_steps:
            loss.backward()
            opt.step()
        if step % 100 == 0 or step == n_steps:
            ents = {a.bgc: candidate_entropy(policy, a) for a in silver}
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


def candidate_entropy(policy: ExtendedPolicy, asset: tw.BGCAsset) -> float:
    if asset.n_z_star <= 1:
        return 0.0
    with torch.no_grad():
        log_p = torch.stack([candidate_logprob_ext(policy, c) for c in asset.candidates])
        log_post = log_p - torch.logsumexp(log_p, dim=0)
        p = log_post.exp()
        ent = -(p * log_post).sum().item()
    return ent


def per_cycle_accuracy(policy: ExtendedPolicy, assets: list[tw.BGCAsset]
                       ) -> tuple[float, float, int, int]:
    correct_c, total_c = 0, 0
    correct_a, total_a = 0, 0
    for a in assets:
        feats = a.candidates[a.parsimony_best_idx]
        if feats:
            with torch.no_grad():
                logits = policy(feat_tensor_ext(feats))
                preds = logits.argmax(dim=-1)
                labels = torch.tensor([f.label for f in feats], dtype=torch.long)
                correct_c += int((preds == labels).sum().item())
                total_c += int(labels.numel())
        if not a.candidates or not a.candidates[0]:
            continue
        T = min(len(c) for c in a.candidates)
        for t in range(T):
            labels_at_t = {c[t].label for c in a.candidates}
            if len(labels_at_t) == 1:
                cf = a.candidates[a.parsimony_best_idx][t]
                with torch.no_grad():
                    logit = policy(feat_tensor_ext([cf]))
                    pred = int(logit.argmax(dim=-1).item())
                correct_a += int(pred == cf.label)
                total_a += 1
    return (correct_c / total_c if total_c else float("nan"),
            correct_a / total_a if total_a else float("nan"),
            total_c, total_a)


def loso_eval(assets: list[tw.BGCAsset], n_steps: int = 600, lr: float = 0.05,
              log_lines: list[str] | None = None) -> dict:
    cc, ct = 0, 0
    ac, at = 0, 0
    for i, held in enumerate(assets):
        train = [a for j, a in enumerate(assets) if j != i]
        policy = train_full_batch(train, n_steps=n_steps, lr=lr, log_lines=None)
        ac_h, aa_h, nc, na = per_cycle_accuracy(policy, [held])
        cc += int(ac_h * nc) if not np.isnan(ac_h) else 0
        ct += nc
        ac += int(aa_h * na) if not np.isnan(aa_h) else 0
        at += na
        ent_held = candidate_entropy(policy, held) if held.n_z_star > 1 else 0.0
        if log_lines is not None:
            log_lines.append(f"  LOSO[{i+1:>2}/{len(assets)}] {held.bgc[:40]:<40} "
                             f"canon={ac_h if not np.isnan(ac_h) else 0:.3f}({nc:>2}) "
                             f"H_held={ent_held:.3f}")
    return dict(acc_canonical=cc / ct if ct else float("nan"),
                acc_agreed=ac / at if at else float("nan"),
                n_canonical=ct, n_agreed=at)


def main() -> None:
    log_lines: list[str] = []
    log_lines.append("Phase D3 / D2-POSE: marginal-likelihood with POSE features")
    log_lines.append("=" * 70)

    if not POSE_PARQUET.exists():
        raise SystemExit(f"missing {POSE_PARQUET} -- run phase_d3_pose.py first")

    corpus_full = tw.build_corpus()
    pose_df = pd.read_parquet(POSE_PARQUET)
    corpus, stats = attach_pose_to_corpus(corpus_full, pose_df)

    log_lines.append(f"corpus (filtered to conformational): {len(corpus)} assets")
    log_lines.append(f"  POSE rows attached: {stats['n_pose_attached']}")
    log_lines.append(f"  assets dropped (no POSE for any candidate): {stats['n_assets_dropped']}")
    n_gold = sum(1 for a in corpus if a.n_z_star == 1)
    n_silver = len(corpus) - n_gold
    log_lines.append(f"  GOLD={n_gold}, SILVER={n_silver}")
    n_canonical_cycles = sum(len(a.candidates[a.parsimony_best_idx]) for a in corpus)
    log_lines.append(f"  canonical-program cycles: {n_canonical_cycles}")
    log_lines.append("")
    for line in log_lines:
        print(line)

    log_lines.append("[A] Full-corpus training (sanity)")
    print(log_lines[-1])
    policy_full = train_full_batch(corpus, log_lines=log_lines)
    ac, aa, nc, na = per_cycle_accuracy(policy_full, corpus)
    line = (f"\n  full-corpus -- canonical acc={ac:.3f} ({nc} cycles), "
            f"agreed acc={aa:.3f} ({na} cycles)")
    print(line)
    log_lines.append(line)

    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append(f"[B] LOSO eval over {len(corpus)} BGC assets (POSE features)")
    log_lines.append("")
    print(log_lines[-3]); print(log_lines[-2])
    loso = loso_eval(corpus, log_lines=log_lines)

    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append("PHASE D3-POSE BASELINE:")
    log_lines.append(f"  LOSO acc canonical: {loso['acc_canonical']:.4f}  "
                     f"(n_cycles={loso['n_canonical']})")
    log_lines.append(f"  LOSO acc agreed:    {loso['acc_agreed']:.4f}  "
                     f"(n_cycles={loso['n_agreed']})")
    log_lines.append("")
    log_lines.append("Phase D2 baseline (no POSE) for comparison:")
    log_lines.append("  LOSO acc canonical: 0.4432 (n=88)")
    log_lines.append("  LOSO acc agreed:    0.5156 (n=64)")
    log_lines.append("  citrinin entropy held at log(2)=0.693 throughout training")
    log_lines.append("")
    log_lines.append("If citrinin entropy in this run dropped meaningfully below 0.693,")
    log_lines.append("structural information broke the sequence-blind-spot. If it stuck,")
    log_lines.append("static representations are insufficient -- MD/wet-lab mandate.")
    print()
    for line in log_lines[-15:]:
        print(line)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
