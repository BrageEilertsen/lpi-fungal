"""Phase D3 ablation: POSE-only policy (no POSITION, no SUBSTRATE_prefix).

Asks the orthogonality question: do the 5 spatial pocket descriptors alone -- with
NO cycle index, NO chain-length / oxidation prefix, NO subclass indicator -- carry
enough information to resolve the citrinin candidate-set entropy?

  If citrinin H drops to ~0  -> POSE is a fully orthogonal structural channel.
                                Spatial geometry alone routes the program; the
                                model doesn't need temporal/substrate context.
  If citrinin H stays high   -> POSE is a complementary multiplier that needs
                                SUBSTRATE/POSITION as context to break symmetry.

Reuses train_weak_pose.py's corpus assembly + loss; the only change is the feature
name list driving feat_tensor_ext.
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

import train_weak as tw  # noqa: E402
import train_weak_pose as twp  # noqa: E402

# Monkey-patch the feature list to POSE-only and resize policy input dim.
twp.EXTENDED_FEATURE_NAMES = twp.POSE_FEAT_NAMES  # 5 dims
twp.N_FEAT_EXT = len(twp.EXTENDED_FEATURE_NAMES)

# Have to rebuild the policy class with the new dim
class PoseOnlyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mu", torch.zeros(twp.N_FEAT_EXT))
        self.register_buffer("sd", torch.ones(twp.N_FEAT_EXT))
        self.linear = torch.nn.Linear(twp.N_FEAT_EXT, tw.N_ACTIONS)
        self.norm_buffer_set = False

    def set_normalization(self, mu, sd):
        self.mu.copy_(mu); self.sd.copy_(sd); self.norm_buffer_set = True

    def forward(self, X):
        if self.norm_buffer_set:
            X = (X - self.mu) / self.sd
        return self.linear(X)


twp.ExtendedPolicy = PoseOnlyPolicy
torch.manual_seed(0xC0FFEE)

OUT_LOG = ROOT / "results" / "phase_d3_train_pose_only.log"


def main():
    log_lines = []
    log_lines.append("Phase D3 ABLATION: POSE-only (no POSITION, no SUBSTRATE)")
    log_lines.append(f"  feature dims: {twp.N_FEAT_EXT}  features: {twp.EXTENDED_FEATURE_NAMES}")
    log_lines.append("=" * 70)

    corpus_full = tw.build_corpus()
    pose_df = pd.read_parquet(ROOT / "data" / "policy" / "phase_d_pose.parquet")
    corpus, stats = twp.attach_pose_to_corpus(corpus_full, pose_df)
    log_lines.append(f"corpus (conformational-only): {len(corpus)} assets, "
                     f"{stats['n_pose_attached']} POSE rows attached")
    n_cycles = sum(len(a.candidates[a.parsimony_best_idx]) for a in corpus)
    log_lines.append(f"  canonical cycles: {n_cycles}")
    log_lines.append("")
    for line in log_lines:
        print(line)

    log_lines.append("[A] Full-corpus training")
    print(log_lines[-1])
    policy = twp.train_full_batch(corpus, log_lines=log_lines)
    ac, aa, nc, na = twp.per_cycle_accuracy(policy, corpus)
    log_lines.append(f"\n  full-corpus -- canonical acc={ac:.3f} ({nc}), agreed acc={aa:.3f} ({na})")
    print(log_lines[-1])

    log_lines.append("")
    log_lines.append(f"[B] LOSO eval over {len(corpus)} assets (POSE-only)")
    print(log_lines[-1])
    loso = twp.loso_eval(corpus, log_lines=log_lines)

    log_lines.append("")
    log_lines.append("=" * 70)
    log_lines.append("PHASE D3 POSE-ONLY ABLATION:")
    log_lines.append(f"  LOSO acc canonical: {loso['acc_canonical']:.4f} (n={loso['n_canonical']})")
    log_lines.append(f"  LOSO acc agreed:    {loso['acc_agreed']:.4f} (n={loso['n_agreed']})")
    log_lines.append("")
    log_lines.append("Reference comparisons on the SAME n=62 conformational corpus:")
    log_lines.append("  D2 substrate-only (n=62): canonical=0.2903  citrinin_H=~0.692")
    log_lines.append("  D3 POSITION+SUBSTRATE+POSE (n=62): canonical=0.3548  citrinin_H_held=0.003")
    log_lines.append("")
    log_lines.append("If citrinin H still drops to near 0 with POSE alone, the structural")
    log_lines.append("channel is fully orthogonal. If H stays high, POSE complements SUBSTRATE.")
    print()
    for line in log_lines[-10:]:
        print(line)

    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOG.write_text("\n".join(log_lines) + "\n")
    print(f"\nwrote {OUT_LOG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
