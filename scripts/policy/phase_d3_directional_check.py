"""Phase D3 directional cross-check: for citrinin / isoterrein / asperlin, run the
POSE+SUBSTRATE LOSO held-out model, then print the posterior P(z | x) over each
BGC's Z* candidates. Argmax candidate's program + SMILES gets compared to the
literature program.

For each of the three target BGCs:
  1. Build the full corpus.
  2. Attach POSE features.
  3. Hold out the target BGC, train on the remaining 15.
  4. Compute log P(z) for each candidate z in Z*(target).
  5. Normalize to posterior. Argmax = the model's directional pick.
  6. Print rank-1 program + SMILES + biological markers.
"""
from __future__ import annotations

import json
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

GROUND_TRUTH = ROOT / "data" / "policy" / "ground_truth_programs.json"
TARGETS = ["BGC0001338", "BGC0000161", "BGC0002180"]  # citrinin, isoterrein, asperlin


def candidate_posterior(policy, asset):
    with torch.no_grad():
        log_p = torch.stack([twp.candidate_logprob_ext(policy, c) for c in asset.candidates])
        log_post = log_p - torch.logsumexp(log_p, dim=0)
        return log_post.exp().numpy()


def program_summary(prog: dict) -> str:
    cycs = " -> ".join(
        f"{c['reduction']}{'+M' if c['c_methyl'] else ''}"
        for c in prog["cycles"]
    )
    return f"starter={prog['starter']:<8s}  ({cycs})  {prog['release']}"


def cross_check_one(bgc_id: str, corpus, gt) -> None:
    target_asset = next((a for a in corpus if a.bgc == bgc_id), None)
    if target_asset is None:
        print(f"\n{bgc_id}: NOT IN CONFORMATIONAL CORPUS (substrate-only or excluded)")
        return
    print(f"\n{'=' * 80}")
    print(f"=== {bgc_id} :: {gt[bgc_id]['family']} ({gt[bgc_id]['compound_name']})")
    print(f"=== |Z*|={target_asset.n_z_star}; chemistry_notes:")
    print(f"    {gt[bgc_id]['chemistry_notes'][:160]}")
    print()

    # Train LOSO with target held out
    torch.manual_seed(0xC0FFEE)
    train_set = [a for a in corpus if a.bgc != bgc_id]
    policy = twp.train_full_batch(train_set, n_steps=600, log_lines=None)
    post = candidate_posterior(policy, target_asset)

    # Sort candidates by posterior (descending)
    gt_candidates = gt[bgc_id]["candidates"]
    parsimony_scores = np.array([c.get("score", -np.inf) for c in gt_candidates])
    order = np.argsort(-post)

    print(f"Rank  Posterior  ParsimonyScore  Program  ->  SMILES")
    print("-" * 100)
    for rank, k in enumerate(order, start=1):
        c = gt_candidates[k]
        smi = c["smiles"]
        if len(smi) > 50:
            smi = smi[:47] + "..."
        prog_str = program_summary(c["program"])
        flag = "  <-- ARGMAX" if rank == 1 else ("  (parsimony-best)" if k == int(np.argmax(parsimony_scores)) else "")
        print(f"  {rank:>2}   {post[k]:>9.4f}   {c['score']:>+7.2f}      {prog_str}{flag}")
        if rank <= 3:
            print(f"          SMILES: {smi}")
    print()
    print(f"Entropy of posterior: H = {-(post * np.log(post + 1e-12)).sum():.4f}  "
          f"(max = log {target_asset.n_z_star} = {np.log(target_asset.n_z_star):.4f})")


def main() -> None:
    gt = json.loads(GROUND_TRUTH.read_text())
    corpus_full = tw.build_corpus()
    pose_df = pd.read_parquet(ROOT / "data" / "policy" / "phase_d_pose.parquet")
    corpus, _ = twp.attach_pose_to_corpus(corpus_full, pose_df)

    for bgc in TARGETS:
        cross_check_one(bgc, corpus, gt)


if __name__ == "__main__":
    main()
