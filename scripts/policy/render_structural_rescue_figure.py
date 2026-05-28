"""Render figure_structural_rescue.pdf -- the §11 load-bearing figure.

Panel A: held-out candidate entropy across three feature regimes (SUBSTRATE_prefix,
POSE only, SUBSTRATE+POSE) for the three flagship SILVER assets (citrinin,
isoterrein, asperlin). Computes the 9 (BGC, regime) entries by running LOSO with
each target held out under each regime; renders as a grouped bar chart with the
log|Z*| ceiling drawn for each BGC.

Panel B: candidate posterior under SUBSTRATE+POSE for isoterrein and asperlin,
showing how the joint channel routes posterior mass to the HR-PKS early-reduction
candidate while actively rejecting the parsimony-best un-biological alternative.

Output: paper/figure_structural_rescue.pdf
"""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "policy"))
sys.path.insert(0, str(ROOT / "src"))

import train_weak as tw
import train_weak_pose as twp

GROUND_TRUTH = ROOT / "data" / "policy" / "ground_truth_programs.json"
OUT_PDF = ROOT / "paper" / "figure_structural_rescue.pdf"
OUT_PNG = ROOT / "paper" / "figure_structural_rescue.png"
OUT_DATA = ROOT / "results" / "phase_d3_figure_data.json"

torch.manual_seed(0xC0FFEE)

TARGETS = [
    ("BGC0001338", "citrinin",   2),
    ("BGC0000161", "isoterrein", 3),
    ("BGC0002180", "asperlin",   5),
]
REGIMES = ["SUBSTRATE_prefix", "POSE_only", "SUBSTRATE+POSE"]
REGIME_COLORS = {"SUBSTRATE_prefix": "#7a7a7a", "POSE_only": "#9c5fcf",
                 "SUBSTRATE+POSE": "#2e7d32"}


# ------- Feature-set selectors ---------------------------------------------------
SUBSTRATE_FEATS = ["t", "frac", "sub_hr", "chain_c", "oxid_sum", "n_red", "prev"]
POSE_FEATS = ["pocket_contact_count_5A", "pocket_polar_contacts",
              "pocket_hydrophobic_contacts", "ligand_sasa_buried_fraction",
              "pocket_volume_A3"]
REGIME_FEATURES = {
    "SUBSTRATE_prefix": SUBSTRATE_FEATS,
    "POSE_only":        POSE_FEATS,
    "SUBSTRATE+POSE":   SUBSTRATE_FEATS + POSE_FEATS,
}


def feat_tensor_regime(feats: list, names: list[str]) -> torch.Tensor:
    rows = [[getattr(f, n) for n in names] for f in feats]
    return torch.tensor(rows, dtype=torch.float32)


def candidate_logprob_regime(policy, feats, names) -> torch.Tensor:
    if not feats:
        return torch.tensor(0.0)
    X = feat_tensor_regime(feats, names)
    y = torch.tensor([f.label for f in feats], dtype=torch.long)
    logits = policy(X)
    log_probs = torch.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, y.unsqueeze(-1)).squeeze(-1).sum()


def nlml_regime(policy, assets, names) -> torch.Tensor:
    out = []
    for a in assets:
        log_p = torch.stack([candidate_logprob_regime(policy, c, names)
                              for c in a.candidates])
        out.append(-torch.logsumexp(log_p, dim=0))
    return torch.stack(out).sum()


def make_policy(input_dim: int) -> torch.nn.Module:
    class P(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("mu", torch.zeros(input_dim))
            self.register_buffer("sd", torch.ones(input_dim))
            self.linear = torch.nn.Linear(input_dim, tw.N_ACTIONS)
            self.norm_set = False
        def forward(self, X):
            if self.norm_set:
                X = (X - self.mu) / self.sd
            return self.linear(X)
        def set_norm(self, mu, sd):
            self.mu.copy_(mu); self.sd.copy_(sd); self.norm_set = True
    return P()


def fit_norm(assets, names) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for a in assets:
        for cf in a.candidates[a.parsimony_best_idx]:
            rows.append([getattr(cf, n) for n in names])
    X = torch.tensor(rows, dtype=torch.float32)
    mu = X.mean(dim=0); sd = X.std(dim=0)
    sd = torch.where(sd > 1e-6, sd, torch.ones_like(sd))
    return mu, sd


def train(assets, regime_names, n_steps=600, lr=0.05):
    policy = make_policy(len(regime_names))
    mu, sd = fit_norm(assets, regime_names)
    policy.set_norm(mu, sd)
    opt = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(n_steps):
        opt.zero_grad()
        loss = nlml_regime(policy, assets, regime_names)
        loss.backward()
        opt.step()
    return policy


def posterior_and_entropy(policy, asset, regime_names) -> tuple[np.ndarray, float]:
    with torch.no_grad():
        log_p = torch.stack([candidate_logprob_regime(policy, c, regime_names)
                              for c in asset.candidates])
        log_post = log_p - torch.logsumexp(log_p, dim=0)
        p = log_post.exp().numpy()
    H = -(p * np.log(p + 1e-12)).sum()
    return p, float(H)


# ------- Main computation --------------------------------------------------------

def main() -> None:
    print("Building corpus + attaching POSE...")
    corpus_full = tw.build_corpus()
    pose_df = pd.read_parquet(ROOT / "data" / "policy" / "phase_d_pose.parquet")
    corpus, _ = twp.attach_pose_to_corpus(corpus_full, pose_df)
    print(f"  filtered to {len(corpus)} conformational assets")

    # ---- Compute Panel A data: H_held[regime][bgc] -----------------------------
    H_held: dict[str, dict[str, float]] = {r: {} for r in REGIMES}
    posteriors_joint: dict[str, np.ndarray] = {}

    for bgc, name, n_zstar in TARGETS:
        target = next((a for a in corpus if a.bgc == bgc), None)
        if target is None:
            print(f"  WARNING: {bgc} not in conformational corpus")
            continue
        train_set = [a for a in corpus if a.bgc != bgc]
        for regime in REGIMES:
            torch.manual_seed(0xC0FFEE)
            names = REGIME_FEATURES[regime]
            policy = train(train_set, names)
            p, H = posterior_and_entropy(policy, target, names)
            H_held[regime][bgc] = H
            if regime == "SUBSTRATE+POSE":
                posteriors_joint[bgc] = p
            print(f"  {regime:<18s} {name:<12s} H={H:.4f}  (max log|Z*|={np.log(n_zstar):.4f})")

    # Save raw data for record
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps({
        "H_held": H_held,
        "posteriors_joint": {b: p.tolist() for b, p in posteriors_joint.items()},
        "targets": [{"bgc": b, "name": n, "n_zstar": z} for b, n, z in TARGETS],
    }, indent=2))
    print(f"\nwrote {OUT_DATA.relative_to(ROOT)}")

    # ---- Render figure ---------------------------------------------------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2),
                                    gridspec_kw={"width_ratios": [1.0, 1.1]})

    # Panel A: grouped bar of H_held per (BGC, regime)
    bgc_labels = [name for _, name, _ in TARGETS]
    x = np.arange(len(bgc_labels))
    width = 0.26
    for i, regime in enumerate(REGIMES):
        vals = [H_held[regime][bgc] for bgc, _, _ in TARGETS]
        offset = (i - 1) * width
        bars = axA.bar(x + offset, vals, width,
                        label=regime.replace("_", "+").replace("prefix", "prefix"),
                        color=REGIME_COLORS[regime], edgecolor="black", linewidth=0.6)
        # value labels
        for bar, v in zip(bars, vals):
            axA.text(bar.get_x() + bar.get_width() / 2, v + 0.04,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    # log|Z*| reference markers
    for i, (_, _, n_zstar) in enumerate(TARGETS):
        ceiling = np.log(n_zstar)
        axA.hlines(ceiling, x[i] - 1.5 * width, x[i] + 1.5 * width,
                    linestyles="dashed", colors="black", linewidth=0.8, alpha=0.6)
        axA.text(x[i] + 1.6 * width, ceiling, f"log{n_zstar}",
                  va="center", ha="left", fontsize=7, alpha=0.7)

    axA.set_xticks(x)
    axA.set_xticklabels(bgc_labels)
    axA.set_ylabel("Held-out candidate entropy $H_{\\mathrm{held}}$ (nats)")
    axA.set_title("A. Two engines: substrate (label-distinguishable) vs. structural (label-symmetric)",
                   fontsize=9.5)
    axA.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    axA.set_ylim(0, max(np.log(n) for _, _, n in TARGETS) * 1.18)
    axA.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel B: candidate posterior under SUBSTRATE+POSE for isoterrein + asperlin
    # Show as horizontal bars sorted by posterior desc; annotate biology.
    gt = json.loads(GROUND_TRUTH.read_text())
    selected = [("BGC0000161", "isoterrein"), ("BGC0002180", "asperlin")]
    n_cands_total = sum(len(gt[b]["candidates"]) for b, _ in selected)
    y_pos = 0
    bar_height = 0.65
    section_gap = 1.0

    for sel_idx, (bgc, name) in enumerate(selected):
        cands = gt[bgc]["candidates"]
        post = posteriors_joint[bgc]
        # Sort by posterior desc; show all
        order = np.argsort(-post)
        # parsimony-best index = the one with max parsimony score
        scores = np.array([c.get("score", -np.inf) for c in cands])
        parsimony_best_idx = int(np.argmax(scores))
        argmax_idx = int(order[0])

        # Section header text
        axB.text(0.0, y_pos + bar_height * 0.5,
                  f"{name} ($|Z^*|$={len(cands)})", ha="left", va="center",
                  fontsize=9, fontweight="bold", transform=axB.transData)
        y_pos += 1.0

        for k in order:
            c = cands[k]
            cycs = ",".join(
                f"{x['reduction']}{'+M' if x['c_methyl'] else ''}"
                for x in c["program"]["cycles"]
            )
            rel = c["program"]["release"].replace("aldol_aromatic", "aldol-arom").replace("hydrolysis", "hydrol")
            label = f"({cycs}) {rel}"
            color = "#2e7d32" if k == argmax_idx else "#bdbdbd"
            if k == parsimony_best_idx and k != argmax_idx:
                color = "#d32f2f"  # parsimony-best that the model rejected -- highlight red
            axB.barh(y_pos, post[k], height=bar_height, color=color,
                      edgecolor="black", linewidth=0.5)
            axB.text(post[k] + 0.02, y_pos, f"{post[k]:.3f}", va="center", fontsize=7)
            axB.text(-0.02, y_pos, label, ha="right", va="center", fontsize=7.5,
                      family="monospace")
            # Mark argmax / parsimony-best
            if k == argmax_idx:
                axB.text(0.5, y_pos + bar_height * 0.55, "model argmax",
                          ha="center", fontsize=6.5, color="#2e7d32",
                          fontweight="bold", transform=axB.transData)
            if k == parsimony_best_idx and k != argmax_idx:
                axB.text(0.5, y_pos + bar_height * 0.55, "parsimony-best (rejected)",
                          ha="center", fontsize=6.5, color="#d32f2f",
                          fontweight="bold", transform=axB.transData)
            y_pos += 1.0
        y_pos += section_gap

    axB.set_xlim(-0.55, 1.05)
    axB.set_ylim(-0.5, y_pos)
    axB.invert_yaxis()
    axB.set_xlabel("Posterior $P(z \\mid x)$ under SUBSTRATE$+$POSE")
    axB.set_title("B. Joint posterior selects the biologically authentic pathway",
                   fontsize=9.5)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_visible(False)
    axB.tick_params(left=False, labelleft=False)
    axB.axvline(0, color="black", linewidth=0.6)
    axB.grid(axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PDF, format="pdf", bbox_inches="tight")
    plt.savefig(OUT_PNG, format="png", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUT_PDF.relative_to(ROOT)}")
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
