"""Generate the paper figures from committed result CSVs/JSON. Reproducible:

    PYTHONPATH=src .venv/bin/python scripts/figures.py

Writes PNGs to results/figures/. Inputs (all committed):
  results/reachability.csv  results/corematch.csv  results/phase0c.csv
  results/differential_summary.json
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "figures"


def _read(name):
    return list(csv.DictReader((RES / name).open()))


def fig_reconstruction_tiers():
    """Fig 2: three reconstruction tiers + the ceiling."""
    reach = _read("reachability.csv")
    exact = sum(r["status"] == "reachable" for r in reach)
    p0c = _read("phase0c.csv")
    tailoring = sum(r["class"] == "verified" for r in p0c)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["exact-match\n(8/326)", "tailoring-latent\nverified (7/106 scanned)",
              "core-match\n(unreliable proxy:\n35–160)"]
    vals = [exact, tailoring, 160]
    colors = ["#2b8cbe", "#74a9cf", "#bdbdbd"]
    ax.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 2, str(v), ha="center")
    ax.set_ylabel("clusters")
    ax.set_title("Reconstructibility of fungal PKS products (MIBiG 4.0, n=326)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_reconstruction_tiers.png", dpi=150)
    plt.close(fig)


def fig_zstar():
    """Fig 3: |Z*| distribution from the tailoring-latent verified set."""
    p0c = _read("phase0c.csv")
    z = collections.Counter(int(r["z_star"]) for r in p0c
                            if r["class"] == "verified" and r["z_star"] not in ("", "-1"))
    fig, ax = plt.subplots(figsize=(5, 4))
    if z:
        ks = sorted(z)
        ax.bar([str(k) for k in ks], [z[k] for k in ks], color="#2b8cbe")
    ax.set_xlabel("|Z*(y)|  (verified explanations per product)")
    ax.set_ylabel("clusters")
    ax.set_title("Program identifiability: |Z*(y)| is near-unique where verified")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_zstar.png", dpi=150)
    plt.close(fig)


def fig_corematch_sensitivity():
    """Fig 4: core-match is an unreliable proxy (swings with one nuisance knob)."""
    # sensitivity values from the core-match analysis (PAPER_READINESS R9)
    knob = ["exact", "0", "1", "2", "3", "inf"]
    vals = [8, 35, 99, 136, 154, 160]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(knob, vals, "o-", color="#cb181d")
    for x, v in zip(knob, vals):
        ax.text(x, v + 3, str(v), ha="center", fontsize=8)
    ax.set_xlabel("max extra ring-closure bonds allowed (nuisance knob)")
    ax.set_ylabel("'reconstructible' clusters")
    ax.set_title("Core-match is unreliable: 8→160 over one parameter")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_corematch_sensitivity.png", dpi=150)
    plt.close(fig)


def fig_differential():
    """Fig 5: differential PoC — reduction/stereo + the 100%→57% transfer."""
    s = json.loads((RES / "differential_summary.json").read_text())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    # left: bacterial reduction + stereo vs baselines
    groups = ["reduction", "KR stereo"]
    mlp = [s["reduction"]["mlp"], s["stereo"]["mlp"]]
    maj = [s["reduction"]["majority"], s["stereo"]["majority"]]
    rule = [s["reduction"]["domain_rule"], s["stereo"]["majority"]]
    x = range(len(groups))
    a1.bar([i - 0.25 for i in x], mlp, 0.25, label="domain-MLP", color="#2b8cbe")
    a1.bar([i for i in x], rule, 0.25, label="domain-rule", color="#74a9cf")
    a1.bar([i + 0.25 for i in x], maj, 0.25, label="majority", color="#bdbdbd")
    a1.set_xticks(list(x))
    a1.set_xticklabels(groups)
    a1.set_ylabel("accuracy")
    a1.set_ylim(0, 1)
    a1.legend(fontsize=8)
    a1.set_title("Bacterial ClusterCAD (leave-cluster-out)")
    # right: fungal transfer
    t = s["transfer"]
    a2.bar(["active\ndomains", "constant\nsynthase domains"],
           [t["active_domains"], t["constant_domains"]], color=["#2b8cbe", "#cb181d"])
    a2.set_ylim(0, 1.05)
    a2.text(0, t["active_domains"] + 0.02, f"{t['active_domains']:.2f}", ha="center")
    a2.text(1, t["constant_domains"] + 0.02, f"{t['constant_domains']:.2f}", ha="center")
    a2.set_ylabel("per-cycle reduction accuracy")
    a2.set_title(f"Fungal transfer (n={t['n_cycles']}): the grammar wall")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_differential.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_reconstruction_tiers()
    fig_zstar()
    fig_corematch_sensitivity()
    fig_differential()
    print(f"wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
