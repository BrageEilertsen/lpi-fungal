# Sound Biosynthetic Design over Iterative Grammars

**A Realizability Geometry for Fungal Polyketides** — Brage Eilertsen, University of Oslo

Reference implementation for the paper of the same name ([`paper/main.pdf`](paper/main.pdf)).

## Overview

Predicting a fungal natural product from its biosynthetic gene cluster (BGC) is usually posed as
direct `BGC → structure` regression, and it stalls (reported balanced accuracy 51–68%). This work
reframes the target: a product is the image `y = Exec_Θ(z; G)` of a latent biosynthetic *program*
`z`, executed under an operator library `Θ` and a domain-derived grammar `G`. Genome mining then
becomes *inference over `z`* from physical observables, and the field's ceiling is a statement about
grammar **identifiability** — how much of `z` the observables pin down — not about how much data
exists.

The repository implements a **sound inverse compiler** over this program space, run in three
directions:

- **Forward** — a deterministic, verifier-grounded executor renders a program to a structure.
- **Inverse** — a beam verifier returns the exact set `Z*` of programs consistent with the
  observables, as a typed verdict: `VERIFIED`, `UNDER-OBSERVED` (naming the next observable that
  would collapse `Z*`), or `OUT-OF-GRAMMAR`.
- **Cross-taxa** — a transfer probe locating where sequence context can substitute for a missing
  per-cycle observable.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # core executor + verifier (RDKit, pandas, Biopython)
pip install -e '.[hmmer]'   # optional: domain-annotation fallback (pyhmmer)
pip install -e '.[ml,dev]'  # optional: policy training (torch / transformers / ESM) + tests
```

Requires Python ≥ 3.11.

## Quickstart

```bash
make test          # run the test suite
make data          # download MIBiG 4.0 and ingest  (set EMAIL=you@inst.edu)
make roundtrip     # executor forward/inverse round-trip on the curated set
make search        # inverse enumeration of Z* per cluster
make reachability  # core-reachability verdicts across the corpus
```

See the `Makefile` for the full target list.

## Repository layout

```
src/lpi/
  executor/         deterministic forward executor (program → structure)
  grammars/         domain-derived grammars (PKS; extensible)
  search/           inverse enumeration of Z* (beam verifier)
  realizability.py  edit-distance geometry over the program space
  observe.py        observability layer (mass / adduct / MS-MS)
  planner/          experiment-selection planner
  model/ train/ eval/   weak-supervision policy (marginal-likelihood loss)
  engine.py         top-level inference entry point (infer_cluster)
  cli/              command-line drivers (ingest, roundtrip, search, …)
paper/              LaTeX source of the manuscript
scripts/            analysis and figure generation
tests/              pytest suite
data/               curated benchmark + derived inventories
```

## Citation

```bibtex
@misc{eilertsen2026realizability,
  title  = {Sound Biosynthetic Design over Iterative Grammars:
            A Realizability Geometry for Fungal Polyketides},
  author = {Eilertsen, Brage},
  year   = {2026},
  note   = {Preprint}
}
```

## License

Released under the MIT License.
