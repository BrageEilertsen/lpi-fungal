# A Biosynthetic Inverse Compiler for Verifier-Grounded Fungal Genome Mining

Open code for the paper *A Biosynthetic Inverse Compiler for Verifier-Grounded Fungal Genome Mining*
(subtitle: *The reconstructibility ceiling in fungal polyketide chemistry*; B. Eilertsen, 2026).
See [`paper/main.pdf`](paper/main.pdf).

A biosynthetic gene cluster (BGC) is not a molecular blueprint but a **constrained latent program
space**: a product is the image `y = Exec_Θ(z; G)` of a biosynthetic program `z` under an operator
library `Θ` and a domain-derived grammar `G`. Genome mining is then *inference over `z`* from physical
observables, not direct `BGC → structure` regression. This repo implements that as a
**grammar-agnostic inverse compiler**:

```
BGC / domain alphabet G
        ↓   grammar compiler  (PKS · NRPS · hybrid)        <- family-specific
   candidate programs z
        ↓   sound executor  Exec_Θ(z; G)
   candidate structures
        ↓   observability layer  (mass · adduct · ppm · MS/MS)   <- SHARED across grammars
   three-state verdict:  VERIFIED · UNDER-OBSERVED · OUT-OF-GRAMMAR
```

Only the grammar/executor is family-specific; the mass/MS-MS observability layer and the
reconstructibility verdict are **shared**, and read only candidate structures (SMILES). The single
entry point is `lpi.engine.infer_cluster(domains, observables, grammar=...)`.

## Milestones (tags on branch `grammar-generalization`)

| tag | what |
|---|---|
| `lpi-v0.2-cross-grammar` | grammar-agnostic `infer()` + NRPS v0 grammar + first-class three-state verdict |
| `lpi-v0.3-observability` | adduct/ppm-tolerant approximate verifier `Z*_ε` + MS/MS scorer + minimum-sufficient-observables planner |
| `lpi-v0.4-hybrid` | typed PKS→NRPS hybrid bridge (`z_hybrid = z_PKS ∘ h_handoff ∘ z_NRPS ∘ r_release`) |
| `lpi-v0.5-reframe` | paper reframed as a grammar-agnostic inverse-compiler framework |
| `lpi-v0.6-real-mining` | blind real-cluster (6-MSA) → accurate-mass → VERIFIED case study |

The earlier empirical core (executor, beam verifier, the MIBiG reconstructibility analysis, the
differential probe) precedes these tags on the same branch.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .            # RDKit + core deps; .[ml] adds torch/transformers
```
All commands assume `PYTHONPATH=src` (the `Makefile` sets it for you).

## Reproduce

Each result maps to one command:

```bash
make data            # MIBiG 4.0 ingest -> 326 fungal PKS pairs
make roundtrip       # curated executor round-trip
make search          # beam verifier -> Z*(y)
make reachability    # exact-match reconstruction: 8/326
make corematch       # core-match sensitivity (a deliberately unreliable proxy: 8 -> 160)
make phase0c         # tailoring-latent reconstruction: 7/106 (64% conditional on a producible core)
make differential    # ΔBGC -> Δstructure transfer probe (1.00 active-domains -> 0.57 fixed-domain)
make crossgrammar    # PKS + NRPS + hybrid through one infer_cluster() interface
make observability   # candidate-collapse benchmark + ppm/τ sensitivity sweep
make realdemo        # blind real-cluster 6-MSA mining demo (MIBiG BGC0001275)
make figures         # paper figures from committed CSV/JSON
make test            # full suite (116 tests)
```

The real-mining demo's full data provenance (cluster, mass, the EI sanity-reference spectrum) is in
[`results/real_demo_6msa_sources.md`](results/real_demo_6msa_sources.md).

## Layout

```
src/lpi/
  engine.py        the observable-constrained inference interface + three-state verdict
  observe.py       shared metabolomics observability layer (adducts, ppm Z*_ε, MS/MS, planner)
  grammars/        base.Grammar protocol; pks / nrps / hybrid grammars (the family-specific half)
  executor/        sound deterministic executors: core.py (PKS), nrps.py, hybrid.py (+ operators)
  chem/            program/peptide representations, molecule repr, formula enumeration
  search/          beam verifier + the PKS candidate generator
  eval/ data/ model/ cli/
data/curated/      hand-curated literature programs (one YAML per system)
scripts/           runnable demos (cross_grammar, observability, real_demo_6msa, ...)
tests/             executor/grammar/observability/hybrid/real-demo tests (pure, deterministic)
paper/main.tex     the manuscript
```

## Scope & honesty

- **PKS** is the developed grammar; **NRPS and hybrid are v0 demonstrations of architecture
  generality, not full family coverage** (a small monomer set, achiral, the tetramic-acid release a
  typed-but-unimplemented operator). The point is that a second/third grammar plugs into the *same*
  observability and verdict layers.
- The mining pipeline has **one real-cluster, real-mass blind case study** (6-MSA); a multi-cluster
  benchmark and real LC-MS/MS validation remain future work. No MS/MS peaks are simulated for any
  real-validation claim.
- All guarantees are **relative** to the operator library, grammar, observables, mass tolerance, and
  adduct model (see the paper's Guarantees subsection).

Open chemistry calls for the domain expert are in [QUESTIONS.md](QUESTIONS.md).
