# lpi-fungal — Latent Program Induction for Iterative Fungal Synthases (dry pipeline)

Computational ("dry") research codebase for the paper *Latent Program Induction for
Iterative Fungal Synthases* (B. Eilertsen, 2026). This repository implements **Arm A**
only — everything that can be validated computationally without a wet lab:

- a **deterministic chemical executor** (the verifier core),
- a **verifier / program search** producing the verified consistent set `Z*(y)`,
- the **iteration controller** and training under the **verifier-constrained marginal
  likelihood**,
- baselines, ablations, and retrospective evaluation.

**Out of scope** (stubs / roadmap only): Arm B (causal activation pipeline),
bioactivity scoring, anything requiring wet-lab data.

## Phase roadmap

| Phase | Content | Gate |
|---|---|---|
| **0** (done — Gate MET) | MIBiG fungal-PKS ingestion + deterministic executor + curated validation | parquet built & pair count reported; executor round-trips ≥80% of the curated gate set; executor pure/deterministic + tested |
| **1** (done — Gate MET) | Beam-search verifier → `Z*(y)`, identifiability (`\|Z*(y)\|`) distribution | known program ∈ `Z*(y)` for every curated system |
| 2 | ESM-2 encoder + controller, verifier-constrained ML objective, baselines, ablations, eval | data-efficiency curve + ablation table + split-wise results, reproducible |

### Arm B / bioactivity (NOT built — roadmap only)
The causal activation pipeline (chromatin-mediated structural causal model, `do(·)`
elicitor ranking) and any bioactivity scoring are intentionally absent. They are
gated on a separate paired-epigenomic data audit (paper §5) and wet-lab data.

## Layout

```
src/lpi/
  chem/       program representation, thioester-sentinel molecule repr, canonicalization
  executor/   the deterministic executor (operators.py, core.py, cyclize.py, aromatic.py stub)
  data/       MIBiG ingestion, curated-set loader
  eval/       round-trip scoring + Gate 0 computation
  cli/        entry points (roundtrip, ingest)
  search/ model/ train/   Phase 1-2 (empty)
data/curated/ hand-curated validation programs (one YAML per system; see SCHEMA.md)
data/processed/fungal_pks_pairs.parquet   ingested fungal PKS pairs
tests/        executor unit/property/regression tests
```

## Setup

Requires Python ≥3.11 (a 3.12 venv is used here; the system 3.10 lacks RDKit wheels we
pin). RDKit + Phase 0 deps only — torch/ESM-2 are deferred to Phase 2 (`pip install -e .[ml]`).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e .          # installs pinned Phase 0 deps from pyproject.toml
```

## Reproduce Phase 0

```bash
make data        # download MIBiG 4.0 (if absent) + build the fungal PKS pairs parquet
make roundtrip   # curated round-trip + Gate 0 status table
make search      # verifier/search over curated set -> Z*(y) + Gate 1 status
make test        # executor + search unit/property/regression tests
make phase0      # data + roundtrip + test
make phase1      # search + test
```

### Manual data download (if `make data` cannot reach the network)
Download `mibig_json_4.0.tar.gz` from <https://dl.secondarymetabolites.org/mibig/> and
extract into `data/raw/` so that `data/raw/mibig_json_4.0/BGC*.json` exist, then:
```bash
.venv/bin/python -m lpi.cli.ingest --email you@example.com   # --offline skips NCBI lineage
```
Fungal classification uses NCBI taxonomy *lineage* (cached to
`data/raw/ncbi_lineage_cache.json`); `--offline` falls back to a genus heuristic.

## Status (Phase 0)
See [PHASE_0_REPORT.md](PHASE_0_REPORT.md) for the gate result, the usable pair count,
and what fails and why. Open chemistry calls for the domain expert are in
[QUESTIONS.md](QUESTIONS.md).
