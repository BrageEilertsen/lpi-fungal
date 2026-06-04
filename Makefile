PY ?= .venv/bin/python
EMAIL ?= your-email@example.com   # NCBI Entrez requires a contact email; override e.g. make data EMAIL=you@inst.edu
MIBIG_URL = https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz
MIBIG_DIR = data/raw/mibig_json_4.0

.PHONY: phase0 phase1 data roundtrip search test mibig crossgrammar observability realdemo seqhead clean

phase0: data roundtrip test
phase1: search test

mibig:
	@if [ ! -d "$(MIBIG_DIR)" ]; then \
		echo "downloading MIBiG 4.0..."; \
		cd data/raw && curl -sS -O "$(MIBIG_URL)" && tar -xzf mibig_json_4.0.tar.gz; \
	else echo "MIBiG already present at $(MIBIG_DIR)"; fi

data: mibig
	PYTHONPATH=src $(PY) -m lpi.cli.ingest --email $(EMAIL)

roundtrip:
	PYTHONPATH=src $(PY) -m lpi.cli.roundtrip

search:
	PYTHONPATH=src $(PY) -m lpi.cli.search

# Reproducibility: pin PYTHONHASHSEED on the beam-scan targets so their unreachable/budget
# split is bit-reproducible from one command. The search enumeration is already hash-independent
# (frozen-dataclass programs, repr-tiebreak beam sort, no set/dict iteration in the scan path),
# so this is defensive determinism hygiene -- it does NOT by itself prevent stale-artifact drift:
# a committed log/csv must still be regenerated when the executor grammar (e.g. the release set) changes.
reachability:
	PYTHONHASHSEED=0 PYTHONPATH=src $(PY) -m lpi.cli.reachability

corematch:
	PYTHONPATH=src $(PY) -m lpi.cli.corematch

phase0c:
	PYTHONPATH=src $(PY) -m lpi.cli.phase0c

clustercad:
	PYTHONPATH=src $(PY) -m lpi.data.clustercad

differential:
	PYTHONPATH=src $(PY) -m lpi.cli.differential

figures:
	PYTHONPATH=src $(PY) scripts/figures.py

crossgrammar:
	PYTHONPATH=src $(PY) scripts/cross_grammar_demo.py

observability:
	PYTHONPATH=src $(PY) scripts/observability_demo.py

realdemo:
	PYTHONPATH=src $(PY) scripts/real_demo_6msa.py

test:
	PYTHONPATH=src $(PY) -m pytest -q

clean:
	rm -rf .pytest_cache **/__pycache__

seqhead:
	PYTHONPATH=src $(PY) scripts/explorations/bacterial_esm_demo.py > results/seqhead.log

census:	## re-scans the 94 too_large cores; PYTHONHASHSEED pinned (see reproducibility note above reachability)
	PYTHONHASHSEED=0 PYTHONPATH=src $(PY) scripts/coverage/oog_census.py

coverage-sweep:
	PYTHONPATH=src $(PY) scripts/coverage/theta_sweep.py

kappa-check:
	PYTHONPATH=src $(PY) scripts/coverage/kappa_monotonicity.py

design-tag:
	PYTHONPATH=src $(PY) scripts/coverage/design_tag.py

structural-rescue:
	PYTHONPATH=src $(PY) scripts/policy/render_structural_rescue_figure.py

two-walls:
	PYTHONPATH=src $(PY) scripts/two_walls.py

gate-frontier:
	PYTHONPATH=src $(PY) scripts/policy/gate_frontier_dedup.py

design-sensitivity:
	PYTHONPATH=src $(PY) scripts/coverage/design_sensitivity.py

kappa-certificate: kappa-check	## alias: reproduces the beta=8000->5e4 kappa inversion-and-restore (Prop 16.4c)

homology-stratify:
	PYTHONPATH=src $(PY) scripts/coverage/homology_stratify.py

b1:	## B1 (decisive): does substrate state s_t clear the 0.567 wall? (faithful (a)-vs-(b) harness)
	PYTHONPATH=src $(PY) scripts/policy/b1_substrate_wall.py

b3:	## B3: free-label scaling law -- does decision count (not cluster count) drive accuracy?
	PYTHONPATH=src $(PY) scripts/policy/b3_free_label_scaling.py

duality:	## Theorem 1 property test: verifier Z* == independent O-consistent set D(y;O), all grammars
	PYTHONPATH=src $(PY) scripts/duality_regression.py

coverage-illusion:	## Exp C survivorship control: is C-methyl burden enriched past the reachability boundary at matched C?
	PYTHONPATH=src $(PY) scripts/coverage/coverage_illusion.py

generability:	## Sec 10.1: classify the in-scope-unreachable cores -- search-limited (beam) vs coverage-limited (chemistry). PYTHONHASHSEED pinned (see reachability note above).
	PYTHONHASHSEED=0 PYTHONPATH=src $(PY) scripts/coverage/generability_scan.py

THETA ?= 0   # build-loop Theta level: 0 = cited baseline (117/11); 1 = +PT_NAPHTHALENE (114/14); see forward_index.py
forward-index:	## Completion 1: forward SOUND index at Theta level $(THETA). `make forward-index THETA=1` runs the build-loop extension. PYTHONHASHSEED pinned.
	THETA=$(THETA) PYTHONHASHSEED=0 PYTHONPATH=src $(PY) scripts/coverage/forward_index.py

forward-index-monotone:	## R_term-monotonicity test: across the Theta levels already run, assert gaps only shrink + recovered only grows + zero demotions (Prop 16.4c as a runnable check)
	PYTHONHASHSEED=0 PYTHONPATH=src $(PY) scripts/coverage/forward_index.py --monotone

rung2-cover:	## Completion 1 build-queue: rung-2 sound set-cover over the 117 certified gaps -- which single soundly-expressible operator buys the most provable reach (reads results/forward_index.csv).
	PYTHONHASHSEED=0 PYTHONPATH=src $(PY) scripts/coverage/rung2_cover.py
