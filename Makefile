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

reachability:
	PYTHONPATH=src $(PY) -m lpi.cli.reachability

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
