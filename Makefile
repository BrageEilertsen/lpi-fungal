PY ?= .venv/bin/python
EMAIL ?= brageei@uio.no
MIBIG_URL = https://dl.secondarymetabolites.org/mibig/mibig_json_4.0.tar.gz
MIBIG_DIR = data/raw/mibig_json_4.0

.PHONY: phase0 data roundtrip test mibig clean

phase0: data roundtrip test

mibig:
	@if [ ! -d "$(MIBIG_DIR)" ]; then \
		echo "downloading MIBiG 4.0..."; \
		cd data/raw && curl -sS -O "$(MIBIG_URL)" && tar -xzf mibig_json_4.0.tar.gz; \
	else echo "MIBiG already present at $(MIBIG_DIR)"; fi

data: mibig
	PYTHONPATH=src $(PY) -m lpi.cli.ingest --email $(EMAIL)

roundtrip:
	PYTHONPATH=src $(PY) -m lpi.cli.roundtrip

test:
	PYTHONPATH=src $(PY) -m pytest -q

clean:
	rm -rf .pytest_cache **/__pycache__
