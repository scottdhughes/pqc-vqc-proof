PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PY := $(BIN)/python
PIP := $(BIN)/pip

.PHONY: venv install local-proof proof hardware-proof clean

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(PIP) install -r requirements.txt

train:
	$(PY) vqc_benchmark.py --mode train --output results/vqc_benchmark.json

local-proof:
	$(PY) vqc_benchmark.py --mode train --output results/vqc_benchmark.json

proof:
	$(PY) vqc_benchmark.py --mode proof --params-file results/vqc_benchmark.json --proof-repetitions 3 --output results/vqc_proof_hardware.json

clean:
	rm -rf __pycache__ results/*.json
