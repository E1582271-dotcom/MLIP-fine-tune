PYTHON ?= python
# Run from a checkout without installing: the package is importable from src/.
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))

.PHONY: all tables figures docs test check lint ingest

all: tables figures docs

tables:
	$(PYTHON) -m mlipft tables

figures:
	$(PYTHON) -m mlipft figures

docs:
	$(PYTHON) -m mlipft docs

test:
	$(PYTHON) -m pytest -m "not ingest"

lint:
	ruff check src tests generation

# Rebuild everything from the committed data, run the tests and the repository checks, and require
# an unchanged working tree.
check: all test
	$(PYTHON) -m mlipft check-repo
	git diff --exit-code -- data results figures docs README.md

# Local only: re-derive data/ from the raw backup (MLIPFT_RAW) with the ingest environment
# (requirements-ingest.lock.txt). KMP_DUPLICATE_LIB_OK avoids an OpenMP clash of torch and numpy.
ingest:
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 $(PYTHON) -m mlipft ingest dft
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 $(PYTHON) -m mlipft ingest models
	KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 $(PYTHON) -m mlipft ingest trajectories --workers 4
	$(PYTHON) -m mlipft ingest collect
	$(PYTHON) -m mlipft ingest manifest
