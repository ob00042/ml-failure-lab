PYTHON ?= python

.PHONY: install lint test run check
install:
	$(PYTHON) -m pip install -e '.[dev]'
lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
test:
	$(PYTHON) -m pytest -q
run:
	$(PYTHON) -m failure_lab.runner run all
check: lint test run
