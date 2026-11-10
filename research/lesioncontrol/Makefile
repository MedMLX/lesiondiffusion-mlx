PYTHON := .venv/bin/python
RESEARCH := research/evaluate_adapter.py research/run_reliability.py research/analyze_reliability.py research/prepare_heldout.py research/qualify_heldout_geometry.py research/prepare_donor_extension.py

.PHONY: env qa test typecheck
env:
	bash environment/setup.sh

qa:
	.venv/bin/ruff check src tests $(RESEARCH)
	.venv/bin/ruff format --check src tests $(RESEARCH)
	$(MAKE) typecheck

typecheck:
	.venv/bin/pyright --project . --pythonpath .venv/bin/python --warnings

test:
	$(PYTHON) -m pytest $(TESTS) $(PYTEST_ARGS)
