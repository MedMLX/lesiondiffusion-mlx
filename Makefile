.DEFAULT_GOAL := qa
.PHONY: env check-env lint typecheck test qa loc clean

TESTS ?= tests
PYTEST_ARGS ?=
PYTHON ?= .venv/bin/python
SOURCE_DIRS ?= scripts src tests typings
GENERATED_PATHS ?= .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
export PYTHONNOUSERSITE := 1

env:
	uv lock --check
	uv sync --locked --extra dev

check-env:
	@command -v "$(PYTHON)" >/dev/null || { echo "Missing project Python: run make env explicitly, or set PYTHON to its existing interpreter" >&2; exit 1; }

lint: check-env
	$(PYTHON) -m ruff check .

typecheck: check-env
	@command -v node >/dev/null || { echo "Maintained host Node is required for Pyright" >&2; exit 1; }
	PYRIGHT_PYTHON_NODEJS_WHEEL=0 PYRIGHT_PYTHON_GLOBAL_NODE=1 $(PYTHON) -m pyright --project . --pythonpath "$(PYTHON)" --warnings
	bash -o pipefail -c 'find scripts src tests typings -type f \( -name "*.py" -o -name "*.pyi" \) -print0 | xargs -0 $(PYTHON) -m ruff check --config pyproject.toml --select TID251 --ignore-noqa --no-force-exclude --'

test: check-env
	$(PYTHON) -m pytest $(TESTS) $(PYTEST_ARGS)

qa: lint typecheck

loc:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	list=$$(mktemp); trap 'rm -f "$$list"' EXIT; \
	for dir in $(SOURCE_DIRS); do \
		if [ -e "$$dir" ]; then find "./$$dir" -type f \
			\( -name '*.py' -o -name '*.pyi' -o -name '*.sh' -o -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.rs' \) -print0 >> "$$list"; fi; \
	done; test -s "$$list" || { echo "No source files found"; exit 1; }; xargs -0 wc -l < "$$list"

clean:
	@set -eu; . ./devtools/repo-paths.sh; \
	validate_repo_paths GENERATED_PATHS $(GENERATED_PATHS); \
	validate_repo_paths SOURCE_DIRS $(SOURCE_DIRS); \
	for path in $(GENERATED_PATHS); do \
		rm -rf -- "$$path"; \
	done; for dir in $(SOURCE_DIRS); do \
		if [ -d "$$dir" ]; then find "./$$dir" -type d -name __pycache__ -prune -exec rm -rf {} +; fi; \
	done
