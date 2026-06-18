MAIN = src/main.py
MYPY_FLAGS = --warn-return-any --warn-unused-ignores --ignore-missing-imports \
--disallow-untyped-defs --check-untyped-defs
CACHES = __pycache__ .mypy_cache .ruff_cache

run:
	uv run python $(MAIN) $(ARGS)

install:
	uv sync

debug:
	uv run python -m pdb $(MAIN) $(ARGS)

clean:
	$(foreach cache, $(CACHES), @rm -rf $$(find . -type d -name "$(cache)"))
	@echo temporary files and caches deleted

format:
	uv run ruff format

lint:
	uv run python -m flake8 src
	uv run python -m mypy src $(MYPY_FLAGS)

lint-strict:
	uv run python -m flake8 src
	uv run python -m mypy src --strict

.PHONY: install run debug clean lint lint-strict 
