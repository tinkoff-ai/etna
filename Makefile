lint: isort-check black-check flake8-check mypy-check spell-check imported-deps-check notebooks-check

isort-check:
	isort --skip etna/libs --sl -c etna/
	isort --skip etna/libs --sl -c tests/

black-check:
	black --check etna/
	black --check tests/

flake8-check:
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N

mypy-check:
	mypy

spell-check:
	codespell etna/ *.md tests/ -L mape,hist
	python -m scripts.notebook_codespell

imported-deps-check:
	python -m scripts.check_imported_dependencies

notebooks-check:
	python black --check examples/*.ipynb

format:
	isort --skip etna/libs --sl etna/
	isort --skip etna/libs --sl tests/
	black etna/
	black tests/
	black examples/*.ipynb
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N
	mypy

.PHONY: deps/release
deps/release:
	@poetry install -E release

.PHONY: release/prerelease
release/prerelease:
	@poetry run python -m scripts.release prerelease

.PHONY: release/prepatch
release/prepatch:
	@poetry run python -m scripts.release prepatch

.PHONY: release/preminor
release/preminor:
	@poetry run python -m scripts.release preminor

.PHONY: release/patch
release/patch:
	@poetry run python -m scripts.release patch

.PHONY: release/minor
release/minor:
	@poetry run python -m scripts.release minor
