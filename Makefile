lint:
	isort --sl -c etna/
	isort --sl -c tests/
	black --check etna/
	black --check tests/
	flake8 etna/
	flake8 tests/ --select E,W,C,F401,N
	mypy

format:
	isort --sl etna/
	isort --sl tests/
	black etna/
	black tests/
	flake8 etna/
	flake8 tests/ --select E,W,C,F401,N
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
