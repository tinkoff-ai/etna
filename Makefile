lint:
	isort --skip etna/libs --sl -c etna/
	isort --skip etna/libs --sl -c tests/
	black --check etna/
	black --check tests/
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N
	mypy --config-file=mypy.ini

format:
	isort --skip etna/libs --sl etna/
	isort --skip etna/libs --sl tests/
	black etna/
	black tests/
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N
	mypy --config-file=mypy.iny

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
