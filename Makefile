lint: isort-check black-check flake8-check mypy-check spell-check imported-deps-check

isort-check:
	isort --skip etna/libs --sl -c etna/
	isort --skip etna/libs --sl -c tests/

black-check:
	black --check etna/
	black --check tests/

flake8-check:
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N --ignore C400,C401,C402,C403,C404,C405,C406,C408,C409,C410,C411,C412,C413,C414,C415,C416,F,E203,W605,E501,W503,D100,D104

mypy-check:
	mypy

spell-check:
	codespell etna/ *.md tests/ -L mape,hist
	python -m scripts.notebook_codespell

imported-deps-check:
	python -m scripts.check_imported_dependencies

format:
	isort --skip etna/libs --sl etna/
	isort --skip etna/libs --sl tests/
	black etna/
	black tests/
	flake8 --exclude etna/libs etna/
	flake8 --exclude etna/libs tests/ --select E,W,C,F401,N --ignore C400,C401,C402,C403,C404,C405,C406,C408,C409,C410,C411,C412,C413,C414,C415,C416,F,E203,W605,E501,W503,D100,D104
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
