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

.PHONY: release
release:
	@bash scripts/release.sh minor


.PHONY: hotfix
hotfix:
	@bash scripts/release.sh patch
