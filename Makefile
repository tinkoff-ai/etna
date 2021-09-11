lint:
	isort --sl -c etna/
	isort --sl -c tests/
	flake8 etna/core etna/models
	flake8 tests/ --select E,W,C,F401,N
	mypy etna/core etna/models --allow-redefinition

format:
	isort --sl etna/
	isort --sl tests/
	flake8 etna/core etna/models
	flake8 tests/ --select E,W,C,F401,N
	mypy etna/core etna/models --allow-redefinition

.PHONY: release
release:
	@bash scripts/release.sh minor


.PHONY: hotfix
hotfix:
	@bash scripts/release.sh patch
