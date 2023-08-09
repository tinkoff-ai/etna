# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help clean copy-tutorials html build-docs Makefile

clean:
	@rm -rf source/api
	@rm -rf build/
	@rm -rf "$(SOURCEDIR)/tutorials"

copy-tutorials:
	@mkdir -p "$(SOURCEDIR)/tutorials"
	@cp ../examples/*.ipynb "$(SOURCEDIR)/tutorials"
	@cp -r ../examples/assets "$(SOURCEDIR)/tutorials/assets"

html:
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

build-docs: clean copy-tutorials html
