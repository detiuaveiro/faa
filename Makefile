# Makefile
SHELL := /bin/bash
.PHONY: all clean venv
.DEFAULT_GOAL := all

# Python environment: a clean venv whose NumPy is compiled from source against the
# system OpenBLAS (found through pkg-config) and tuned to this CPU. Needs a C/C++
# compiler, pkg-config and the OpenBLAS headers (Debian/Ubuntu: libopenblas-dev).
VENV ?= venv
PYTHON ?= python3
NUMPY_FLAGS := -Csetup-args=-Dblas=openblas -Csetup-args=-Dlapack=openblas -Csetup-args=-Dcpu-baseline=native

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install --no-binary numpy $(NUMPY_FLAGS) "$$(grep -oE '"numpy[^"]*"' pyproject.toml | tr -d '"')"
	$(VENV)/bin/pip install .
	$(VENV)/bin/python -c "import numpy as np; b = np.show_config(mode='dicts')['Build Dependencies']['blas']; print('NumPy', np.__version__, 'BLAS:', b['name'], b['version'], b['lib directory'])"

all clean:
	@term_cols=$$(tput cols 2>/dev/null || echo 80); \
	total=$$(find . -mindepth 2 -name "Makefile" ! -path "./venv*" ! -path "*/volumes/*" ! -path "*/.git/*" | wc -l); \
	if [ "$$total" -eq 0 ]; then echo "No Makefiles found."; exit 0; fi; \
	current=0; \
	echo "Build started at $$(date)" > build.log; \
	tput civis || true; \
	find . -mindepth 2 -name "Makefile" ! -path "./venv*" ! -path "*/volumes/*" ! -path "*/.git/*" -print0 2>/dev/null | while IFS= read -r -d '' mkfile; do \
		dir=$$(dirname "$$mkfile"); \
		current=$$((current + 1)); \
		percent=$$((current * 100 / total)); \
		text_len=$$(( 11 + $${#current} + $${#total} + $${#dir} )); \
		width=$$(( term_cols - text_len )); \
		if [ $$width -lt 10 ]; then width=10; fi; \
		filled=$$((percent * width / 100)); \
		empty=$$((width - filled)); \
		if [ $$filled -gt 0 ]; then bar_filled=$$(printf "%$${filled}s" "" | tr " " "#"); else bar_filled=""; fi; \
		if [ $$empty -gt 0 ]; then bar_empty=$$(printf "%$${empty}s" "" | tr " " " "); else bar_empty=""; fi; \
		printf "\r%3d%%|%s%s| %d/%d [%s]\033[K" "$$percent" "$$bar_filled" "$$bar_empty" "$$current" "$$total" "$$dir"; \
		if ! $(MAKE) -s -C "$$dir" $@ >> build.log 2>&1; then \
			printf "\nError processing %s. See build.log\n" "$$dir"; \
		fi; \
	done; \
	echo ""; \
	tput cnorm || true; \
	echo "Done."
