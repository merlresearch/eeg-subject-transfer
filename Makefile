# Copyright (C) Mitsubishi Electric Research Labs (MERL) 2023
#
# SPDX-License-Identifier: AGPL-3.0-or-later
SHELL=/bin/bash

# global variables
SRC_NAME=src
SCRIPTS_NAME=scripts
BASE_PYTHON=python3.8    # Edit to use another python version or path
VENV_NAME=venv
PYTHON=$(VENV_NAME)/bin/python3
EGG=$(SRC_NAME).egg-info

all: setup lint

test: setup
	$(VENV_NAME)/bin/pytest

lint: setup
	$(VENV_NAME)/bin/pre-commit run --all-files

# Create virtualenv and install project and dev dependencies.
# Use *.egg-info as a concrete output, so that `make` will actually check whether this target is up-to-date
.PHONY: setup
setup: $(EGG)

$(EGG): $(PYTHON) setup.py requirements.txt requirements-dev.txt
	$(PYTHON) -m pip install -U pip setuptools wheel
	$(PYTHON) -m pip install --extra-index-url https://download.pytorch.org/whl/cu113 -Ue .
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(VENV_NAME)/bin/pre-commit install

$(PYTHON):
	$(BASE_PYTHON) -m pip install virtualenv
	$(BASE_PYTHON) -m virtualenv $(VENV_NAME)

# Careful!
# useful e.g. after modifying __init__.py files
destroy-setup: confirm
	rm -rf $(EGG)
	rm -rf $(VENV_NAME)

.PHONY: confirm
confirm:
	@( read -p "Confirm? [y/n]: " sure && case "$$sure" in [yY]) true;; *) false;; esac )
