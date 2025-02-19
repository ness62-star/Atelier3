# Makefile for automating pipeline tasks

# Variables
PYTHON = python
VENV_ACTIVATE = venv\Scripts\activate

# Installation of dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Code Quality Checks (formatting & security)
check:
	flake8 . --max-line-length=100
	bandit -r .

# Data Preparation
prepare:
	$(PYTHON) main.py --prepare_data

# Model Training
train:
	$(PYTHON) main.py --train

# Run Tests
test:
	pytest tests/

# Full CI/CD Pipeline
all: install check prepare train test
