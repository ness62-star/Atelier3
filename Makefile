# Makefile for automating pipeline tasks

# Variables
PYTHON = python
VENV_ACTIVATE = call venv\Scripts\activate

# Installation of dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Code Quality Checks (formatting & security)
check:
	$(VENV_ACTIVATE) && python -m flake8 . --max-line-length=100

# Data Preparation
prepare:
	$(VENV_ACTIVATE) && python main.py --prepare_data

train:
	$(VENV_ACTIVATE) && python main.py --train_file churn-bigml-80.csv --test_file churn-bigml-20.csv --max_depth 10

# Run Tests
test:
	$(VENV_ACTIVATE) && pytest tests/

# Full CI/CD Pipeline
all: install check prepare train test
