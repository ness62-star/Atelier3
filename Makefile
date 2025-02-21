# Makefile for automating pipeline tasks
# Variables
PYTHON = python
VENV_ACTIVATE = call venv\Scripts\activate

# Installation of dependencies
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Code Quality Checks
check:
	$(VENV_ACTIVATE) && python -m flake8 . --max-line-length=100

# Data Preparation
prepare:
	$(VENV_ACTIVATE) && python main.py --prepare_data --sample_fraction 1.0

# Training with optimized parameters
train:
	$(VENV_ACTIVATE) && python main.py --train_file churn-bigml-80.csv --test_file churn-bigml-20.csv --max_depth 3 --sample_fraction 1.0

# Create test directory if it doesn't exist
create_test_dir:
	if not exist tests mkdir tests

# Run Tests
test: create_test_dir
	$(VENV_ACTIVATE) && pytest tests/

# Full CI/CD Pipeline
all: install check prepare train test
