# Pytest Usage Guide

This document provides a practical overview of common `pytest` commands and workflows.

## Requirements

- Python 3.x
- pytest ≥ 6

Install pytest if needed:

```bash
pip install pytest
# or
poetry add pytest
```

## Running Tests
### Run all tests

```sh
pytest
```

### Run tests in a specific file

```sh
pytest tests/test_feature_extractors.py
```

### Run tests in a directory

```sh
pytest tests/
```

## Verbose Output and Debugging
### Show detailed test results

```sh
pytest -v
```

### Show print() output (disable output capture)

```sh
pytest -s
```

### Common combination (recommended during development)

```sh
pytest -sv
```

## Running Specific Tests
### Run a single test function

```sh
pytest -sv tests/core/test_feature_extractors.py::test_basic
```

### Run a test inside a test class

```sh
pytest tests/test_model.py::TestModel::test_forward
```
