# SigCLR
A signal CLR architecture

## Tasks

This repository is [xc](https://xcfile.dev) compliant. The following tasks are available:

### format

Format all python files found in the repository. It will overwrite the files.

```bash
sort .
black .
```

### lint

Lint all python files found in the repository.

```bash
mypy .
flake8 .
```

### test

Run all tests found in the tests directory. Extra arguments can be passed to pytest.

```bash
poetry run pytest $@
```

### install

Install this python package.

```bash
poetry install $@
```
