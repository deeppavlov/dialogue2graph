# Contributing to Dialog2Graph

Thank you for your interest in contributing to the Dialog2Graph project! We welcome contributions from the community to help improve and expand Dialog2Graph tool.

## Getting Started

1. Create a branch for your work. Preferable branch prefixes are `feat`, `fix`, `exp`.
2. Switch to your branch

```bash
git checkout <your_branch_name>
```

3. Set up the development environment (it is activated automatically)

```bash
poetry install --with docs,lint,tests
```

To delete all the virtual environments, run

```bash
poetry env remove --all
```

## Updating Dependencies

We use poetry as a dependency management tool. `poetry.lock` contains all dependencies for the current project. In order to update versions specified in the `poetry.lock`, run

```bash
poetry update
```

## How to Contribute

1. Experiment with the graphs (see the following section for a detail) or add new features to Dialog2Graph tool

2. Check linting and try to reformat your code running

    ```bash
    poetry run poe lint
    poetry run poe format
    ```

3. Create a pull request

**Tips:**

- choose a proper name for your pull request,
- add clear description of new features and fixes

## Setting Experiments

All conducted experiments should be stored in the `./experiments` folder, each experiment saved in the separate folder and named standardly, `exp<YYYY>_<MM>_<DD>_<hypothesis>`.

To make new experiment with automatic folder creation run

```bash
poetry new <experiment_name>
```

or, alternatively, if you've already created the folder, run

```bash
poetry init
```

**Note**: no images are allowed to put into folder. Please, consider using external links for attaching or using image files

## Coding Guidelines

- Follow PEP 8 style guide for Python code
- Write clear and concise comments
- Include docstrings for functions and classes
- Write unit tests for new features or bug fixes

## Pull Request Format

- Name of your PR (keep it simple yet meaningful)
- Short description (provide list of changes)

When having any doubts, you can simply create a draft PR and request a review before merging your request.

## Reporting Issues

If you encounter any bugs or have feature requests, please, open an issue on the GitHub. Provide as much detail as possible, including:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Graph visulisation if possible

## Current Focus Areas

We are currently working on supporting various types of graphs.

Supported graph types:

- [x] chain
- [x] single cycle
- [x] multi-cycle graph
- [x] complex graph with cycles

Currently unsupported graph types:

- [ ] single node cycle
