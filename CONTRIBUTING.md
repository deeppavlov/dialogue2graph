# Contributing to chatsky-llm-integration

Thank you for your interest in contributing to the dff-llm-integration project! We welcome contributions from the community to help improve and expand this Chatsky LLM-Autoconfig tool.

## Getting Started

1. Create a branch for your work. Branch names should start with the goal they are created (`feat`, `fix` etc.)
2. Checkout to your branch

```bash
git checkout <your_branch_name>
```

3. Set up the development environment:

```bash
poetry install --with docs,lint,tests
```

The environment will be activated automatically.

If you want to delete all the virtual environments, run

```bash
poetry env remove --all
```

## Updating Dependencies

We use poetry.lock to ensure that all builds with the same lock file have the same 3rd-party library versions. This lets us know whether workflow fails due to our part or because a dependency update breaks something.

In order to update versions specified in poetry.lock, run

```bash
poetry update
```

## How to test your code

To test your algorithm or pipeline you need to follow these steps:

1. Decorate your class with AlgorithRegistry.register() like so:

```python
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry

@AlgorithmRegistry.register(input_type=BaseGraph, output_type=list[Dialogue])
class DialogueSampler(DialogueGenerator)
```

2. Make sure that `input_type` and `output_type` are matching with the signature of `.invoke()` method
3. Run

```bash
poetry run python dev_packages/chatsky_llm_autoconfig/chatsky_llm_autoconfig/autometrics/run_autometrics.py
```

4. If metrics didn't drop then everything is fine. Add, commit and push as usual.

## How to Contribute

1. Make your changes and test hypothesis in the `./experiments` folder as it is described in **Conducting experiments** section

3. Ensure linting using commands as

```
poetry run poe lint
poetry run poe format
```

4. Create a pull request with clear description of fixed and features

## Pull Request format

Please, include short description about your PR in it, give it a simple and inderstandable name.
You can always create a draft PR and request review before you request to merge it into main repository.

## Conducting experiments

Until any of the code make it way to the main repo it should be tested in `./experiments` folder.
Each of the experiments must lay in the separate folder with name like `<YYYY.MM.DD>_<experiment_name>`.
Inside of this directory must be a `report.md` file with results, metrics, future plans and other relevant information.

**!!! Do not put images into the folder you are commiting, use GoogleDrive instead !!!**

## Coding Guidelines

- Follow PEP 8 style guide for Python code
- Write clear and concise comments
- Include docstrings for functions and classes
- Write unit tests for new features or bug fixes

## Reporting Issues

If you encounter any bugs or have feature requests, please open an issue on the GitHub repository. Provide as much detail as possible, including:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Graph visulisation if possible

## Current Focus Areas

We are currently working on supporting various types of graphs. Here's the current status:

Supported types of graphs:

- [x] chain
- [x] single cycle

Currently unsupported types:

- [ ] single node cycle
- [ ] multi-cycle graph
- [ ] incomplete graph
- [ ] complex graph with cycles
