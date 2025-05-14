# Dialog2Graph

Dialog2Graph allows you to effortlessly create *chatsky flows* and scripts from dialogs using Large Language Models.

## Contents

```
./dialog2graph - source code
./examples - usage scenarios
./experiments - test field for conducting experiments
./prompt_cache - utils for LLM output caching
./scripts - scripts for `poethepoet` automation 
```

## Current Progress

Supported graph types:

- [x]  chain
- [x]  single cycle
- [x]  multi-cycle graph
- [x]  complex graph with cycles

Currently unsupported graph types:

- [ ]  single node cycle

## How to Start

Install poetry v. 1.8.4 ([detailed installation guide](https://python-poetry.org/docs/))

```bash
pipx install poetry==1.8.4
```

Clone this repo and install project dependencies

```bash
git clone https://github.com/deeppavlov/dialog2graph.git
cd dialog2graph
poetry install
```

Consider installing **PyGraphviz** from [here](https://pygraphviz.github.io/), if you are planning to visualize your graphs. Then add it to the poetry environment.

```bash
poetry add pygraphviz
```

Ensure that dependencies were installed correctly by running any Python script

```bash
poetry run python <your_file_name>.py
```

Create `.env` file to store credentials

**Note:** never hardcode your personal tokens and other sensitive credentials. Use the `.env` file to store them.

## How to Use

See `dialog2graph` usage examples:

1. [Learn base classes usage](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/basics/base_classes_usage.ipynb)

2. [Learn dialog2graph pipelines and model configuration](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/pipeline_usage/pipeline_example.ipynb)

3. [Generate dialog graph using LLMs](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/data_generation/LoopedGraphGenerator_example.ipynb)

4. [Evaluate graph and dialogs](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/evaluation/examples_of_metrics_usage.ipynb)

5. [Learn CLI interface](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/cli_usage/main.ipynb)

6. [Augment dialogs on dialog graph](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/dialog_augmentation/dialogue_augmentation_example.ipynb)

7. [Extend a dialog graph](https://github.com/deeppavlov/dialog2graph/blob/dev/examples/dialog_extension/dialog_extender_example.ipynb)

## How to Contribute

See contribution guideline [CONTRIBUTING.md](https://github.com/deeppavlov/dialog2graph/blob/dev/CONTRIBUTING.md)
