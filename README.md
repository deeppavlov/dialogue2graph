# chatsky-llm-integration

Chatsky LLM-Autoconfig allows you to effortlessly create chatsky flows and scripts from dialogues using Large Language Models.

## How to start?

You can simply clone this repo and run poetry install to install all dependencies

```bash
git clone https://github.com/deeppavlov/chatsky-llm-autoconfig.git
cd chatsky-llm-autoconfig
poetry install
```

Now you can try to run some scripts or previous experiments to see if everything is working as expected.

To run python file using poetry run the following:

```bash
poetry run python <your_file_name>.py
```

**!!! Put your tokens and other sensitive credentials only in `.env` files and never hardcode them !!!**

## Contents

```
./data - Examples, tests and other dialogue data in JSON format
./experiments - Test field for experimental features, test data and results
./scripts - Here we put scripts needed for `poethepoet` automation (you probably do not need to look inside)
./dev_packages/chatsky_llm_autoconfig - Directory containing all the code for the `chatsky_llm_autoconfig` module
```

## Current progress

Supported types of graphs:

- [x]  chain
- [x]  single cycle

Currently unsupported types:

- [ ]  single node cycle
- [ ]  multi-cycle graph
- [ ]  incomplete graph
- [ ]  complex graph with cycles

## Current algorithms progress

- Generate graph from scratch by topic (input: topic, output: graph) (for dataset generation)
  - algorithms.topic_graph_generation.CycleGraphGenerator

- change graph without changing graph structure (input: old graph + topic, output: new graph) (for dataset generation)

- sampling from dialogue graph (input: graph, output: dialogue) (for dataset generation)
  - algorithms.dialogue_generation.DialogueSampler

- augmentation of dialogue (input: dialogue, output: dialogue) (for dataset generation)
  - algorithms.dialogue_generation.DialogAugmentation

- generate graph from scratch by dialogue (input: dialogue, output: graph) (decisive algorithm)
  - GeneralGraphGenerator (experiments/2024.11.14_dialogue2graph)

- generate graph based on existing graph (input: old graph + dialog, output: new graph) (extended decision algorithm)
  - AppendChain (experiments/2024.11.17_concatenating_subchains_prompt)

## How to contribute?

You can find contribution guideline in [CONTRIBUTING.md](https://github.com/deeppavlov/chatsky-llm-autoconfig/blob/main/CONTRIBUTING.md)
