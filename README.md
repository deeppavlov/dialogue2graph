# Dialogue2Graph

Dialogue2Graph allows you to effortlessly create *chatsky flows* and scripts from dialogues using Large Language Models.

## Contents

```
./dialogue2graph - source code
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
git clone https://github.com/deeppavlov/dialogue2graph.git
cd dialogue2graph
poetry install
```

If you are planning to visualize your graphs consider installing **PyGraphviz** from [here](https://pygraphviz.github.io/) and also add it to the poetry environment.

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

### Generate synthetic graph on certain topic

Choose LLMs for generating and validating dialogue graph and invoke graph generation

```python
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
from langchain_community.chat_models import ChatOpenAI


gen_model = ChatOpenAI(
    model='gpt-4o',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)
val_model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0,
)

pipeline = LoopedGraphGenerator(
    generation_model=gen_model,
    validation_model=val_model,
)

generated_graph = pipeline.invoke(topic="restaurant reservation")
```

### Sample dialogues from existing dialogue graph

Create graph instance and invoke sampler to get dialogue list

```python
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.pipelines.core.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogueSampler()
sampler.invoke(graph=G) #-> list of Dialogue objects
```

## How to Contribute

See contribution guideline [CONTRIBUTING.md](https://github.com/deeppavlov/dialogue2graph/blob/dev/CONTRIBUTING.md)
