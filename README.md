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

Choose LLMs for generating and validating dialog graph and invoke graph generation

```python
from dialog2graph.datasets.complex_dialogs.generation import LoopedGraphGenerator
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

### Sample dialogs from existing dialog graph

Create graph instance and invoke sampler to get dialog list

```python
from dialog2graph.pipelines.core.dialog_sampling import RecursiveDialogSampler
from dialog2graph.pipelines.core.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogSampler()
sampler.invoke(graph=G) #-> list of Dialog objects
```

## How to Contribute

See contribution guideline [CONTRIBUTING.md](https://github.com/deeppavlov/dialog2graph/blob/dev/CONTRIBUTING.md)
