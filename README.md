# chatsky-llm-integration

Chatsky LLM-Autoconfig allows you to effortlessly create **chatsky flows** and scripts from dialogues using Large Language Models.

## Contents

```
./dialogue2graph - source code
./experiments - test field for discovering project features, contributing experimental features, analysing test data and results
./scripts - scripts for `poethepoet` automation 
```

## Current progress

Supported graph types:

- [x]  chain
- [x]  single cycle
- [x]  multi-cycle graph
- [x]  complex graph with cycles

Currently unsupported graph types:

- [ ]  single node cycle

## How to start

Install [poetry](https://python-poetry.org/docs/)

Сlone this repo and install project dependencies:

```bash
git clone https://github.com/deeppavlov/chatsky-llm-autoconfig.git
cd chatsky-llm-autoconfig
poetry install
```

Ensure that dependencies were installed correctly by running any python script:

```bash
poetry run python <your_file_name>.py
```

Create `.env` file to store credentials

**Note:** never hardcode your personal tokens and other sensitive credentials. Use the `.env` file to store them.

## How to use

### Generate synthetic graph on certain topic

Choose LLMs for generating and validating dialogue graph and invoke graph generation

```python
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
from langchain_openai import ChatOpenAI


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
    
topics = [
    "technical support conversation",
    "job interview",
    "restaurant reservation",
    "online shopping checkout",
    "travel booking"
]

generated_graph = pipeline.invoke(topic=topics[0])
```

### Sample dialogues from existing dialogue graph

Create graph instance and invoke sampler to get dialogue list

```python
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from chatsky_llm_autoconfig.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogueSampler()
sampler.invoke(graph=G) #-> list of Dialogue objects
```

## How to contribute?

See contribution guideline [CONTRIBUTING.md](https://github.com/deeppavlov/chatsky-llm-autoconfig/blob/main/CONTRIBUTING.md)
