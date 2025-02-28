# chatsky-llm-integration

Chatsky LLM-Autoconfig allows you to effortlessly create chatsky flows and scripts from dialogues using Large Language Models.

## Requirements

Python >=3.9, <3.12
Poetry

## How to start?

Clone this repo and run `poetry install` to install all dependencies

```bash
git clone https://github.com/deeppavlov/chatsky-llm-autoconfig.git
cd chatsky-llm-autoconfig
poetry install
```
Create `.env` file for credentials. 

**Note:** !!! Put your tokens and other sensitive credentials only in `.env` files and never hardcode them !!!

Try to run some scripts or previous experiments to see if everything is working as expected. To run python file using poetry run the following:

```bash
poetry run python <your_file_name>.py
```

## Contents

```
./experiments - Test field for experimental features, test data and results
./scripts - Here we put scripts needed for `poethepoet` automation (you probably do not need to look inside)
./dialogue2graph - Directory containing all the code for the `dialogue2graph` module
```

## Current progress

Supported types of graphs:

- [x]  chain
- [x]  single cycle
- [x]  multi-cycle graph
- [x]  complex graph with cycles

Currently unsupported types:

- [ ]  single node cycle

## How to use?

We provide several of using our library for various tasks.

### Data generation

Get dialogue on certain topics using LLMs.

```python
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
from langchain_openai import ChatOpenAI


generation_model = ChatOpenAI(
        model='o1-mini',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.2
)
    
validation_model = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0
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

result = pipeline.invoke(topic='account information change', use_cache=False)
```

### Dialogue sampling

Sample existing dialogs from graph

```python
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from chatsky_llm_autoconfig.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogueSampler()
sampler.invoke(graph=G) #-> list of Dialogue objects

```

## How to contribute?

You can find contribution guideline in [CONTRIBUTING.md](https://github.com/deeppavlov/chatsky-llm-autoconfig/blob/main/CONTRIBUTING.md)
