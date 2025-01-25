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
- [x]  multi-cycle graph
- [x]  complex graph with cycles

Currently unsupported types:

- [ ]  single node cycle
- [ ]  incomplete graph

## How to use
We provide several of using our library for various tasks.
### Data generation

```python
from chatsky_llm_autoconfig.algorithms.cycle_graph_generation_pipeline import GraphGenerationPipeline
from langchain_openai import ChatOpenAI


generation_model = ChatOpenAI(
        model='deepseek/deepseek-reasoner',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.2
    )
    
validation_model = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    temperature=0

pipeline = GraphGenerationPipeline(
        generation_model=generation_model,
        validation_model=validation_model
    )
    
topics = [
    "technical support conversation",
    "job interview",
    "restaurant reservation",
    "online shopping checkout",
    "travel booking"
]

successful_generations = []

for topic in topics:

    try:
        result = pipeline(topic)
        
        # Check the result type
        if isinstance(result, GraphGenerationResult):
            print(f"✅ Successfully generated graph for {topic}")
            # Save the full result with the graph and dialog
            successful_generations.append({
                "graph": result.graph.model_dump(),
                "topic": result.topic,
                "dialogues": [d.model_dump() for d in result.dialogues]
            })
        else:  # isinstance(result, GenerationError)
            print(f"❌ Failed to generate graph for {topic}")
            print(f"Error type: {result.error_type}")
            print(f"Error message: {result.message}")
                
    except Exception as e:
        print(f"❌ Unexpected error processing topic '{topic}': {str(e)}")
        continue
```

### Dialogue sampling

```python
from chatsky_llm_autoconfig.algorithms.dialogue_generation import RecursiveDialogueSampler
from chatsky_llm_autoconfig.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogueSampler()
sampler.invoke(graph=G) #-> list of Dialogue objects

```

### Graph generation
### Evaluation

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
