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

## How to use

We provide several of using our library for various tasks.

### Data generation

```python
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
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

successful_generations = []

for topic in topics:
    try:
        result = pipeline(topic, use_cache=False)
        
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
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from chatsky_llm_autoconfig.graph import Graph

G = Graph(graph_dict={...})

sampler = RecursiveDialogueSampler()
sampler.invoke(graph=G) #-> list of Dialogue objects

```

## How to contribute?

You can find contribution guideline in [CONTRIBUTING.md](https://github.com/deeppavlov/chatsky-llm-autoconfig/blob/main/CONTRIBUTING.md)
