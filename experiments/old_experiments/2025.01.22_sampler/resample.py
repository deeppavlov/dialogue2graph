import json
import pathlib

from chatsky_llm_autoconfig.algorithms.cycle_graph_generation_pipeline import GraphGenerationPipeline
from chatsky_llm_autoconfig.settings import EnvSettings
from chatsky_llm_autoconfig.graph import Graph
from chatsky_llm_autoconfig.metrics.automatic_metrics import all_utterances_present
from chatsky_llm_autoconfig.utils import save_json

from langchain_openai  import ChatOpenAI
env_settings = EnvSettings()

generation_model = ChatOpenAI(model=env_settings.GENERATION_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL, temperature=1)
validation_model = ChatOpenAI(model=env_settings.FORMATTER_MODEL_NAME, api_key=env_settings.OPENAI_API_KEY, base_url=env_settings.OPENAI_BASE_URL)


data =[]
for n in pathlib.Path("gen").iterdir():
    with open(n, encoding="utf-8") as f:
        print(n)
        data.extend(json.load(f))


pipeline = GraphGenerationPipeline(
                    generation_model=generation_model,
                    validation_model=validation_model
                )

for idx, d in enumerate(data):        
    print("Sampling dialogues...", idx)
    graph = Graph(d['graph'])
    sampled_dialogues = pipeline.dialogue_sampler.invoke(graph, 15)
    print(f"Sampled {len(sampled_dialogues)} dialogues")
    if not all_utterances_present(graph, sampled_dialogues):
        print("Failed to sample valid dialogues - not all utterances are present ", idx)
        break
    data[idx]['dialogues'] = [d.model_dump() for d in sampled_dialogues]

save_json(data=data, filename='generated.json')

