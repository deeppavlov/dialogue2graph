import json
from pathlib import Path
from dialogue2graph.pipelines.d2g_algo.pipeline import Pipeline
from dialogue2graph.pipelines.models import ModelsAPI

models = ModelsAPI()


def generate_light(dialogues: str, config: dict, output_path: str):
    """Generates graph from dialogues via d2g_algo pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config == {}:
        filler_name = "chatgpt-4o-latest"
        formatter_name = "gpt-4o-mini"
        filler_temp = 0
        formatter_temp = 0
        sim_name = "BAAI/bge-m3"
        device = "cpu"
    else:
        filler_name = config["models"].get("filler-model", {}).get("name", "chatgpt-4o-latest")
        formatter_name = config["models"].get("formatter-model", {}).get("name", "gpt-4o-mini")
        filler_temp = config["models"].get("filler-model", {}).get("temperature", 0)
        formatter_temp = config["models"].get("formatter-model", {}).get("temperature", 0)
        sim_name = config["models"].get("sim-model", {}).get("name", "BAAI/bge-m3")
        device = config["models"].get("sim-model", {}).get("device", "cpu")

    filling_llm = models("llm", name=filler_name, temp=filler_temp)
    formatting_llm = models("llm", name=formatter_name, temp=formatter_temp)
    sim_model = models("similarity", name=sim_name, device=device)

    pipeline = Pipeline(filling_llm=filling_llm, formatting_llm=formatting_llm, sim_model=sim_model)

    result = pipeline.invoke(dialogues)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)
