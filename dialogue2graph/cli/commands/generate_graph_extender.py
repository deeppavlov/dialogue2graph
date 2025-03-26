import json
from pathlib import Path
from dialogue2graph.pipelines.d2g_extender.pipeline import Pipeline
from dialogue2graph.pipelines.models import ModelsAPI

models = ModelsAPI()


def generate_extender(dialogues: str, config: dict, output_path: str):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config == {}:
        extender_name = "chatgpt-4o-latest"
        temp1 = 0
        filler_name = "chatgpt-4o-latest"
        temp2 = 0
        sim_name = "BAAI/bge-m3"
        device = "cpu"
    else:
        extender_name = config["models"].get("extender-model", {}).get("name", "chatgpt-4o-latest")
        temp1 = config["models"].get("extender-model", {}).get("temperature", 0)
        filler_name = config["models"].get("filler-model", {}).get("name", "chatgpt-4o-latest")
        temp2 = config["models"].get("filler-model", {}).get("temperature", 0)
        sim_name = config["models"].get("sim-model", {}).get("name", "BAAI/bge-m3")
        device = config["models"].get("sim-model", {}).get("device", "cpu")

    extending_llm = models("llm", name=extender_name, temp=temp1)
    filling_llm = models("llm", name=filler_name, temp=temp2)
    sim_model = models("similarity", name=sim_name, device=device)

    pipeline = Pipeline(extending_llm=extending_llm, filling_llm=filling_llm, sim_model=sim_model)

    result = pipeline.invoke(dialogues)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)
