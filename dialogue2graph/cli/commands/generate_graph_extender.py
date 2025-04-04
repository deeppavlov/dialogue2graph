import json
from pathlib import Path
from dialogue2graph.pipelines.d2g_extender.pipeline import Pipeline
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType


ms = ModelStorage()


def generate_extender(dialogues: str, config: dict, output_path: str):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config == {}:
        extender_name = "chatgpt-4o-latest"
        extender_temp = 0
        filler_name = "chatgpt-4o-latest"
        filler_temp = 0
        formatter_name = "gpt-4o-mini"
        formatter_temp = 0
        sim_name = "BAAI/bge-m3"
        device = "cpu"
    else:
        extender_name = config["models"].get("extender-model", {}).get("name", "chatgpt-4o-latest")
        extender_temp = config["models"].get("extender-model", {}).get("temperature", 0)
        filler_name = config["models"].get("filler-model", {}).get("name", "chatgpt-4o-latest")
        filler_temp = config["models"].get("filler-model", {}).get("temperature", 0)
        formatter_name = config["models"].get("formatter-model", {}).get("name", "gpt-4o-mini")
        formatter_temp = config["models"].get("formatter-model", {}).get("temperature", 0)
        sim_name = config["models"].get("sim-model", {}).get("name", "BAAI/bge-m3")
        device = config["models"].get("sim-model", {}).get("device", "cpu")

    extending_llm = models("llm", name=extender_name, temp=extender_temp)
    filling_llm = models("llm", name=filler_name, temp=filler_temp)
    formatting_llm = models("llm", name=formatter_name, temp=formatter_temp)
    sim_model = models("similarity", name=sim_name, device=device)

    pipeline = Pipeline(
        "d2g_ext",
        extending_llm,
        filling_llm,
        formatting_llm,
        sim_model,
        step1_evals=[],
        extender_evals=[],
        step2_evals=[],
        end_evals=[],
        step=1,
    )

    raw_data = PipelineRawDataType(dialogs=dialogues)
    result, _ = pipeline.invoke(raw_data)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)
