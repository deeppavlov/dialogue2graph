import json
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.d2g_algo.pipeline import Pipeline


def generate_algo(dialogues: str, config: dict, output_path: str):
    """Generates graph from dialogues via d2g_algo pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config == {}:
        filler_name = "chatgpt-4o-latest"
        temp = 0
        embedder_name = "BAAI/bge-m3"
        device = "cpu"
    else:
        filler_name = config["models"].get("filler-model", {}).get("name", "chatgpt-4o-latest")
        temp = config["models"].get("filler-model", {}).get("temperature", 0)
        embedder_name = config["models"].get("embedder-model", {}).get("name", "BAAI/bge-m3")
        device = config["models"].get("embedder-model", {}).get("device", "cpu")

    filling_llm = ChatOpenAI(model=filler_name, temperature=temp)
    embedder = HuggingFaceEmbeddings(model_name=embedder_name, model_kwargs={"device": device})

    pipeline = Pipeline(filling_llm=filling_llm, embedder=embedder)

    result = pipeline.invoke(dialogues)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)
