import json
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from dialogue2graph.pipelines.d2g_llm.pipeline import Pipeline


def generate_llm(dialogues: str, config: dict, output_path: str):
    """Generates graph from dialogues via d2g_llm pipeline using parameters from config
    and saves graph dictionary to output_path"""

    if config == {}:
        grouper_name = "chatgpt-4o-latest"
        temp1 = 0
        filler_name = "chatgpt-4o-latest"
        temp2 = 0
        embedder_name = "BAAI/bge-m3"
        device = "cpu"
    else:
        grouper_name = config["models"].get("grouper-model", {}).get("name", "chatgpt-4o-latest")
        temp1 = config["models"].get("grouper-model", {}).get("temperature", 0)
        filler_name = config["models"].get("filler-model", {}).get("name", "chatgpt-4o-latest")
        temp2 = config["models"].get("filler-model", {}).get("temperature", 0)
        embedder_name = config["models"].get("embedder-model", {}).get("name", "BAAI/bge-m3")
        device = config["models"].get("embedder-model", {}).get("device", "cpu")

    grouping_llm = ChatOpenAI(model=grouper_name, temperature=temp1)
    filling_llm = ChatOpenAI(model=filler_name, temperature=temp2)
    embedder = HuggingFaceEmbeddings(model_name=embedder_name, model_kwargs={"device": device})

    pipeline = Pipeline(grouping_llm=grouping_llm, filling_llm=filling_llm, embedder=embedder)

    result = pipeline.invoke(dialogues)
    print("Result:", result.graph_dict)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result.graph_dict, f, indent=2, ensure_ascii=False)
