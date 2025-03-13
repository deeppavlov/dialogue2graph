import json
import os
from pathlib import Path
from langchain_openai import ChatOpenAI
from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator


def generate_data(topic: str, output_path: str):
    """Generate dialogue data for a given topic"""
    gen_model = ChatOpenAI(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    val_model = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

    pipeline = LoopedGraphGenerator(
        generation_model=gen_model,
        validation_model=val_model,
    )

    result = pipeline.invoke(topic=topic)

    print("Result:", result)

    # Save results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([obj.model_dump() for obj in result], f, indent=2, ensure_ascii=False)
