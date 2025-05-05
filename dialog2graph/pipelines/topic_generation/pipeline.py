from __future__ import annotations

import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Tuple

from dialog2graph.pipelines.core.graph import Graph
from dialog2graph.pipelines.report import PipelineReport
from dialog2graph.pipelines.core.pipeline import BasePipeline
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.datasets.complex_dialogs.generation import LoopedGraphGenerator

load_dotenv()


class TopicGenerationPipeline(BasePipeline):
    """
    Pipeline for generating dialog graph based on a given topic.

    This pipeline utilizes multiple language models (LLMs) for topic generation,
    validation, and theme validation. If the specified models are not present
    in the provided model storage, default configurations are added.

    Attributes:
        model_storage (ModelStorage): The storage object containing available models.
        generation_llm (str): Key for the LLM used for topic generation. Defaults to "looped_graph_generation_llm:v1".
        validation_llm (str): Key for the LLM used for validation. Defaults to "looped_graph_validation_llm:v1".
        cycle_ends_llm (str): Key for the LLM for dialog sampler to find cycle ends. Defaults to "looped_graph_cycle_ends_llm:v1.
        theme_validation_llm (str): Key for the LLM used for theme validation. Defaults to "looped_graph_theme_validation_llm:v1".

    Methods:

        invoke(topic: str, output_path: str):
            Executes the pipeline for a given topic and saves the output to the specified path.

            Args:
                topic (str): The topic for which the dialog graph is generated.
                output_path (str): The file path where the output will be saved.

            Returns:
                dict: The generated output from the pipeline.
    """

    def __init__(
        self,
        model_storage: ModelStorage,
        generation_llm: str = "looped_graph_generation_llm:v1",
        validation_llm: str = "looped_graph_validation_llm:v1",
        cycle_ends_llm: str = "looped_graph_cycle_ends_llm:v1",
        theme_validation_llm: str = "looped_graph_theme_validation_llm:v1",
    ):
        super().__init__(
            steps=[
                LoopedGraphGenerator(
                    model_storage=model_storage,
                    generation_llm=generation_llm,
                    validation_llm=validation_llm,
                    cycle_ends_llm=cycle_ends_llm,
                    theme_validation_llm=theme_validation_llm,
                )
            ]
        )

    def _validate_pipeline(self):
        pass

    def invoke(self, topic: str, output_path: str) -> Tuple[Graph, PipelineReport]:
        """
        Executes a series of steps to process a given topic and saves the output graph to a specified file.

        Args:
            topic (str): The topic to be processed by the pipeline steps.
            output_path (str): The JSON file path where the graph will be saved.

        Returns:
            Tuple[Graph, PipelineReport]: A tuple containing the generated graph and a pipeline report.

        Raises:
            OSError: If there is an issue creating directories or writing to the output file.
        """
        for step in self.steps:
            output = step.invoke(topic=topic)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        return output
