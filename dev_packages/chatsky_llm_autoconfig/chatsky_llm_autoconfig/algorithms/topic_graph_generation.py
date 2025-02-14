from typing import Optional
from chatsky_llm_autoconfig.algorithms.base import TopicGraphGenerator
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
# from langchain.prompts import PromptTemplate
from chatsky_llm_autoconfig.schemas import GraphGenerationResult
from chatsky_llm_autoconfig.algorithms.cycle_graph_generation_pipeline import GraphGenerationPipeline, CycleGraphGenerator
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from chatsky_llm_autoconfig.prompts import extra_edge_prompt

from langchain_core.language_models.chat_models import BaseChatModel

# from pydantic import Field
# from typing import ClassVar

@AlgorithmRegistry.register(input_type=str, output_type=list)
class LoopedGraphGenerator(TopicGraphGenerator):

    generation_model: BaseChatModel = None
    validation_model: BaseChatModel = None
    pipeline: GraphGenerationPipeline = None

    def __init__(self, generation_model: BaseChatModel, validation_model: BaseChatModel):
        super().__init__()
        self.generation_model = generation_model
        self.validation_model = validation_model
        self.pipeline = GraphGenerationPipeline(
                    generation_model=generation_model,
                    validation_model=validation_model
                )
    
    def invoke(self, topic) -> list[dict]:
        print(f"\n{'='*50}")
        print(f"Generating graph for topic: {topic}")
        print(f"{'='*50}")
        successful_generations = []
        try:
            result = self.pipeline(topic)
            
            # Проверяем тип результата
            if isinstance(result, GraphGenerationResult):
                print(f"✅ Successfully generated graph for {topic}")
                # Сохраняем полный результат с графом и диалогами
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

        return successful_generations


# @AlgorithmRegistry.register(input_type=BaseGraph, output_type=BaseGraph)
class ExtraEdgeGenerator:

    generation_model: BaseChatModel = None
    graph_generator: CycleGraphGenerator

    def __init__(self, generation_model: BaseChatModel):
        self.generation_model = generation_model
        self.graph_generator = CycleGraphGenerator()

    def invoke(self, graph: BaseGraph) -> BaseGraph:


        current_graph = self.graph_generator.invoke(
                    model=self.generation_model,
                    prompt=extra_edge_prompt,
                    graph_json=graph.graph_dict
                )


        return current_graph