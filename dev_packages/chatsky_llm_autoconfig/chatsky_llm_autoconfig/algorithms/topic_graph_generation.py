from typing import Optional
from chatsky_llm_autoconfig.algorithms.base import TopicGraphGenerator
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry
from chatsky_llm_autoconfig.schemas import DialogueGraph
from langchain.prompts import PromptTemplate
from chatsky_llm_autoconfig.schemas import DialogueGraph
from chatsky_llm_autoconfig.schemas import GraphGenerationResult
from chatsky_llm_autoconfig.algorithms.cycle_graph_generation_pipeline import GraphGenerationPipeline
from langchain_core.output_parsers import PydanticOutputParser
from chatsky_llm_autoconfig.graph import BaseGraph, Graph
from langchain_core.language_models.chat_models import BaseChatModel

from pydantic import Field
from typing import ClassVar


@AlgorithmRegistry.register(input_type=str, output_type=BaseGraph)
class CycleGraphGenerator(TopicGraphGenerator):
    """Generator specifically for topic-based cyclic graphs"""

    def __init__(self):
        super().__init__()

    def invoke(self, model: BaseChatModel, prompt: PromptTemplate, **kwargs) -> BaseGraph:
        """
        Generate a cyclic dialogue graph based on the topic input.

        Args:
            model (BaseChatModel): The model to use for generation
            prompt (PromptTemplate): Prepared prompt template
            **kwargs: Additional arguments for formatting the prompt

        Returns:
            BaseGraph: Generated Graph object with cyclic structure
        """
        # Создаем цепочку: промпт -> модель -> парсер
        parser = PydanticOutputParser(pydantic_object=DialogueGraph)
        chain = prompt | model | parser

        # Передаем kwargs как входные данные для цепочки
        return Graph(chain.invoke(kwargs))

    async def ainvoke(self, *args, **kwargs):
        """
        Async version of invoke - to be implemented
        """
        pass


class LoopedGraphGenerator(TopicGraphGenerator):
    def __init__(self, generation_model, validation_model):
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