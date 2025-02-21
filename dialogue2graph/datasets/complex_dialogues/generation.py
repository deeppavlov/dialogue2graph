from enum import Enum
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from pydantic import BaseModel, Field
import networkx as nx

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.metrics.automatic_metrics import all_utterances_present
from dialogue2graph.metrics.llm_metrics import are_triplets_valid, is_theme_valid
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.algorithms import TopicGraphGenerator
from dialogue2graph.pipelines.core.schemas import GraphGenerationResult, DialogueGraph
from dialogue2graph.utils.prompt_caching import setup_cache

from .prompts import cycle_graph_generation_prompt_enhanced, cycle_graph_repair_prompt


class ErrorType(str, Enum):
    """Types of errors that can occur during generation"""

    INVALID_GRAPH_STRUCTURE = "invalid_graph_structure"
    TOO_MANY_CYCLES = "too_many_cycles"
    SAMPLING_FAILED = "sampling_failed"
    INVALID_THEME = "invalid_theme"
    GENERATION_FAILED = "generation_failed"


class GenerationError(BaseModel):
    """Base error with essential fields"""

    error_type: ErrorType
    message: str


PipelineResult = Union[GraphGenerationResult, GenerationError]


class CycleGraphGenerator(TopicGraphGenerator):
    """Generator specifically for topic-based cyclic graphs"""

    def __init__(self):
        super().__init__()

    def invoke(self, model: BaseChatModel, prompt: PromptTemplate, use_cache=True, **kwargs) -> BaseGraph:
        """
        Generate a cyclic dialogue graph based on the topic input.
        """
        if use_cache:
            setup_cache()
        parser = PydanticOutputParser(pydantic_object=DialogueGraph)
        chain = prompt | model | parser
        return Graph(chain.invoke(kwargs))

    async def ainvoke(self, *args, **kwargs):
        """Async version of invoke - to be implemented"""
        pass

    def evaluate(self, *args, report_type="dict", **kwargs):
        pass


@dataclass
class GenerationPipeline(BaseModel):
    generation_model: BaseChatModel
    validation_model: BaseChatModel
    graph_generator: CycleGraphGenerator = Field(default_factory=CycleGraphGenerator)
    generation_prompt: Optional[PromptTemplate] = Field(default_factory=lambda: cycle_graph_generation_prompt_enhanced)
    repair_prompt: Optional[PromptTemplate] = Field(default_factory=lambda: cycle_graph_repair_prompt)
    min_cycles: int = 2
    max_fix_attempts: int = 3
    dialogue_sampler: RecursiveDialogueSampler = Field(default_factory=RecursiveDialogueSampler)
    use_cache: bool = True

    def __init__(
        self,
        generation_model: BaseChatModel,
        validation_model: BaseChatModel,
        generation_prompt: Optional[PromptTemplate],
        repair_prompt: Optional[PromptTemplate],
        min_cycles: int = 0,
        max_fix_attempts: int = 2,
        use_cache: bool = True,
    ):
        super().__init__(
            generation_model=generation_model,
            validation_model=validation_model,
            generation_prompt=generation_prompt,
            repair_prompt=repair_prompt,
            min_cycles=min_cycles,
            max_fix_attempts=max_fix_attempts,
            use_cache=True,
        )

    def validate_graph_cycle_requirement(self, graph: BaseGraph, min_cycles: int = 2) -> Dict[str, Any]:
        """Checks the graph for cycle requirements"""
        print("\n🔍 Checking graph requirements...")
        try:
            cycles = list(nx.simple_cycles(graph.graph))
            cycles_count = len(cycles)
            print(f"🔄 Found {cycles_count} cycles in the graph:")
            for i, cycle in enumerate(cycles, 1):
                print(f"Cycle {i}: {' -> '.join(map(str, cycle + [cycle[0]]))}")

            meets_requirements = cycles_count >= min_cycles
            print(
                "✅ Graph meets cycle requirements"
                if meets_requirements
                else f"❌ Graph doesn't meet cycle requirements (minimum {min_cycles} cycles needed)"
            )
            return {"meets_requirements": meets_requirements, "cycles": cycles, "cycles_count": cycles_count}

        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            raise

    def check_and_fix_transitions(self, graph: BaseGraph, max_attempts: int = 3) -> Dict[str, Any]:
        """Checks transitions in the graph and attempts to fix invalid ones via LLM"""
        print("Validating initial graph")
        initial_validation = are_triplets_valid(graph, self.validation_model, return_type="detailed")
        if initial_validation["is_valid"]:
            return {"is_valid": True, "graph": graph, "validation_details": {"invalid_transitions": [], "attempts_made": 0, "fixed_count": 0}}

        initial_invalid_count = len(initial_validation["invalid_transitions"])
        current_graph = graph
        current_attempt = 0

        while current_attempt < max_attempts:
            print(f"\n🔄 Fix attempt {current_attempt + 1}/{max_attempts}")
            try:
                current_graph = self.graph_generator.invoke(
                    model=self.generation_model,
                    prompt=self.repair_prompt,
                    invalid_transitions=initial_validation["invalid_transitions"],
                    graph_json=current_graph.graph_dict,
                    use_cache=self.use_cache,
                )

                validation = are_triplets_valid(current_graph, self.validation_model, return_type="detailed")
                if validation["is_valid"]:
                    return {
                        "is_valid": True,
                        "graph": current_graph,
                        "validation_details": {"invalid_transitions": [], "attempts_made": current_attempt + 1, "fixed_count": initial_invalid_count},
                    }
                else:
                    print(f"⚠️ Found these {validation['invalid_transitions']} invalid transitions after fix attempt")

            except Exception as e:
                print(f"⚠️ Error during fix attempt: {str(e)}")
                break

            current_attempt += 1

        remaining_invalid = len(validation["invalid_transitions"])
        return {
            "is_valid": False,
            "graph": current_graph,
            "validation_details": {
                "invalid_transitions": validation["invalid_transitions"],
                "attempts_made": current_attempt,
                "fixed_count": initial_invalid_count - remaining_invalid,
            },
        }

    def generate_and_validate(self, topic: str) -> PipelineResult:
        """Generates and validates a dialogue graph for given topic"""
        try:
            print("Generating Graph ...")
            graph = self.graph_generator.invoke(model=self.generation_model, prompt=self.generation_prompt, topic=topic, use_cache=self.use_cache)

            cycle_validation = self.validate_graph_cycle_requirement(graph, self.min_cycles)
            if not cycle_validation["meets_requirements"]:
                return GenerationError(
                    error_type=ErrorType.TOO_MANY_CYCLES,
                    message=f"Graph requires minimum {self.min_cycles} cycles, found {cycle_validation['cycles_count']}",
                )

            print("Sampling dialogues...")
            sampled_dialogues = self.dialogue_sampler.invoke(graph, 15)
            print(f"Sampled {len(sampled_dialogues)} dialogues")
            if not all_utterances_present(graph, sampled_dialogues):
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - not all utterances are present"
                )

            theme_validation = is_theme_valid(graph, self.validation_model, topic)
            if not theme_validation["value"]:
                return GenerationError(error_type=ErrorType.INVALID_THEME, message=f"Theme validation failed: {theme_validation['description']}")

            print("Validating and fixing transitions...")
            transition_validation = self.check_and_fix_transitions(graph=graph, max_attempts=self.max_fix_attempts)

            if not transition_validation["is_valid"]:
                invalid_transitions = transition_validation["validation_details"]["invalid_transitions"]
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message=f"Found {len(invalid_transitions)} invalid transitions"
                    f"after {transition_validation['validation_details']['attempts_made']} fix attempts",
                )

            return GraphGenerationResult(graph=transition_validation["graph"].graph_dict, topic=topic, dialogues=sampled_dialogues)

        except Exception as e:
            return GenerationError(error_type=ErrorType.GENERATION_FAILED, message=f"Unexpected error during generation: {str(e)}")

    def __call__(self, topic: str) -> PipelineResult:
        """Shorthand for generate_and_validate"""
        return self.generate_and_validate(topic)


class LoopedGraphGenerator(TopicGraphGenerator):
    generation_model: BaseChatModel
    validation_model: BaseChatModel
    pipeline: GenerationPipeline

    def __init__(self, generation_model: BaseChatModel, validation_model: BaseChatModel):
        super().__init__(
            generation_model=generation_model,
            validation_model=validation_model,
            pipeline=GenerationPipeline(
                generation_model=generation_model,
                validation_model=validation_model,
                generation_prompt=cycle_graph_generation_prompt_enhanced,
                repair_prompt=cycle_graph_repair_prompt,
            ),
        )

    def invoke(self, topic, use_cache=True) -> list[dict]:
        print(f"\n{'='*50}")
        print(f"Generating graph for topic: {topic}")
        print(f"{'='*50}")
        successful_generations = []
        try:
            result = self.pipeline(topic, use_cache=use_cache)

            if isinstance(result, GraphGenerationResult):
                print(f"✅ Successfully generated graph for {topic}")
                successful_generations.append(
                    {"graph": result.graph.model_dump(), "topic": result.topic, "dialogues": [d.model_dump() for d in result.dialogues]}
                )
            else:
                print(f"❌ Failed to generate graph for {topic}")
                print(f"Error type: {result.error_type}")
                print(f"Error message: {result.message}")

        except Exception as e:
            print(f"❌ Unexpected error processing topic '{topic}': {str(e)}")

        return successful_generations

    def evaluate(self, *args, report_type="dict", **kwargs):
        return super().evaluate(*args, report_type=report_type, **kwargs)
