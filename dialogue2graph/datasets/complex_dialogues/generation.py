import logging
from enum import Enum
from typing import Optional, Dict, Any, Union

from pydantic import BaseModel, Field
import networkx as nx

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler
from dialogue2graph.metrics.no_llm_metrics import match_triplets_dg, is_greeting_repeated, is_closed_too_early
from dialogue2graph.metrics.llm_metrics import are_triplets_valid, is_theme_valid
from dialogue2graph.pipelines.core.graph import BaseGraph, Graph
from dialogue2graph.pipelines.core.algorithms import TopicGraphGenerator
from dialogue2graph.pipelines.core.schemas import GraphGenerationResult, DialogueGraph
from dialogue2graph.utils.prompt_caching import setup_cache, add_uuid_to_prompt

from .prompts import cycle_graph_generation_prompt_informal, cycle_graph_repair_prompt, graph_example

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CycleGraphGenerator(BaseModel):
    cache: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)

    def invoke(self, model: BaseChatModel, prompt: PromptTemplate, seed=None, **kwargs) -> BaseGraph:
        """
        Generate a cyclic dialogue graph based on the topic input.
        """

        # Add UUID to the prompt template
        original_template = prompt.template
        prompt.template = add_uuid_to_prompt(original_template, seed)

        parser = PydanticOutputParser(pydantic_object=DialogueGraph)
        chain = prompt | model | parser

        # Reset template to original
        prompt.template = original_template
        return Graph(chain.invoke(kwargs).model_dump())

    async def ainvoke(self, *args, **kwargs):
        """Async version of invoke - to be implemented"""
        pass

    def evaluate(self, *args, report_type="dict", **kwargs):
        pass


class GenerationPipeline(BaseModel):
    cache: Optional[Any] = Field(default=None, exclude=True)
    generation_model: BaseChatModel
    theme_validation_model: BaseChatModel
    validation_model: BaseChatModel
    graph_generator: CycleGraphGenerator = Field(default_factory=CycleGraphGenerator)
    generation_prompt: Optional[PromptTemplate] = Field(default_factory=lambda: cycle_graph_generation_prompt_informal)
    repair_prompt: Optional[PromptTemplate] = Field(default_factory=lambda: cycle_graph_repair_prompt)
    min_cycles: int = 2
    max_fix_attempts: int = 3
    dialogue_sampler: RecursiveDialogueSampler = Field(default_factory=RecursiveDialogueSampler)
    seed: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        generation_model: BaseChatModel,
        theme_validation_model: BaseChatModel,
        validation_model: BaseChatModel,
        generation_prompt: Optional[PromptTemplate],
        repair_prompt: Optional[PromptTemplate],
        min_cycles: int = 2,
        max_fix_attempts: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__(
            generation_model=generation_model,
            theme_validation_model=theme_validation_model,
            validation_model=validation_model,
            generation_prompt=generation_prompt,
            repair_prompt=repair_prompt,
            min_cycles=min_cycles,
            max_fix_attempts=max_fix_attempts,
            seed=seed,
        )
        self.seed = seed
        if self.seed:
            self.cache = setup_cache()

    def validate_graph_cycle_requirement(self, graph: BaseGraph, min_cycles: int = 2) -> Dict[str, Any]:
        """Checks the graph for cycle requirements"""
        logger.info("🔍 Checking graph requirements...")
        try:
            cycles = list(nx.simple_cycles(graph.graph))
            cycles_count = len(cycles)
            logger.info(f"🔄 Found {cycles_count} cycles in the graph:")
            for i, cycle in enumerate(cycles, 1):
                logger.info(f"Cycle {i}: {' -> '.join(map(str, cycle + [cycle[0]]))}")

            number_cycle_requirement = cycles_count >= min_cycles
            no_start_cycle_requirement = not any([1 in c for c in cycles])
            meets_requirements = number_cycle_requirement and no_start_cycle_requirement

            if meets_requirements:
                logger.info("✅ Graph meets cycle requirements")
            else:
                logger.info(f"❌ Graph doesn't meet cycle requirements (minimum {min_cycles} cycles needed)")
            return {"meets_requirements": meets_requirements, "cycles": cycles, "cycles_count": cycles_count}

        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            raise

    def check_and_fix_transitions(self, graph: BaseGraph, max_attempts: int = 3) -> Dict[str, Any]:
        """Checks transitions in the graph and attempts to fix invalid ones via LLM"""
        logger.info("Validating initial graph")
        initial_validation = are_triplets_valid(graph, self.validation_model, return_type="detailed")
        logger.info("Finished validating initial graph")
        if initial_validation["is_valid"]:
            return {"is_valid": True, "graph": graph, "validation_details": {"invalid_transitions": [], "attempts_made": 0, "fixed_count": 0}}

        initial_invalid_count = len(initial_validation["invalid_transitions"])
        current_graph = graph
        current_attempt = 0
        logger.warning(f"⚠️ Found these {initial_validation['invalid_transitions']} invalid transitions")
        while current_attempt < max_attempts:
            logger.info(f"🔄 Fix attempt {current_attempt + 1}/{max_attempts}")
            try:
                # Add UUID to repair prompt
                if self.repair_prompt:
                    original_template = self.repair_prompt.template
                    self.repair_prompt.template = add_uuid_to_prompt(original_template, seed=self.seed)

                current_graph = self.graph_generator.invoke(
                    model=self.generation_model,
                    prompt=self.repair_prompt,
                    invalid_transitions=initial_validation["invalid_transitions"],
                    graph_json=current_graph.graph_dict,
                    seed=self.seed,
                )

                # Reset template
                if self.repair_prompt:
                    self.repair_prompt.template = original_template

            except Exception as e:
                logger.error(f"⚠️ Error during fix attempt: {str(e)}")
                break

            current_attempt += 1

        validation = are_triplets_valid(current_graph, self.validation_model, return_type="detailed")
        if validation["is_valid"]:
            return {
                "is_valid": True,
                "graph": current_graph,
                "validation_details": {"invalid_transitions": [], "attempts_made": current_attempt + 1, "fixed_count": initial_invalid_count},
            }
        else:
            logger.warning(f"⚠️ Found these {validation['invalid_transitions']} invalid transitions after fix attempt")
            remaining_invalid = len(validation.get("invalid_transitions", []))
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
            logger.info("Generating Graph ...")
            graph = self.graph_generator.invoke(
                model=self.generation_model, prompt=self.generation_prompt, graph_example=graph_example, topic=topic, seed=self.seed
            )
            logger.info(f"Graph generated is {graph.graph_dict}")
            if not graph.edges_match_nodes():
                return GenerationError(error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Generated graph is wrong: edges don't match nodes")
            graph = graph.remove_duplicated_nodes()
            if graph is None:
                return GenerationError(error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Generated graph is wrong: utterances in nodes doubled")

            cycle_validation = self.validate_graph_cycle_requirement(graph, self.min_cycles)
            if not cycle_validation["meets_requirements"]:
                return GenerationError(
                    error_type=ErrorType.TOO_MANY_CYCLES,
                    message=f"Graph requires minimum {self.min_cycles} cycles, found {cycle_validation['cycles_count']}",
                )

            logger.info("Sampling dialogues...")
            sampled_dialogues = self.dialogue_sampler.invoke(graph, 15)
            logger.info(f"Sampled {len(sampled_dialogues)} dialogues")
            if not match_triplets_dg(graph, sampled_dialogues)["value"]:
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - not all utterances are present"
                )
            if is_greeting_repeated(sampled_dialogues):
                return GenerationError(error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - Opening is repeated")
            if is_closed_too_early(sampled_dialogues):
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - Closing appears in the middle of a dialogue"
                )

            theme_validation = is_theme_valid(graph, self.theme_validation_model, topic)
            if not theme_validation["value"]:
                return GenerationError(error_type=ErrorType.INVALID_THEME, message=f"Theme validation failed: {theme_validation['description']}")

            logger.info("Validating and fixing transitions...")
            transition_validation = self.check_and_fix_transitions(graph=graph, max_attempts=self.max_fix_attempts)
            logger.info("Finished validating and fixing transitions")

            if not transition_validation["is_valid"]:
                invalid_transitions = transition_validation["validation_details"]["invalid_transitions"]
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message=f"Found {len(invalid_transitions)} invalid transitions "
                    f"after {transition_validation['validation_details']['attempts_made']} fix attempts",
                )

            graph = transition_validation["graph"]
            if transition_validation["validation_details"]["attempts_made"]:
                if not graph.edges_match_nodes():
                    return GenerationError(error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Generated graph is wrong: edges don't match nodes")
                graph = graph.remove_duplicated_nodes()
                if graph is None:
                    return GenerationError(
                        error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Generated graph is wrong: utterances in nodes doubled"
                    )
                print("Sampling dialogues...")
                sampled_dialogues = self.dialogue_sampler.invoke(graph, 15)
                print(f"Sampled {len(sampled_dialogues)} dialogues")
                if not match_triplets_dg(graph, sampled_dialogues)["value"]:
                    return GenerationError(
                        error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - not all utterances are present"
                    )

            logger.info(f"going to return: {transition_validation['graph'].graph_dict}")
            ret = GraphGenerationResult(graph=transition_validation["graph"].graph_dict, topic=topic, dialogues=sampled_dialogues)
            logger.info(f"ret: {ret}")
            return ret

        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}")
            return GenerationError(error_type=ErrorType.GENERATION_FAILED, message=f"Unexpected error during generation: {str(e)}")

    def __call__(self, topic: str) -> PipelineResult:
        """Shorthand for generate_and_validate"""
        return self.generate_and_validate(topic)


class LoopedGraphGenerator(TopicGraphGenerator):
    generation_model: BaseChatModel
    validation_model: BaseChatModel
    pipeline: GenerationPipeline

    def __init__(self, generation_model: BaseChatModel, validation_model: BaseChatModel, theme_validation_model: BaseChatModel):
        super().__init__(
            generation_model=generation_model,
            validation_model=validation_model,
            theme_validation_model=theme_validation_model,
            pipeline=GenerationPipeline(
                generation_model=generation_model,
                validation_model=validation_model,
                theme_validation_model=theme_validation_model,
                generation_prompt=cycle_graph_generation_prompt_informal,
                repair_prompt=cycle_graph_repair_prompt,
            ),
        )

    def invoke(self, topic, seed=42) -> list[dict]:
        print(f"\n{'=' * 50}")
        print(f"Generating graph for topic: {topic}")
        print(f"{'=' * 50}")
        successful_generations = []
        try:
            self.pipeline.seed = seed
            result = self.pipeline(topic)

            if isinstance(result, GraphGenerationResult):
                logger.info(f"✅ Successfully generated graph for {topic}")
                logger.info(f"ID: {result.dialogues[0].id}")
                successful_generations.append(
                    {
                        "graph": result.graph.model_dump(),
                        "topic": result.topic,
                        "dialogues": [dia.model_dump() for dia in result.dialogues],  # The dialogues are already dictionaries
                    }
                )
            else:
                logger.info(f"❌ Failed to generate graph for {topic}")
                logger.error(f"Error type: {result.error_type}")
                logger.error(f"Error message: {result.message}")

        except Exception as e:
            logger.error(f"❌ Unexpected error processing topic '{topic}': {str(e)}")

        return successful_generations

    def evaluate(self, *args, report_type="dict", **kwargs):
        return super().evaluate(*args, report_type=report_type, **kwargs)
