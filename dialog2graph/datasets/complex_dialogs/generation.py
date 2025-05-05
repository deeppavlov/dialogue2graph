"""
Generation
----------

The module provides graph generator capable of creating complex validated graphs.
"""

from enum import Enum
from typing import Optional, Dict, Any, Union
from datetime import datetime

from pydantic import BaseModel, Field
import networkx as nx

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_openai import ChatOpenAI


from dialog2graph.pipelines.core.dialog_sampling import RecursiveDialogSampler
from dialog2graph.metrics.no_llm_metrics import match_dg_triplets
from dialog2graph.metrics.llm_metrics import are_triplets_valid, is_theme_valid
from dialog2graph.metrics.no_llm_validators import (
    is_greeting_repeated_regex,
    is_dialog_closed_too_early_regex,
)
from dialog2graph.pipelines.core.graph import BaseGraph, Graph, Metadata
from dialog2graph.pipelines.core.algorithms import TopicGraphGenerator
from dialog2graph.pipelines.core.schemas import GraphGenerationResult, DialogGraph
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.utils.prompt_caching import setup_cache, add_uuid_to_prompt

from .prompts import (
    cycle_graph_generation_prompt_informal,
    cycle_graph_repair_prompt,
    graph_example,
)

# Configure logging
from dialog2graph.utils.logger import Logger

logger = Logger(__name__)


class ErrorType(str, Enum):
    """Error types that can occur during generation"""

    INVALID_GRAPH_STRUCTURE = "invalid_graph_structure"
    TOO_FEW_CYCLES = "too_few_cycles"
    SAMPLING_FAILED = "sampling_failed"
    INVALID_THEME = "invalid_theme"
    GENERATION_FAILED = "generation_failed"


class GenerationError(BaseModel):
    """Base error with essential fields"""

    error_type: ErrorType
    message: str


PipelineResult = Union[GraphGenerationResult, GenerationError]


class CycleGraphGenerator(BaseModel):
    """Class for generating graph with cycles
    Attributes:
        cache: Caching mechanism
        model_storage: Storage for models
    """

    cache: Optional[Any] = Field(default=None, exclude=True)
    model_storage: ModelStorage = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)

    def invoke(
        self, generation_llm: str, prompt: PromptTemplate, seed=None, **kwargs
    ) -> BaseGraph:
        """
        Generate a cyclic dialog graph based on the topic input.
        Args:
            generation_llm: Name of the model to use for graph generation.
            prompt: Prompt to use for graph generation.
            seed: Seed for the generation.
            kwargs: Additional arguments for the prompt.
        Returns:
            Dialog graph.
        """

        # Add UUID to the prompt template
        original_template = prompt.template
        prompt.template = add_uuid_to_prompt(original_template, seed)

        parser = PydanticOutputParser(pydantic_object=DialogGraph)
        chain = prompt | self.model_storage.storage[generation_llm].model | parser

        # Reset template to original
        prompt.template = original_template
        models_config = self.model_storage.model_dump()
        for model_key in models_config["storage"]:
            keys_to_pop = []
            for key in models_config["storage"][model_key]["config"]:
                if "api_key" in key or "api_base" in key or "base_url" in key:
                    keys_to_pop.append(key)
            for key in keys_to_pop:
                models_config["storage"][model_key]["config"].pop(key, None)
        metadata = Metadata(
            generator_name="cycle_generator",
            models_config=models_config,
            schema_version="v1",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        return Graph(
            graph_dict=chain.invoke(kwargs).model_dump(),
            metadata=metadata,
        )

    async def ainvoke(self, *args, **kwargs):
        """Async version of invoke - to be implemented"""
        pass

    def evaluate(self, *args, report_type="dict", **kwargs):
        pass


class GenerationPipeline(BaseModel):
    """Class for generation pipeline
    Attributes:
        cache: Caching mechanism
        model_storage: Storage for models
        generation_llm: Name of the model to use for graph generation
        validation_llm: Name of the model to use for graph validation
        cycle_ends_llm: Name of the model to use for cycle ends detection
        theme_validation_llm: Name of the model to use for theme validation
        generation_prompt: Prompt to use for graph generation
    """

    cache: Optional[Any] = Field(default=None, exclude=True)
    model_storage: ModelStorage
    generation_llm: str
    validation_llm: str
    cycle_ends_llm: str
    theme_validation_llm: str
    graph_generator: CycleGraphGenerator = Field(default_factory=CycleGraphGenerator)
    generation_prompt: Optional[PromptTemplate] = Field(
        default_factory=lambda: cycle_graph_generation_prompt_informal
    )
    repair_prompt: Optional[PromptTemplate] = Field(
        default_factory=lambda: cycle_graph_repair_prompt
    )
    min_cycles: int = 2
    max_fix_attempts: int = 3
    dialog_sampler: RecursiveDialogSampler = Field(
        default_factory=RecursiveDialogSampler
    )
    seed: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        model_storage: ModelStorage,
        generation_llm: str,
        validation_llm: str,
        cycle_ends_llm: str,
        theme_validation_llm: str,
        generation_prompt: Optional[PromptTemplate],
        repair_prompt: Optional[PromptTemplate],
        min_cycles: int = 2,
        max_fix_attempts: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Initialize the GenerationPipeline with the given parameters.

        Parameters:
        model_storage (ModelStorage): Storage for models to use in the pipeline.
        generation_llm (str): Name of the model to use for graph generation.
        validation_llm (str): Name of the model to use for graph validation.
        cycle_ends_llm (str): Name of the model to use for cycle ends detection.
        theme_validation_llm (str): Name of the model to use for theme validation.
        generation_prompt (Optional[PromptTemplate]): Prompt to use for graph generation.
        repair_prompt (Optional[PromptTemplate]): Prompt to use for graph repair.
        min_cycles (int): Minimum number of cycles to generate in the graph.
        max_fix_attempts (int): Maximum number of attempts to fix the graph.
        seed (Optional[int]): Seed to use for caching.
        """
        super().__init__(
            model_storage=model_storage,
            generation_llm=generation_llm,
            validation_llm=validation_llm,
            cycle_ends_llm=cycle_ends_llm,
            theme_validation_llm=theme_validation_llm,
            generation_prompt=generation_prompt,
            repair_prompt=repair_prompt,
            min_cycles=min_cycles,
            max_fix_attempts=max_fix_attempts,
            seed=seed,
        )
        self.seed = seed
        if self.seed:
            self.cache = setup_cache()
        self.graph_generator = CycleGraphGenerator(model_storage=model_storage)

    def validate_graph_cycle_requirement(
        self, graph: BaseGraph, min_cycles: int = 2
    ) -> Dict[str, Any]:
        """
        Validate whether a graph meets the cycle requirements.

        Checks if the graph contains at least `min_cycles` cycles and
        that none of the cycles contain the start node (node 1).

        Args:
        - graph (BaseGraph): The graph to validate.
        - min_cycles (int): The minimum number of cycles required (default: 2).

        Returns:
        - A dictionary containing the result of the validation, including:
          - `meets_requirements` (bool): True if the graph meets the cycle requirements.
          - `cycles` (List[List[int]]): A list of cycles found in the graph.
          - `cycles_count` (int): The number of cycles found in the graph.
        """
        logger.info("🔍 Checking graph requirements...")
        try:
            cycles = list(nx.simple_cycles(graph.graph))
            cycles_count = len(cycles)
            logger.info(f"🔄 Found {cycles_count} cycles in the graph:")
            for i, cycle in enumerate(cycles, 1):
                logger.info(f"Cycle {i}: {' -> '.join(map(str, cycle + [cycle[0]]))}")

            number_cycle_requirement = cycles_count >= min_cycles
            no_start_cycle_requirement = not any([1 in c for c in cycles])
            if not no_start_cycle_requirement:
                logger.info("Detected cycle containing start node")

            meets_requirements = number_cycle_requirement and no_start_cycle_requirement

            if meets_requirements:
                logger.info("✅ Graph meets cycle requirements")
            else:
                logger.info(
                    f"❌ Graph doesn't meet cycle requirements (minimum {min_cycles} cycles needed)"
                )
            return {
                "meets_requirements": meets_requirements,
                "cycles": cycles,
                "cycles_count": cycles_count,
            }

        except Exception as e:
            logger.error(f"❌ Validation error: {str(e)}")
            raise

    def check_and_fix_transitions(
        self, graph: BaseGraph, max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Checks transitions in the graph and tries to fix invalid ones via LLM.

        Args:
        - graph (BaseGraph): The graph to check and fix.
        - max_attempts (int): The maximum number of attempts to fix the graph. Defaults to 3.

        Returns:
        - A dictionary containing the result of the validation, including:
          - `is_valid` (bool): True if the graph is valid.
          - `graph` (BaseGraph): The fixed graph.
          - `validation_details` (Dict[str, Any]): A dictionary containing details about the
            validation, including:
            - `invalid_transitions` (List[Tuple[int, int, str]]): The invalid transitions in
              the graph.
            - `attempts_made` (int): The number of attempts made to fix the graph.
            - `fixed_count` (int): The number of invalid transitions fixed.
        """
        logger.info("Validating initial graph")
        initial_validation = are_triplets_valid(
            graph,
            self.model_storage.storage[self.validation_llm].model,
            return_type="detailed",
        )
        logger.info("Finished validating initial graph")
        if initial_validation["is_valid"]:
            return {
                "is_valid": True,
                "graph": graph,
                "validation_details": {
                    "invalid_transitions": [],
                    "attempts_made": 0,
                    "fixed_count": 0,
                },
            }

        initial_invalid_count = len(initial_validation["invalid_transitions"])
        current_graph = graph
        current_attempt = 0
        logger.warning(
            f"⚠️ Found these {initial_validation['invalid_transitions']} invalid transitions"
        )
        while current_attempt < max_attempts:
            logger.info(f"🔄 Fix attempt {current_attempt + 1}/{max_attempts}")
            try:
                # Add UUID to repair prompt
                if self.repair_prompt:
                    original_template = self.repair_prompt.template
                    self.repair_prompt.template = add_uuid_to_prompt(
                        original_template, seed=self.seed
                    )

                current_graph = self.graph_generator.invoke(
                    generation_llm=self.generation_llm,
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

        validation = are_triplets_valid(
            current_graph,
            self.model_storage.storage[self.validation_llm].model,
            return_type="detailed",
        )
        if validation["is_valid"]:
            return {
                "is_valid": True,
                "graph": current_graph,
                "validation_details": {
                    "invalid_transitions": [],
                    "attempts_made": current_attempt + 1,
                    "fixed_count": initial_invalid_count,
                },
            }
        else:
            logger.warning(
                f"⚠️ Found these {validation['invalid_transitions']} invalid transitions after fix attempt"
            )
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
        """
        Generates a graph and validates it according to the following criteria:
        1. The graph has at least min_cycles cycles.
        2. The graph is valid according to the validation model.
        3. The graph is themed according to the theme validation model.
        4. The graph has no invalid transitions after attempting to fix them up to max_fix_attempts times.
        5. Dialogs sampled from the graph cover all nodes and cover all edges.

        Args:
            topic: The topic to generate the graph for.

        Returns:
            GraphGenerationResult: A GraphGenerationResult object containing the generated graph, metadata, and sampled dialogs.
            If the graph fails to meet any of the criteria, a GenerationError is returned instead.
        """
        try:
            logger.info("Generating Graph ...")
            graph = self.graph_generator.invoke(
                generation_llm=self.generation_llm,
                prompt=self.generation_prompt,
                graph_example=graph_example,
                topic=topic,
                seed=self.seed,
            )
            logger.info(f"Graph generated is {graph.graph_dict}")
            if not graph.match_edges_nodes():
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message="Generated graph is wrong: edges don't match nodes",
                )
            graph = graph.remove_duplicated_nodes()
            if graph is None:
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message="Generated graph is wrong: utterances in nodes doubled",
                )

            cycle_validation = self.validate_graph_cycle_requirement(
                graph, self.min_cycles
            )
            if not cycle_validation["meets_requirements"]:
                return GenerationError(
                    error_type=ErrorType.TOO_FEW_CYCLES,
                    message=f"Graph requires minimum {self.min_cycles} cycles, found {cycle_validation['cycles_count']}",
                )

            logger.info("Sampling dialogs...")
            sampled_dialogs = self.dialog_sampler.invoke(
                graph, self.model_storage.storage[self.cycle_ends_llm].model, 15
            )
            logger.info(f"Sampled {len(sampled_dialogs)} dialogs")
            if not match_dg_triplets(graph, sampled_dialogs)["value"]:
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED,
                    message="Failed to sample valid dialogs - not all utterances are present",
                )
            if is_greeting_repeated_regex(sampled_dialogs):
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED,
                    message="Failed to sample valid dialogs - Opening phrases are repeated",
                )
            if is_dialog_closed_too_early_regex(sampled_dialogs):
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED,
                    message="Failed to sample valid dialogs - Closing phrases appear in the middle of a dialog",
                )

            theme_validation = is_theme_valid(
                graph,
                self.model_storage.storage[self.theme_validation_llm].model,
                topic,
            )
            if not theme_validation["value"]:
                return GenerationError(
                    error_type=ErrorType.INVALID_THEME,
                    message=f"Theme validation failed: {theme_validation['description']}",
                )

            logger.info("Validating and fixing transitions...")
            transition_validation = self.check_and_fix_transitions(
                graph=graph, max_attempts=self.max_fix_attempts
            )
            logger.info("Finished validating and fixing transitions")

            if not transition_validation["is_valid"]:
                invalid_transitions = transition_validation["validation_details"][
                    "invalid_transitions"
                ]
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message=f"Found {len(invalid_transitions)} invalid transitions "
                    f"after {transition_validation['validation_details']['attempts_made']} fix attempts",
                )

            graph = transition_validation["graph"]
            if transition_validation["validation_details"]["attempts_made"]:
                if not graph.match_edges_nodes():
                    return GenerationError(
                        error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                        message="Generated graph is wrong: edges don't match nodes",
                    )
                graph = graph.remove_duplicated_nodes()
                if graph is None:
                    return GenerationError(
                        error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                        message="Generated graph is wrong: utterances in nodes doubled",
                    )
                logger.info("Sampling dialogs...")
                sampled_dialogs = self.dialog_sampler.invoke(
                    graph, self.model_storage.storage[self.cycle_ends_llm].model, 15
                )
                logger.info("Sampled %d dialogs", len(sampled_dialogs))
                if not match_dg_triplets(graph, sampled_dialogs)["value"]:
                    return GenerationError(
                        error_type=ErrorType.SAMPLING_FAILED,
                        message="Failed to sample valid dialogs - not all utterances are present",
                    )

            return GraphGenerationResult(
                graph=transition_validation["graph"].graph_dict,
                metadata=transition_validation["graph"].metadata,
                topic=topic,
                dialogs=sampled_dialogs,
            )

        except Exception as e:
            logger.error(f"Unexpected error during generation: {str(e)}")
            return GenerationError(
                error_type=ErrorType.GENERATION_FAILED,
                message=f"Unexpected error during generation: {str(e)}",
            )

    def __call__(self, topic: str) -> PipelineResult:
        """Shorthand for generate_and_validate"""
        return self.generate_and_validate(topic)


class LoopedGraphGenerator(TopicGraphGenerator):
    """Graph generator for topic-based dialog generation with model storage support
    Attributes:
        model_storage (ModelStorage): Model storage to take models from.
        generation_llm (str): Name of the model to use for graph generation.
        validation_llm (str): Name of the model to use for validation.
        cycle_ends_llm (str): Name of the model to use for finding cycle ends.
        theme_validation_llm (str): Name of the model to use for theme validation.
        pipeline (GenerationPipeline): Generation pipeline to use for graph generation.
    """

    model_storage: ModelStorage = Field(description="Model storage")
    generation_llm: str = Field(
        description="LLM for graph generation", default="looped_graph_generation_llm:v1"
    )
    validation_llm: str = Field(
        description="LLM for validation", default="looped_graph_validation_llm:v1"
    )
    cycle_ends_llm: str = Field(
        description="LLM for dialog sampler to find cycle ends",
        default="looped_graph_cycle_ends_llm:v1",
    )
    theme_validation_llm: str = Field(
        description="LLM for theme validation",
        default="looped_graph_theme_validation_llm:v1",
    )
    pipeline: GenerationPipeline

    def __init__(
        self,
        model_storage: ModelStorage,
        generation_llm: str = "looped_graph_generation_llm:v1",
        validation_llm: str = "looped_graph_validation_llm:v1",
        cycle_ends_llm: str = "looped_graph_cycle_ends_llm:v1",
        theme_validation_llm: str = "looped_graph_theme_validation_llm:v1",
    ):
        # if model is not in model storage put the default model there
        """
        Initialize a LoopedGraphGenerator instance.

        Args:
            model_storage (ModelStorage): The model storage t take models from.
            generation_llm (str): The LLM to use for graph generation. Defaults to "looped_graph_generation_llm:v1".
            validation_llm (str): The LLM to use for validation. Defaults to "looped_graph_validation_llm:v1".
            cycle_ends_llm (str): The LLM to use for finding cycle ends. Defaults to "looped_graph_cycle_ends_llm:v1".
            theme_validation_llm (str): The LLM to use for theme validation. Defaults to "looped_graph_theme_validation_llm:v1".

        If the specified LLMs are not present in the provided model storage, default models are added.
        """
        model_storage.add(
            key=generation_llm,
            config={
                "model_name": "chatgpt-4o-latest",
                "temperature": 0,
            },
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=validation_llm,
            config={
                "model_name": "gpt-3.5-turbo",
                "temperature": 0,
            },
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=cycle_ends_llm,
            config={
                "model_name": "chatgpt-4o-latest",
                "temperature": 0,
            },
            model_type=ChatOpenAI,
        )

        model_storage.add(
            key=theme_validation_llm,
            config={
                "model_name": "gpt-3.5-turbo",
                "temperature": 0,
            },
            model_type=ChatOpenAI,
        )

        super().__init__(
            model_storage=model_storage,
            generation_llm=generation_llm,
            validation_llm=validation_llm,
            cycle_ends_llm=cycle_ends_llm,
            theme_validation_llm=theme_validation_llm,
            pipeline=GenerationPipeline(
                model_storage=model_storage,
                generation_llm=generation_llm,
                validation_llm=validation_llm,
                cycle_ends_llm=cycle_ends_llm,
                theme_validation_llm=theme_validation_llm,
                generation_prompt=cycle_graph_generation_prompt_informal,
                repair_prompt=cycle_graph_repair_prompt,
            ),
        )

    def invoke(self, topic, seed=42) -> list[dict]:
        # TODO: add docs
        """
        Generates a dialog graph for a given topic using the configured pipeline.

        This method utilizes the pipeline to generate a dialog graph based on the
        specified topic. It logs the process and handles potential errors during
        generation. The method returns a list containing dictionaries with details
        of successfully generated graphs.

        Args:
            topic (str): The topic for which the dialog graph is to be generated.
            seed (int, optional): A seed value to ensure reproducibility of the graph
                generation process. Defaults to 42.

        Returns:
            list[dict]: A list of dictionaries containing the graph, metadata, topic,
            and dialogs for successfully generated graphs.
        """

        logger.info(f"\n{'=' * 50}")
        logger.info("Generating graph for topic: %s", topic)
        logger.info(f"{'=' * 50}")
        successful_generations = []
        try:
            self.pipeline.seed = seed
            result = self.pipeline(topic)

            if isinstance(result, GraphGenerationResult):
                logger.info(f"✅ Successfully generated graph for {topic}")
                successful_generations.append(
                    {
                        "graph": result.graph.model_dump(),
                        "metadata": result.metadata.model_dump(),  # The metadata is already a dictionary
                        "topic": result.topic,
                        "dialogs": [
                            dia.model_dump() for dia in result.dialogs
                        ],  # The dialogs are already dictionaries
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
        # TODO: add docs
        return super().evaluate(*args, report_type=report_type, **kwargs)
