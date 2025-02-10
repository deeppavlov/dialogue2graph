from dataclasses import dataclass
from typing import Optional, Dict, Any
import networkx as nx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate

# from chatsky_llm_autoconfig.algorithms.topic_graph_generation import CycleGraphGenerator
# from chatsky_llm_autoconfig.algorithms.dialogue_generation import RecursiveDialogueSampler
# from chatsky_llm_autoconfig.metrics.automatic_metrics import all_utterances_present
# from chatsky_llm_autoconfig.metrics.llm_metrics import are_triples_valid, is_theme_valid
# from chatsky_llm_autoconfig.graph import BaseGraph
# from chatsky_llm_autoconfig.prompts import cycle_graph_generation_prompt_enhanced, cycle_graph_repair_prompt
from openai import BaseModel

from enum import Enum
from typing import Union

from dialogue2graph.pipelines.core.schemas import GraphGenerationResult


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


@dataclass
class GenerationPipeline:
    generation_model: BaseChatModel
    validation_model: BaseChatModel
    graph_generator: CycleGraphGenerator
    generation_prompt: PromptTemplate
    repair_prompt: PromptTemplate
    min_cycles: int = 2
    max_fix_attempts: int = 3

    def __init__(
        self,
        generation_model: BaseChatModel,
        validation_model: BaseChatModel,
        generation_prompt: Optional[PromptTemplate] = None,
        repair_prompt: Optional[PromptTemplate] = None,
        min_cycles: int = 0,
        max_fix_attempts: int = 2,
    ):
        self.generation_model = generation_model
        self.validation_model = validation_model
        self.graph_generator = CycleGraphGenerator()
        self.dialogue_sampler = RecursiveDialogueSampler()

        self.generation_prompt = generation_prompt or cycle_graph_generation_prompt_enhanced
        self.repair_prompt = repair_prompt or cycle_graph_repair_prompt

        self.min_cycles = min_cycles
        self.max_fix_attempts = max_fix_attempts

    def validate_graph_cycle_requirement(self, graph: BaseGraph, min_cycles: int = 2) -> Dict[str, Any]:
        """
        Проверяет граф на соответствие требованиям по количеству циклов
        """
        print("\n🔍 Checking graph requirements...")

        try:
            cycles = list(nx.simple_cycles(graph.graph))
            cycles_count = len(cycles)

            print(f"🔄 Found {cycles_count} cycles in the graph:")
            for i, cycle in enumerate(cycles, 1):
                print(f"Cycle {i}: {' -> '.join(map(str, cycle + [cycle[0]]))}")

            meets_requirements = cycles_count >= min_cycles

            if not meets_requirements:
                print(f"❌ Graph doesn't meet cycle requirements (minimum {min_cycles} cycles needed)")
            else:
                print("✅ Graph meets cycle requirements")

            return {"meets_requirements": meets_requirements, "cycles": cycles, "cycles_count": cycles_count}

        except Exception as e:
            print(f"❌ Validation error: {str(e)}")
            raise

    def check_and_fix_transitions(self, graph: BaseGraph, max_attempts: int = 3) -> Dict[str, Any]:
        """
        Проверяет переходы в графе и пытается исправить невалидные через LLM
        """
        print("Validating initial graph")

        initial_validation = are_triples_valid(graph, self.validation_model, return_type="detailed")
        if initial_validation["is_valid"]:
            return {"is_valid": True, "graph": graph, "validation_details": {"invalid_transitions": [], "attempts_made": 0, "fixed_count": 0}}

        initial_invalid_count = len(initial_validation["invalid_transitions"])
        current_graph = graph
        current_attempt = 0

        while current_attempt < max_attempts:
            print(f"\n🔄 Fix attempt {current_attempt + 1}/{max_attempts}")

            try:
                # Используем generation_model для исправления графа
                current_graph = self.graph_generator.invoke(
                    model=self.generation_model,
                    prompt=self.repair_prompt,
                    invalid_transitions=initial_validation["invalid_transitions"],
                    graph_json=current_graph.graph_dict,
                )

                # Проверяем исправленный граф используя validation_model
                validation = are_triples_valid(current_graph, self.validation_model, return_type="detailed")
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
        """
        Generates and validates a dialogue graph for given topic
        """
        try:
            # 1. Generate initial graph
            print("Generating Graph ...")
            graph = self.graph_generator.invoke(model=self.generation_model, prompt=self.generation_prompt, topic=topic)

            # 2. Validate cycles
            cycle_validation = self.validate_graph_cycle_requirement(graph, self.min_cycles)
            if not cycle_validation["meets_requirements"]:
                return GenerationError(
                    error_type=ErrorType.TOO_MANY_CYCLES,
                    message=f"Graph requires minimum {self.min_cycles} cycles, found {cycle_validation['cycles_count']}",
                )

            # 3. Generate and validate dialogues
            print("Sampling dialogues...")
            sampled_dialogues = self.dialogue_sampler.invoke(graph, 15)
            print(f"Sampled {len(sampled_dialogues)} dialogues")
            print(sampled_dialogues)
            if not all_utterances_present(graph, sampled_dialogues):
                return GenerationError(
                    error_type=ErrorType.SAMPLING_FAILED, message="Failed to sample valid dialogues - not all utterances are present"
                )

            # 4. Validate theme
            theme_validation = is_theme_valid(graph, self.validation_model, topic)
            if not theme_validation["value"]:
                return GenerationError(error_type=ErrorType.INVALID_THEME, message=f"Theme validation failed: {theme_validation['description']}")

            # 5. Validate and fix transitions
            print("Validating and fixing transitions...")
            transition_validation = self.check_and_fix_transitions(graph=graph, max_attempts=self.max_fix_attempts)

            if not transition_validation["is_valid"]:
                invalid_transitions = transition_validation["validation_details"]["invalid_transitions"]
                return GenerationError(
                    error_type=ErrorType.INVALID_GRAPH_STRUCTURE,
                    message=f"Found {len(invalid_transitions)} invalid transitions after {transition_validation['validation_details']['attempts_made']} fix attempts",
                )

            # All validations passed - return successful result
            return GraphGenerationResult(graph=transition_validation["graph"].graph_dict, topic=topic, dialogues=sampled_dialogues)

        except Exception as e:
            return GenerationError(error_type=ErrorType.GENERATION_FAILED, message=f"Unexpected error during generation: {str(e)}")

    def __call__(self, topic: str) -> PipelineResult:
        """Shorthand for generate_and_validate"""
        return self.generate_and_validate(topic)
