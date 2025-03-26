from dialogue2graph.datasets.complex_dialogues.generation import (
    CycleGraphGenerator,
    GenerationPipeline,
    LoopedGraphGenerator,
    ErrorType,
    GenerationError,
)
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.core.graph import Graph, BaseGraph
from dialogue2graph.pipelines.core.dialogue_sampling import RecursiveDialogueSampler

# from dialogue2graph.metrics.automatic_metrics import all_utterances_present
from langchain_core.language_models.chat_models import BaseChatModel


class MockChatModel(BaseChatModel):
    """Mock chat model for testing"""

    def invoke(self, *args, **kwargs):
        return {"generations": [{"text": "test response"}]}

    def _llm_type(self):
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_cycle_graph_generator_init():
    """Test CycleGraphGenerator initialization"""
    generator = CycleGraphGenerator()
    assert isinstance(generator, CycleGraphGenerator)


def test_generation_pipeline_init():
    """Test GenerationPipeline initialization"""
    model = MockChatModel()
    pipeline = GenerationPipeline(
        generation_model=model, theme_validation_model=model, validation_model=model, generation_prompt=None, repair_prompt=None
    )
    assert isinstance(pipeline, GenerationPipeline)


def test_looped_graph_generator_init():
    """Test LoopedGraphGenerator initialization"""
    model = MockChatModel()
    generator = LoopedGraphGenerator(generation_model=model, validation_model=model, theme_validation_model=model)
    assert isinstance(generator, LoopedGraphGenerator)


def test_dialogue_init():
    """Test Dialogue initialization"""
    messages = [{"participant": "assistant", "text": "Hello"}, {"participant": "user", "text": "Hi"}]
    dialogue = Dialogue.from_list(messages)
    assert isinstance(dialogue, Dialogue)
    assert len(dialogue.messages) == 2


def test_graph_init():
    """Test Graph initialization"""
    graph_dict = {"nodes": [{"id": 1, "label": "start", "is_start": True, "utterances": ["Hello"]}], "edges": []}
    graph = Graph(graph_dict=graph_dict)
    assert isinstance(graph, BaseGraph)


def test_recursive_dialogue_sampler_init():
    """Test RecursiveDialogueSampler initialization"""
    sampler = RecursiveDialogueSampler()
    assert isinstance(sampler, RecursiveDialogueSampler)


def test_error_type_enum():
    """Test ErrorType enum initialization"""
    assert ErrorType.INVALID_GRAPH_STRUCTURE == "invalid_graph_structure"
    assert ErrorType.TOO_MANY_CYCLES == "too_many_cycles"
    assert ErrorType.SAMPLING_FAILED == "sampling_failed"
    assert ErrorType.INVALID_THEME == "invalid_theme"
    assert ErrorType.GENERATION_FAILED == "generation_failed"


def test_generation_error_init():
    """Test GenerationError initialization"""
    error = GenerationError(error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Test error")
    assert isinstance(error, GenerationError)
    assert error.error_type == ErrorType.INVALID_GRAPH_STRUCTURE
    assert error.message == "Test error"
