from dialog2graph.datasets.complex_dialogs.generation import (
    CycleGraphGenerator,
    GenerationPipeline,
    LoopedGraphGenerator,
    ErrorType,
    GenerationError,
)
from dialog2graph.pipelines.core.dialog import Dialog
from dialog2graph.pipelines.core.graph import Graph, BaseGraph
from dialog2graph.pipelines.core.dialog_sampling import RecursiveDialogSampler
from dialog2graph.pipelines.model_storage import ModelStorage

# from dialog2graph.metrics.automatic_metrics import all_utterances_present


def test_cycle_graph_generator_init():
    """Test CycleGraphGenerator initialization"""
    generator = CycleGraphGenerator()
    assert isinstance(generator, CycleGraphGenerator)


def test_generation_pipeline_init():
    """Test GenerationPipeline initialization"""
    pipeline = GenerationPipeline(
        model_storage=ModelStorage(),
        generation_llm="generation_llm",
        validation_llm="validation_llm",
        cycle_ends_llm="cycle_ends_llm",
        theme_validation_llm="theme_validation_llm",
        generation_prompt=None,
        repair_prompt=None,
    )
    assert isinstance(pipeline, GenerationPipeline)


def test_looped_graph_generator_init():
    """Test LoopedGraphGenerator initialization"""
    generator = LoopedGraphGenerator(ModelStorage())
    assert isinstance(generator, LoopedGraphGenerator)


def test_dialog_init():
    """Test Dialog initialization"""
    messages = [
        {"participant": "assistant", "text": "Hello"},
        {"participant": "user", "text": "Hi"},
    ]
    dialog = Dialog.from_list(messages)
    assert isinstance(dialog, Dialog)
    assert len(dialog.messages) == 2


def test_graph_init():
    """Test Graph initialization"""
    graph_dict = {
        "nodes": [
            {"id": 1, "label": "start", "is_start": True, "utterances": ["Hello"]}
        ],
        "edges": [],
    }
    graph = Graph(graph_dict=graph_dict)
    assert isinstance(graph, BaseGraph)


def test_recursive_dialog_sampler_init():
    """Test RecursiveDialogSampler initialization"""
    sampler = RecursiveDialogSampler()
    assert isinstance(sampler, RecursiveDialogSampler)


def test_error_type_enum():
    """Test ErrorType enum initialization"""
    assert ErrorType.INVALID_GRAPH_STRUCTURE == "invalid_graph_structure"
    assert ErrorType.TOO_FEW_CYCLES == "too_few_cycles"
    assert ErrorType.SAMPLING_FAILED == "sampling_failed"
    assert ErrorType.INVALID_THEME == "invalid_theme"
    assert ErrorType.GENERATION_FAILED == "generation_failed"


def test_generation_error_init():
    """Test GenerationError initialization"""
    error = GenerationError(
        error_type=ErrorType.INVALID_GRAPH_STRUCTURE, message="Test error"
    )
    assert isinstance(error, GenerationError)
    assert error.error_type == ErrorType.INVALID_GRAPH_STRUCTURE
    assert error.message == "Test error"
