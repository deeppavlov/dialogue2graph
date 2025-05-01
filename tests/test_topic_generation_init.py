import pytest
import dotenv

from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.datasets.complex_dialogues.generation import (
    GenerationPipeline,
    LoopedGraphGenerator,
)

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)

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