import json
import pytest
import dotenv
from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.datasets.augment_dialogues.augmentation import DialogueAugmenter
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.datasets.augment_dialogues.prompts import augmentation_prompt_3_vars
from dialogue2graph.metrics.no_llm_metrics.metrics import is_correct_length, match_roles

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)


@pytest.fixture(scope="session")
def test_data():
    """
    Read JSON data once per pytest session
    (scope="session") to avoid re-reading the file.
    """
    with open("tests/test_dialogue_augmentation_data.json", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def dialogue(test_data):
    """
    Dialogue for the positive scenario.
    """
    raw_dialogue = test_data[0]["dialogue"]
    return Dialogue(**raw_dialogue)


@pytest.fixture
def positive_augmentation_result(test_data):
    """
    Raw augmentation result in dict format.
    """
    return test_data[1]["positive_aug_result"]


@pytest.fixture
def negative_augmentation_result(test_data):
    """
    Raw augmentation result in dict format.
    """
    return test_data[2]["negative_aug_result"]


@pytest.fixture
def positive_augmented_dialogues(test_data):
    """
    Augmented dialogues as Dialogue objects in a list.
    """
    return test_data[3]["positive_aug_dialogues"]


@pytest.fixture
def negative_augmented_dialogue(test_data):
    """
    Augmented dialogues as Dialogue objects in a list.
    """
    return test_data[4]["negative_aug_dialogue"]


@pytest.fixture(scope="module")
def ms():
    """Shared model storage with test configurations"""
    storage = ModelStorage()
    storage.add(
        key="generation-llm",
        config={"model_name": "gpt-4o-mini-2024-07-18", "temperature": 0.7},
        model_type="llm",
    )
    storage.add(
        key="formatting-llm",
        config={"model_name": "gpt-3.5-turbo", "temperature": 0.7},
        model_type="llm",
    )
    return storage


@pytest.fixture
def augmenter(ms):
    """Preconfigured augmenter for most tests"""
    return DialogueAugmenter(
        model_storage=ms,
        generation_llm="generation-llm",
        formatting_llm="formatting-llm",
    )


def test_augmentation(augmenter, dialogue):
    """Test full augmentation workflow"""
    result = augmenter.invoke(
        dialogue=dialogue,
        topic="Responding to DMs on Instagram/Facebook.",
        prompt=augmentation_prompt_3_vars,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(aug_dialogue, Dialogue) for aug_dialogue in result)


def test_dialogues_creation_positive(augmenter, positive_augmentation_result):
    """Test dialogue creation from raw API response"""
    dialogues = augmenter._create_dialogues(positive_augmentation_result)

    assert len(dialogues) == 3
    assert all(isinstance(aug_dialogue, Dialogue) for aug_dialogue in dialogues)


def test_dialogues_creation_negative(augmenter, negative_augmentation_result):
    """Test dialogue creation from raw API response"""
    result = augmenter._create_dialogues(negative_augmentation_result)
    assert "Creating a list of Dialogue objects failed" in result


def test_metric_validation_positive(dialogue, positive_augmented_dialogues):
    """Validate core metrics with known-good data"""
    aug_dialogue = Dialogue(**positive_augmented_dialogues[0])

    assert match_roles(dialogue, aug_dialogue)
    assert is_correct_length(dialogue, aug_dialogue)


def test_metric_validation_negative(dialogue, negative_augmented_dialogue):
    """Validate core metrics with known-good data"""
    aug_dialogue = Dialogue(**negative_augmented_dialogue)
    assert is_correct_length(dialogue, aug_dialogue) is False


def test_error_handling_with_invalid_llm(ms, dialogue):
    """Test error handling with invalid LLM configuration"""
    ms.add(
        key="invalid-llm",
        config={"model_name": "outdated-model", "temperature": 0.7},
        model_type="llm",
    )
    augmenter = DialogueAugmenter(
        model_storage=ms, generation_llm="invalid-llm", formatting_llm="invalid-llm"
    )
    result = augmenter.invoke(dialogue=dialogue, prompt=augmentation_prompt_3_vars)
    assert "Augmentation failed" in result


def test_empty_dialogue(augmenter):
    """Test error handling with empty dialogue"""
    result = augmenter.invoke(
        dialogue=Dialogue(messages=[]), prompt=augmentation_prompt_3_vars
    )
    assert "Preprocessing failed: no messages" in result


def test_empty_prompt(augmenter, dialogue):
    """Test error handling with empty prompt"""
    result = augmenter.invoke(dialogue=dialogue, prompt="")
    assert "Preprocessing failed: prompt" in result
