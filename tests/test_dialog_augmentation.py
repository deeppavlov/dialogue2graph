import json
import pytest
import dotenv
from dialog2graph.pipelines.core.dialog import Dialog
from dialog2graph.datasets.augment_dialogs.augmentation import DialogAugmenter
from dialog2graph.pipelines.model_storage import ModelStorage
from dialog2graph.datasets.augment_dialogs.prompts import augmentation_prompt_3_vars
from dialog2graph.metrics.no_llm_metrics.metrics import is_correct_length, match_roles

from langchain_openai import ChatOpenAI

dotenv.load_dotenv()
if not dotenv.find_dotenv():
    pytest.skip("Skipping test as .env file is not found", allow_module_level=True)


@pytest.fixture(scope="session")
def test_data():
    """
    Read JSON data once per pytest session
    (scope="session") to avoid re-reading the file.
    """
    with open("tests/test_dialog_augmentation_data.json", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture
def dialog(test_data):
    """
    Dialog for the positive scenario.
    """
    raw_dialog = test_data[0]["dialog"]
    return Dialog(**raw_dialog)


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
def positive_augmented_dialogs(test_data):
    """
    Augmented dialogs as Dialog objects in a list.
    """
    return test_data[3]["positive_aug_dialogs"]


@pytest.fixture
def negative_augmented_dialog(test_data):
    """
    Augmented dialogs as Dialog objects in a list.
    """
    return test_data[4]["negative_aug_dialog"]


@pytest.fixture(scope="module")
def ms():
    """Shared model storage with test configurations"""
    storage = ModelStorage()
    storage.add(
        key="generation-llm",
        config={"model_name": "gpt-4o-mini-2024-07-18", "temperature": 0.7},
        model_type=ChatOpenAI,
    )
    storage.add(
        key="formatting-llm",
        config={"model_name": "gpt-3.5-turbo", "temperature": 0.7},
        model_type=ChatOpenAI,
    )
    return storage


@pytest.fixture
def augmenter(ms):
    """Preconfigured augmenter for most tests"""
    return DialogAugmenter(
        model_storage=ms,
        generation_llm="generation-llm",
        formatting_llm="formatting-llm",
    )


def test_augmentation(augmenter, dialog):
    """Test full augmentation workflow"""
    result = augmenter.invoke(
        dialog=dialog,
        topic="Responding to DMs on Instagram/Facebook.",
        prompt=augmentation_prompt_3_vars,
    )

    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(aug_dialog, Dialog) for aug_dialog in result)


def test_dialogs_creation_positive(augmenter, positive_augmentation_result):
    """Test dialog creation from raw API response"""
    dialogs = augmenter._create_dialogs(positive_augmentation_result)

    assert len(dialogs) == 3
    assert all(isinstance(aug_dialog, Dialog) for aug_dialog in dialogs)


def test_dialogs_creation_negative(augmenter, negative_augmentation_result):
    """Test dialog creation from raw API response"""
    result = augmenter._create_dialogs(negative_augmentation_result)
    assert "Creating a list of Dialog objects failed" in result


def test_metric_validation_positive(dialog, positive_augmented_dialogs):
    """Validate core metrics with known-good data"""
    aug_dialog = Dialog(**positive_augmented_dialogs[0])

    assert match_roles(dialog, aug_dialog)
    assert is_correct_length(dialog, aug_dialog)


def test_metric_validation_negative(dialog, negative_augmented_dialog):
    """Validate core metrics with known-good data"""
    aug_dialog = Dialog(**negative_augmented_dialog)
    assert is_correct_length(dialog, aug_dialog) is False


def test_error_handling_with_invalid_llm(ms, dialog):
    """Test error handling with invalid LLM configuration"""
    ms.add(
        key="invalid-llm",
        config={"model_name": "outdated-model", "temperature": 0.7},
        model_type=ChatOpenAI,
    )
    augmenter = DialogAugmenter(
        model_storage=ms, generation_llm="invalid-llm", formatting_llm="invalid-llm"
    )
    result = augmenter.invoke(dialog=dialog, prompt=augmentation_prompt_3_vars)
    assert "Augmentation failed" in result


def test_empty_dialog(augmenter):
    """Test error handling with empty dialog"""
    result = augmenter.invoke(
        dialog=Dialog(messages=[]), prompt=augmentation_prompt_3_vars
    )
    assert "Preprocessing failed: no messages" in result


def test_empty_prompt(augmenter, dialog):
    """Test error handling with empty prompt"""
    result = augmenter.invoke(dialog=dialog, prompt="")
    assert "Preprocessing failed: prompt" in result
