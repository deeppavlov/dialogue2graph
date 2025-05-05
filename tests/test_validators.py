import pytest
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from dialog2graph import Dialog
from dialog2graph.metrics.no_llm_validators import (
    is_greeting_repeated_regex,
    is_dialog_closed_too_early_regex,
)
from dialog2graph.metrics.llm_validators import (
    is_greeting_repeated_emb_llm,
    is_dialog_closed_too_early_emb_llm,
)
from dialog2graph.pipelines.model_storage import ModelStorage


dotenv.load_dotenv()


@pytest.fixture
def model_storage():
    model_storage = ModelStorage()
    model_storage.add(
        "my_emb",
        config={"model_name": "BAAI/bge-m3"},
        model_type=HuggingFaceEmbeddings,
    )
    model_storage.add(
        "my_llm", config={"model_name": "gpt-4o-mini"}, model_type=ChatOpenAI
    )
    return (model_storage, "my_emb", "my_llm")


@pytest.fixture
def good_example():
    messages = [
        {"participant": "assistant", "text": "Hello! How can I assist you today?"},
        {"participant": "user", "text": "I need help with my account"},
        {
            "participant": "assistant",
            "text": "Certainly! How can I assist you with your account today?",
        },
        {"participant": "user", "text": "I forgot my password"},
        {
            "participant": "assistant",
            "text": "No problem. I can help you reset your password. Would you like me to send a reset link to your email?",
        },
        {"participant": "user", "text": "Yes, please"},
        {
            "participant": "assistant",
            "text": "I've sent a reset link to your registered email address. Is there anything else I can help you with?",
        },
        {"participant": "user", "text": "Yes, actually I need something else"},
        {
            "participant": "assistant",
            "text": "Certainly! How can I assist you with your account today?",
        },
        {"participant": "user", "text": "I want to close my account"},
        {
            "participant": "assistant",
            "text": "I've noted your issue and forwarded it to our support team. Would you like to do anything else?",
        },
        {"participant": "user", "text": "No, thank you"},
        {"participant": "assistant", "text": "Goodbye! Have a great day!"},
    ]
    return [Dialog().from_list(messages)]


@pytest.fixture
def hello_x2_example():
    messages = [
        {
            "participant": "assistant",
            "text": "Hello! How can I assist you with accessibility accommodations today?",
        },
        {"participant": "user", "text": "I'm looking for accessible facilities."},
        {
            "participant": "assistant",
            "text": "Certainly, let's explore other accommodation options. What are your preferences?",
        },
        {"participant": "user", "text": "That works for me."},
        {
            "participant": "assistant",
            "text": "Great! I've reserved the accommodation for you.",
        },
        {"participant": "user", "text": "I need to make another reservation."},
        {
            "participant": "assistant",
            "text": "Hello! How can I assist you with accessibility accommodations today?",
        },
        {"participant": "user", "text": "I need a wheelchair accessible room."},
        {
            "participant": "assistant",
            "text": "What type of accessibility accommodation do you need?",
        },
        {"participant": "user", "text": "Yes, I need it on the ground floor."},
        {
            "participant": "assistant",
            "text": "Do you have any specific requirements or preferences?",
        },
        {"participant": "user", "text": "I have decided not to proceed."},
        {
            "participant": "assistant",
            "text": "If you need any further assistance, feel free to let me know. Have a great day!",
        },
    ]
    return [Dialog().from_list(messages)]


@pytest.fixture
def bye_x2_example():
    messages = [
        {"participant": "assistant", "text": "Hello! How can I assist you today?"},
        {
            "participant": "user",
            "text": "I want to report an unauthorized subscription renewal.",
        },
        {
            "participant": "assistant",
            "text": "I'm sorry about that. May I have your account number, please?",
        },
        {"participant": "user", "text": "Actually, I don't want to proceed."},
        {
            "participant": "assistant",
            "text": "Understood. If you change your mind, feel free to reach out. Have a good day!",
        },
        {"participant": "user", "text": "Actually, I want to proceed."},
        {
            "participant": "assistant",
            "text": "I'm sorry about that. May I have your account number, please?",
        },
        {"participant": "user", "text": "Sure, it's 123456."},
        {
            "participant": "assistant",
            "text": "Thank you. I'm looking into your account. Please hold on for a moment.",
        },
        {"participant": "user", "text": "Thank you."},
        {
            "participant": "assistant",
            "text": "I've canceled the unauthorized renewal and initiated a refund. Is there anything else I can help you with?",
        },
        {"participant": "user", "text": "No, that's all."},
        {
            "participant": "assistant",
            "text": "Glad I could help. If you need anything else, feel free to ask. Goodbye!",
        },
    ]
    return [Dialog().from_list(messages)]


def test_re_start(good_example, hello_x2_example):
    assert is_greeting_repeated_regex(good_example) is False
    assert is_greeting_repeated_regex(hello_x2_example) is True


def test_re_end(good_example, bye_x2_example):
    assert is_dialog_closed_too_early_regex(good_example) is False
    assert is_dialog_closed_too_early_regex(bye_x2_example) is True


@pytest.mark.skipif(not dotenv.find_dotenv(), reason="No env. file was found")
def test_emb_llm_start(good_example, hello_x2_example, model_storage):
    assert is_greeting_repeated_emb_llm(good_example, *model_storage) is False
    assert is_greeting_repeated_emb_llm(hello_x2_example, *model_storage) is True


@pytest.mark.skipif(not dotenv.find_dotenv(), reason="No env. file was found")
def test_emb_llm_end(good_example, bye_x2_example, model_storage):
    assert is_dialog_closed_too_early_emb_llm(good_example, *model_storage) is False
    assert is_dialog_closed_too_early_emb_llm(bye_x2_example, *model_storage) is True
