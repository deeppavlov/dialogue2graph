"""
Validators
--------------------------
This module contains validators to evaluate dialogs

"""

from typing import List

from pydantic import BaseModel, Field

from dialogue2graph.pipelines.core.dialogue import Dialogue
from dialogue2graph.pipelines.model_storage import ModelStorage
from dialogue2graph.metrics.similarity import compare_strings

from langchain_core.language_models.chat_models import BaseChatModel
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


START_TURNS = [
    "Greetings! How can I assist you?",
    "Greetings! How can I help you?",
    "Greetings! Would you like to do this?",
    "Greetings! Could you tell me this?",
    "Hello! How can I assist you?",
    "Hello! How can I help you?",
    "Hello! Would you like to do this?",
    "Hello! Could you tell me this?",
    "Hi! How can I assist you?",
    "Hi! How can I help you?",
    "Hi! Would you like to do this?",
    "Hi! Could you tell me this?",
    "Welcome to our assistant service! How can I assist you?",
    "Welcome to our assistant service! How can I help you?",
    "Welcome to our assistant service! Would you like to do this?",
    "Welcome to our assistant service! Could you tell me this?",
]

END_TURNS = [
    "Thank you for contacting us. Have a great day!",
    "You're welcome! Have a great day.",
    "Request confirmed. We're here to help if you have any other needs.",
    "You're welcome! Have a great day!",
    "Alright, if you need any further assistance, feel free to reach out. Have a great day!",
    "Alright, feel free to reach out if you need anything else. Have a great day!",
    "Alright, if you need anything else, feel free to reach out. Have a great day!",
    "I'm sorry to see you go. Your subscription has been canceled. If you have any feedback, feel free to reach out to us.",
    "Alright, if you have any other questions in the future, feel free to reach out. Have a great day!",
    "Alright, if you need any further assistance, feel free to reach out. Have a great presentation!",
]

START_THRESHOLD = 0.2
END_THRESHOLD = 0.2


def _message_has_greeting_llm(model: BaseChatModel, text: str) -> bool:
    class OpeningValidation(BaseModel):
        isOpening: bool = Field(
            description="Whether the given utterance is considered greeting or not"
        )

    start_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
    You are given a dialog turn.
    TURN: {text}
    EVALUATE:
    - Does the turn contain greeting phrases used to open a conversation?

    Reply in JSON format:
    {{"isOpening": true or false}}
    """,
    )
    parser = PydanticOutputParser(pydantic_object=OpeningValidation)
    opening_val_chain = start_prompt | model | parser
    result = opening_val_chain.invoke({"text": text})
    return result.isOpening


def _message_has_closing_llm(model: BaseChatModel, text: str) -> bool:
    class ClosingValidation(BaseModel):
        isClosing: bool = Field(
            description="Whether the given utterance is considered closing or not"
        )

    close_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
    You are given a dialog turn.
    TURN: {text}
    EVALUATE:
    - Does the turn contain phrases used to close a conversation?

    Reply in JSON format:
    {{"isClosing": true or false}}
    """,
    )
    parser = PydanticOutputParser(pydantic_object=ClosingValidation)
    closing_val_chain = close_prompt | model | parser
    result = closing_val_chain.invoke({"text": text})
    return result.isClosing


def is_greeting_repeated_emb_llm(
    dialogs: List[Dialogue],
    model_storage: ModelStorage,
    embedder_name: str,
    llm_name: str,
    starts: list = None,
) -> bool:
    """
    Checks if greeting is repeated within dialogues using pairwise distance and LLM assessment.
    Args:
        dialogs (List[Dialogue]): Dialog list from graph.
        model_storage (ModelStorage): Model storage containing embedder and LLM model for evaluation.
        embedder_name (str): Name of embedder model in model storage (ModelStorage).
        llm_name (str): Name of LLM in model storage (ModelStorage).
        starts (list): List of opening phrases. Defaults to None, so standard opening phrases are used.
    Returns
        bool: True if greeting has been repeated, False otherwise.
    """
    if not starts:
        starts = START_TURNS

    if model_storage.storage.get(embedder_name):
        if not model_storage.storage.get(embedder_name).model_type == "emb":
            raise TypeError(f"The {embedder_name} model is not an embedder")
        embedder_model = model_storage.storage[embedder_name].model
    else:
        raise KeyError(
            f"The embedder {embedder_name} not found in the given ModelStorage"
        )

    if model_storage.storage.get(llm_name):
        if not model_storage.storage.get(llm_name).model_type == "llm":
            raise TypeError(f"The {llm_name} model is not an LLM")
        llm_model = model_storage.storage[llm_name].model
    else:
        raise KeyError(f"The LLM {llm_name} not found in the given ModelStorage")

    for dialog in dialogs:
        for i, message in enumerate(dialog.messages):
            if i != 0 and message.participant == "assistant":
                message_is_start = [
                    compare_strings(
                        start,
                        message.text,
                        embedder=embedder_model,
                        embedder_th=START_THRESHOLD,
                    )
                    for start in starts
                ]
                if any(message_is_start):
                    llm_eval = _message_has_greeting_llm(llm_model, message.text)
                    if llm_eval:
                        return True

    return False


def is_dialog_closed_too_early_emb_llm(
    dialogs: List[Dialogue],
    model_storage: ModelStorage,
    embedder_name: str,
    llm_name: str,
    ends: list = None,
) -> bool:
    """
    Checks if assistant tried to close dialogue in the middle using pairwise distance and LLM assessment.
    Args:
        dialogs (List[Dialogue]): Dialog list from graph.
        model_storage (ModelStorage): Model storage containing embedder and LLM model for evaluation.
        embedder_name (str): Name of embedder model in model storage (ModelStorage).
        llm_name (str): Name of LLM in model storage (ModelStorage).
        ends (list): List of closing phrases. Defaults to None, so standard closing phrases are used.
    Returns
        bool: True if greeting has been repeated, False otherwise.
    """
    if not ends:
        ends = END_TURNS

    if model_storage.storage.get(embedder_name):
        if not model_storage.storage.get(embedder_name).model_type == "emb":
            raise TypeError(f"The {embedder_name} model is not an embedder")
        embedder_model = model_storage.storage[embedder_name].model
    else:
        raise KeyError(
            f"The embedder {embedder_name} not found in the given ModelStorage"
        )

    if model_storage.storage.get(llm_name):
        if not model_storage.storage.get(llm_name).model_type == "llm":
            raise TypeError(f"The {llm_name} model is not an LLM")
        llm_model = model_storage.storage[llm_name].model
    else:
        raise KeyError(f"The LLM {llm_name} not found in the given ModelStorage")

    for dialog in dialogs:
        last_turn_idx = len(dialog.messages) - 1
        for i, message in enumerate(dialog.messages):
            if i != last_turn_idx and message.participant == "assistant":
                message_is_end = [
                    compare_strings(
                        end,
                        message.text,
                        embedder=embedder_model,
                        embedder_th=END_THRESHOLD,
                    )
                    for end in ends
                ]
                if any(message_is_end):
                    llm_eval = _message_has_closing_llm(llm_model, message.text)
                    if llm_eval:
                        return True

    return False
