"""
Validators
--------------------------
This module contains validators to evaluate dialogs

"""

from typing import List
import re

from dialogue2graph.pipelines.core.dialogue import Dialogue


def _message_has_greeting_re(regex: str, text: str) -> bool:
    return bool(re.match(regex, text, flags=re.IGNORECASE))


def _message_has_closing_re(regex: str, text: str) -> bool:
    return bool(re.search(regex, text, flags=re.IGNORECASE))


def is_greeting_repeated_regex(dialogs: List[Dialogue], regex: str = None) -> bool:
    """
    Checks if greeting is repeated within dialogues using regular expression.
    Args:
        dialogs (List[Dialogue]): Dialog list from graph.
        regex (str): Regular expression to find start turns. Defaults to None, so standard regex is used.
    Returns
        bool: True if greeting has been repeated, False otherwise.
    """
    if not regex:
        regex = r"^hello|^hi|^greetings"
    for dialog in dialogs:
        for i, message in enumerate(dialog.messages):
            if (
                i != 0
                and message.participant == "assistant"
                and _message_has_greeting_re(regex, message.text)
            ):
                return True
    return False


def is_dialog_closed_too_early_regex(
    dialogs: List[Dialogue], regex: str = None
) -> bool:
    """
    Checks if assistant tried to close dialogue in the middle using regular expression.
    Args:
        dialogs (List[Dialogue]): Dialog list from graph.
        regex (str): Regular expression to find end turns. Defaults to None, so standard regex is used.
    Returns
        bool: True if closing appeared too early, False otherwise.
    """
    if not regex:
        regex = r"have a (great|good|nice) day.$|goodbye.$"
    for dialog in dialogs:
        last_turn_idx = len(dialog.messages) - 1
        for i, message in enumerate(dialog.messages):
            if (
                i != last_turn_idx
                and message.participant == "assistant"
                and _message_has_closing_re(regex, message.text)
            ):
                return True
    return False
