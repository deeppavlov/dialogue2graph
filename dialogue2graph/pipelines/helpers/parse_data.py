# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
import json
from typing import List
from pathlib import PosixPath
import logging
from pydantic import TypeAdapter, ValidationError
from dialogue2graph.pipelines.core.dialogue import Dialogue, DialogueMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_data(data: Dialogue | list[Dialogue] | dict | list[list] | list[dict] | str) -> List[Dialogue]:
    """Validate and convert user's data into list of Dialogue
    Input data can be as follows:
    [{'participant': user or assistant, 'text': text}]
    {'messages': [{'participant': user or assistant, 'text': text}]}
    [[{'participant': user or assistant, 'text': text}]]
    [{'messages': [{'participant': user or assistant, 'text': text}]}]
    or same in json file presented by file path
    return list, or empty list when error
    """

    try:
        validation = TypeAdapter(Dialogue | List[DialogueMessage] | List[List[DialogueMessage]] | PosixPath | List[Dialogue]).validate_python(data)
    except ValidationError as e:
        logger.error(f"Input data validation error: {e}")
        return []
    if isinstance(validation, PosixPath) and data.endswith(".json"):
        with open(data) as f:
            dialogues = json.load(f)
            if isinstance(dialogues, dict):
                dialogues = [dialogues]
        try:
            validation = TypeAdapter(Dialogue | List[DialogueMessage] | List[List[DialogueMessage]] | List[Dialogue]).validate_python(dialogues)
        except ValidationError as e:
            logger.error(f"File {data} data validation error: {e}")
            return []

    if isinstance(validation, Dialogue):
        dialogues = [validation]
    elif isinstance(validation, List) and len(validation) > 0:
        if isinstance(validation[0], Dialogue):
            dialogues = validation
        elif isinstance(validation[0], DialogueMessage):
            dialogues = [Dialogue(messages=validation)]
        elif isinstance(validation[0], List) and isinstance(validation[0][0], DialogueMessage):
            dialogues = [Dialogue(messages=dialogue) for dialogue in validation]
    else:
        dialogues = []
    return dialogues
