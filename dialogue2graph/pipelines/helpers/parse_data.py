# from dialogue2graph.pipelines.core.pipeline import Pipeline as BasePipeline
import json
from typing import List, Optional
from pydantic import BaseModel
from pathlib import PosixPath
import logging
from pydantic import TypeAdapter, ValidationError
from dialogue2graph.pipelines.core.dialogue import Dialogue, DialogueMessage
from dialogue2graph.pipelines.core.algorithms import InputParser
from dialogue2graph.pipelines.core import schemas
from dialogue2graph.pipelines.core import graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# RawDialogsType = Dialogue | list[Dialogue] | dict | list[list] | list[dict] | str
RawDialogsType = dict | list[list] | list[dict] | Dialogue | list[Dialogue] | str
RawGraphType = schemas.DialogueGraph | dict


class PipelineRawDataType(BaseModel):
    dialogs: RawDialogsType
    supported_graph: Optional[schemas.DialogueGraph] | None = None
    true_graph: Optional[schemas.DialogueGraph] | None = None


class PipelineDataType(BaseModel):
    dialogs: list[Dialogue]
    supported_graph: Optional[graph.Graph] | None = None
    true_graph: Optional[graph.Graph] | None = None


class DataParser(InputParser):

    def _validate_raw_graph(self, raw_graph: schemas.DialogueGraph) -> graph.Graph | None:
        if raw_graph is not None:
            try:
                graph_validation = TypeAdapter(schemas.DialogueGraph).validate_python(raw_graph)
            except ValidationError as e:
                logger.error(f"Input data validation error: {e}")
                return None
            return graph.Graph(graph_validation.model_dump())
        else:
            return None

    def _validate_raw_dialogs(self, raw_dialogs: RawDialogsType) -> list[Dialogue]:
        try:
            dialog_validation = TypeAdapter(
                List[DialogueMessage] | List[List[DialogueMessage]] | Dialogue | List[Dialogue] | PosixPath
            ).validate_python(raw_dialogs)
        except ValidationError as e:
            logger.error(f"Input data validation error: {e}")
            return []
        return dialog_validation

    def invoke(self, raw_data: PipelineRawDataType) -> PipelineDataType:
        """Validate and convert user's data into list of Dialogue
        Input data can be as follows:
        [{'participant': user or assistant, 'text': text}]
        {'messages': [{'participant': user or assistant, 'text': text}]}
        [[{'participant': user or assistant, 'text': text}]]
        [{'messages': [{'participant': user or assistant, 'text': text}]}]
        or same in json file presented by file path
        return list, or empty list when error
        """

        dialog_validation = self._validate_raw_dialogs(raw_data.dialogs)
        if isinstance(dialog_validation, PosixPath):
            if raw_data.dialogs.endswith(".json"):
                try:
                    with open(raw_data.dialogs) as f:
                        raw_dialogs = json.load(f)
                except OSError as e:
                    logger.error("Error %s reading file: %s", e, raw_data.dialogs)
                    return []
                if isinstance(raw_dialogs, dict):
                    raw_dialogs = [raw_dialogs]
                dialog_validation = self._validate_raw_dialogs(raw_dialogs)
            else:
                logger.error("File extension is not json: %s", raw_data.dialogs)
                return []

        if isinstance(dialog_validation, Dialogue):
            dialogues = [dialog_validation]
        elif isinstance(dialog_validation, List) and dialog_validation:
            if isinstance(dialog_validation[0], Dialogue):
                dialogues = dialog_validation
            elif isinstance(dialog_validation[0], DialogueMessage):
                dialogues = [Dialogue(messages=dialog_validation)]
            elif isinstance(dialog_validation[0], List) and isinstance(dialog_validation[0][0], DialogueMessage):
                dialogues = [Dialogue(messages=dialogue) for dialogue in dialog_validation]
        else:
            dialogues = []
        supported_graph_validation = self._validate_raw_graph(raw_data.supported_graph)
        true_graph_validation = self._validate_raw_graph(raw_data.true_graph)

        return PipelineDataType(dialogs=dialogues, supported_graph=supported_graph_validation, true_graph=true_graph_validation)

    def evaluate(self, *args, report_type="dict", **kwargs):
        return super().evaluate(*args, report_type=report_type, **kwargs)
