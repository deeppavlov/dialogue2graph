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
RawDialogsType = dict | list[list] | list[dict] | Dialogue | list[Dialogue] | PosixPath
ValidatedDialogType = List[DialogueMessage] | List[List[DialogueMessage]] | Dialogue | List[Dialogue]
RawGraphType = schemas.DialogueGraph | dict | PosixPath
ValidatedGraphType = schemas.DialogueGraph | None

class PipelineRawDataType(BaseModel):
    dialogs: RawDialogsType
    supported_graph: Optional[schemas.DialogueGraph] | None = None
    true_graph: Optional[schemas.DialogueGraph] | None = None

class PipelineDataType(BaseModel):
    dialogs: list[Dialogue]
    supported_graph: Optional[graph.Graph] | None = None
    true_graph: Optional[graph.Graph] | None = None


class RawDGParser(InputParser):

    def _validate_raw_graph(self, raw_graph: RawGraphType) -> ValidatedGraphType | PosixPath:
        if raw_graph is not None:
            try:
                graph_validation = TypeAdapter(schemas.DialogueGraph | PosixPath).validate_python(raw_graph)
            except ValidationError as e:
                logger.error(f"Input data validation error: {e}")
                return None
            return graph_validation
        else:
            return None

    def _validate_raw_dialogs(self, raw_dialogs: RawDialogsType) -> ValidatedDialogType | PosixPath:
        try:
            dialog_validation = TypeAdapter(ValidatedDialogType | PosixPath).validate_python(raw_dialogs)
        except ValidationError as e:
            logger.error(f"Input data validation error: {e}")
            return []
        return dialog_validation

    def _get_dialogues_from_file(self, file_path: PosixPath) -> ValidatedDialogType:
        if file_path.suffix == ".json":
            try:
                with open(file_path) as f:
                    raw_dialogs = json.load(f)
            except OSError as e:
                logger.error("Error %s reading file: %s", e, file_path)
                return []
            if not isinstance(raw_dialogs, dict):
                logger.error("Data is not dict in json file: %s", file_path)
                return []  
            if "dialogs" not in raw_dialogs:
                logger.error("No 'dialogs' key in json file: %s", file_path)
                return []
            raw_dialogs = raw_dialogs["dialogs"]           
            if isinstance(raw_dialogs, dict):
                raw_dialogs = [raw_dialogs]
            return self._validate_raw_dialogs(raw_dialogs)
        else:
            logger.error("File extension is not json: %s", file_path)
            return []

    def _get_graph_from_file(self, file_path: PosixPath, key_name: str) -> ValidatedGraphType:
        if file_path.suffix == ".json":
            try:
                with open(file_path) as f:
                    raw_graph = json.load(f)
            except OSError as e:
                logger.error("Error %s reading file: %s", e, file_path)
                return None
            if not isinstance(raw_graph, dict):
                logger.error("Data is not dict in json file: %s", file_path)
                return None
            if key_name not in raw_graph:
                logger.error("No %s key in json file: %s", key_name, file_path)
                return None
            raw_graph = raw_graph[key_name]   
            if isinstance(raw_graph, list) and raw_graph:
                raw_graph = raw_graph[0]
            return self._validate_raw_dialogs(raw_graph)
        else:
            logger.error("File extension is not json: %s", file_path)
            return None

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
            dialog_validation = self._get_dialogues_from_file(raw_data.dialogs)

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
        if isinstance(supported_graph_validation , PosixPath):
            supported_graph_validation  = self._get_graph_from_file(raw_data.supported_graph, "graph")

        true_graph_validation = self._validate_raw_graph(raw_data.true_graph)
        if isinstance(true_graph_validation , PosixPath):
            true_graph_validation  = self._get_graph_from_file(raw_data.true_graph, "true_graph")

        if supported_graph_validation is not None:
            supported_graph_validation = graph.Graph(supported_graph_validation.model_dump())
        if true_graph_validation is not None:
            true_graph_validation = graph.Graph(true_graph_validation.model_dump())

        return PipelineDataType(
            dialogs=dialogues,
            supported_graph=supported_graph_validation,
            true_graph=true_graph_validation
            )

    def evaluate(self, *args, report_type="dict", **kwargs):
        return super().evaluate(*args, report_type=report_type, **kwargs)
