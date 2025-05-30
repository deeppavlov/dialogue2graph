"""
Helper ParseData
------------------

The module contains parser to extract user data containing dialogs and graphs
"""

# from dialog2graph.pipelines.core.pipeline import Pipeline as BasePipeline
import json
from typing import List, Optional
from pydantic import BaseModel
from pathlib import PosixPath
from pydantic import TypeAdapter
from dialog2graph.pipelines.core.dialog import Dialog, DialogMessage
from dialog2graph.pipelines.core.algorithms import RawDataParser
from dialog2graph.pipelines.core import schemas
from dialog2graph.pipelines.core import graph
from dialog2graph.utils.logger import Logger

logger = Logger(__name__)

RawDialogsType = dict | list[list] | list[dict] | Dialog | list[Dialog] | PosixPath
ValidatedDialogType = (
    List[DialogMessage] | List[List[DialogMessage]] | Dialog | List[Dialog]
)
RawGraphType = schemas.DialogGraph | dict | PosixPath
ValidatedGraphType = schemas.DialogGraph | None


class PipelineRawDataType(BaseModel):
    dialogs: RawDialogsType
    supported_graph: Optional[RawGraphType] | None = None
    true_graph: Optional[RawGraphType] | None = None


class PipelineDataType(BaseModel):
    dialogs: list[Dialog]
    supported_graph: Optional[graph.Graph] | None = None
    true_graph: Optional[graph.Graph] | None = None


class RawDGParser(RawDataParser):
    """Parser of raw user data with dialogs and graphs"""

    def _validate_raw_graph(
        self, raw_graph: RawGraphType
    ) -> ValidatedGraphType | PosixPath:
        """Validate raw graph data

        Args:
            raw_graph: graph in a form of either schemas.DialogGraph or file path

        Returns:
            schemas.DialogGraph or PosixPath when raw_graph is file_path
            None when validation error
        """
        if raw_graph is None:
            return None
        if isinstance(raw_graph, schemas.DialogGraph):
            return raw_graph
        if isinstance(raw_graph, PosixPath):
            return raw_graph
        return schemas.DialogGraph.model_validate(raw_graph)

    def _validate_raw_dialogs(
        self, raw_dialogs: RawDialogsType
    ) -> ValidatedDialogType | PosixPath:
        """Validate raw dialogs data

        Args:
            raw_dialogs: dialogs in a form of RawDialogsType

        Returns:
            ValidatedDialogType or PosixPath when raw_dialogs is file_path
            Empty list when validation error
        """
        if raw_dialogs is None:
            raise ValueError("Raw dialogs data is None")
        return TypeAdapter(ValidatedDialogType | PosixPath).validate_python(raw_dialogs)

    def _get_dialogs_from_file(self, file_path: PosixPath) -> ValidatedDialogType:
        """Extract dialogs from file_path

        Args:
            file_path: file to work with

        Returns:
            validated dialogs or empty list if any error
        """
        if file_path.suffix == ".json":
            with open(file_path) as f:
                raw_dialogs = json.load(f)
            if isinstance(raw_dialogs, dict) and "dialogs" in raw_dialogs:
                raw_dialogs = raw_dialogs["dialogs"]
            elif isinstance(raw_dialogs, list):
                pass
            else:
                raise ValueError(
                    "Data is not list or dict with 'dialogs' key in json file: %s"
                )
            return self._validate_raw_dialogs(raw_dialogs)
        else:
            raise ValueError("File extension must be json: %s", file_path)

    def _get_graph_from_file(
        self, file_path: PosixPath, key: str
    ) -> ValidatedGraphType:
        """Extract graph from file_path

        Args:
            file_path: file to work with
            key: key to search graph data in a dict from file

        Returns:
            validated graph or None if validation unsuccessful
        """
        if file_path.suffix != ".json":
            raise ValueError("File extension must be json: %s", file_path)

        with open(file_path) as f:
            raw_graph = json.load(f)

        if not isinstance(raw_graph, dict) or key not in raw_graph:
            raise ValueError(
                "Invalid data structure or missing key '%s' in file: %s", key, file_path
            )
        raw_graph_data = raw_graph[key]
        if isinstance(raw_graph_data, list) and raw_graph_data:
            raw_graph_data = raw_graph_data[0]

        return self._validate_raw_graph(raw_graph_data)

    def invoke(self, raw_data: PipelineRawDataType) -> PipelineDataType:
        """Validate and convert user's data into list of Dialogs

        Args:
            raw_data: data to parse
                raw_data.dialogs can be as follows:
                    [{'participant': user or assistant, 'text': text}]
                    {'messages': [{'participant': user or assistant, 'text': text}]}
                    [[{'participant': user or assistant, 'text': text}]]
                    [{'messages': [{'participant': user or assistant, 'text': text}]}]
                    or same in json file presented by file path
                raw_data.supported_graph and raw_data.true_graph:
                    either schemas.DialogGraph or file path

        Returns: PipelineDataType with dialogs and graphs
        """

        def process_dialogs(dialogs):
            validation = self._validate_raw_dialogs(dialogs)
            if isinstance(validation, PosixPath):
                validation = self._get_dialogs_from_file(validation)

            if isinstance(validation, Dialog):
                return [validation]

            if isinstance(validation, Dialog):
                return [validation]
            elif isinstance(validation, list) and validation:
                if isinstance(validation[0], Dialog):
                    return validation
                elif isinstance(validation[0], DialogMessage):
                    return [Dialog(messages=validation)]
                elif isinstance(validation[0], list) and isinstance(
                    validation[0][0], DialogMessage
                ):
                    return [Dialog(messages=dialog) for dialog in validation]
            return []

        def process_graph(graph_data, key):
            validation = self._validate_raw_graph(graph_data)
            if isinstance(validation, PosixPath):
                validation = self._get_graph_from_file(validation, key)
            return graph.Graph(validation.model_dump()) if validation else None

        dialogs = process_dialogs(raw_data.dialogs)
        supported_graph = process_graph(raw_data.supported_graph, "graph")
        true_graph = process_graph(raw_data.true_graph, "true_graph")

        return PipelineDataType(
            dialogs=dialogs, supported_graph=supported_graph, true_graph=true_graph
        )

    def evaluate(self, *args, report_type="dict", **kwargs):
        return super().evaluate(*args, report_type=report_type, **kwargs)
