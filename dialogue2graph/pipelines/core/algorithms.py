"""
Algorithms
-----------

The module contains base classes for different algorithms.
"""

import abc
from typing import List, Literal, Union
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from pandas import DataFrame
from dialogue2graph.pipelines.core.graph import BaseGraph
from dialogue2graph.pipelines.core.dialogue import Dialogue


class BaseAlgorithm(BaseModel, abc.ABC):
    """
    Base class for all algorithms that interact with Dialogues or Graphs.

    This class defines the interface for invoking algorithms, both synchronously and asynchronously.
    """

    @abc.abstractmethod
    def invoke(self, *args, use_cache=False, seed=42, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    async def ainvoke(self, *args, use_cache=False, seed=42, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, *args, report_type: Literal["dict", "dataframe"] = "dict", **kwargs
    ) -> Union[dict, DataFrame]:
        raise NotImplementedError


class DialogueGenerator(BaseAlgorithm):
    """
    Base class for generating Dialogues from a Graph object.

    This class is intended for sampling Dialogues based on a given Graph structure.

    Args:
        graph: The Graph object used for generating the Dialogue.
        start_node: The starting node in the Graph for the generation process (default=1).
        end_node: The ending node in the Graph for the generation process (optional).
        topic: The topic to guide the generation process (optional).
    """

    def __init__(self):
        super().__init__()

    def invoke(
        self, graph: BaseGraph, start_node: int = 1, end_node: int = 0, topic: str = ""
    ) -> List[Dialogue]:
        raise NotImplementedError


class DialogAugmentation(BaseAlgorithm):
    """
    Base class for augmenting Dialogues.

    This class takes a Dialogue as input and returns an augmented Dialogue as output.
    It is designed for data augmentation or other manipulations of Dialogues.

    Args:
        dialogue: The Dialogue object to be augmented.
        topic: The topic to guide the augmentation process (optional).
    """

    def __init__(self) -> None:
        super().__init__()

    def invoke(self, dialogue: Dialogue, topic: str = "") -> Dialogue:
        raise NotImplementedError


class GraphAugmentation(BaseAlgorithm):
    """Base class for augmenting Graphs

    Args:
        topic: The topic to guide the augmentation process (optional).
        graph: The Graph object to be augmented."""

    def invoke(self, topic: str, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, topic: str, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError


class TopicGraphGenerator(BaseAlgorithm):
    """Base class for topic-based graph generation"""

    def invoke(self, topic: str, model: BaseChatModel) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, topic: str) -> BaseGraph:
        raise NotImplementedError


class GraphGenerator(BaseAlgorithm):
    """Base class for graph generation"""

    def invoke(self, dialogue: Dialogue) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, dialogue: Dialogue) -> BaseGraph:
        raise NotImplementedError


class GraphExtender(BaseAlgorithm):
    """Base class for extending graph"""

    def invoke(self, dialogue: Dialogue, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, dialogue: Dialogue, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError


class RawDataParser(BaseAlgorithm):
    """Base class for user data parsing"""

    def invoke(self, data):
        raise NotImplementedError

    async def ainvoke(self, data):
        raise NotImplementedError
