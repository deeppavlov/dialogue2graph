from typing import List
from pydantic import BaseModel
import abc
from chatsky_llm_autoconfig.graph import BaseGraph
from chatsky_llm_autoconfig.dialogue import Dialogue
from langchain_core.language_models.chat_models import BaseChatModel


class BaseAlgorithm(BaseModel, abc.ABC):
    """
    Base class for all algorithms that interact with Dialogues or Graphs.

    This class defines the interface for invoking algorithms, both synchronously and asynchronously.
    """

    @abc.abstractmethod
    def invoke(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    async def ainvoke(self, *args, **kwargs):
        raise NotImplementedError


class DialogueGenerator(BaseAlgorithm):
    """
    Base class for generating Dialogues from a Graph object.

    This class is intended for sampling Dialogues based on a given Graph structure.

    :param graph: The Graph object used for generating the Dialogue.
    :param start_node: The starting node in the Graph for the generation process (default=1).
    :param end_node: The ending node in the Graph for the generation process (optional).
    :param topic: The topic to guide the generation process (optional).
    """

    def __init__(self):
        super().__init__()

    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = 0, topic: str = "") -> List[Dialogue]:
        raise NotImplementedError


class DialogAugmentation(BaseAlgorithm):
    """
    Base class for augmenting Dialogues.

    This class takes a Dialogue as input and returns an augmented Dialogue as output.
    It is designed for data augmentation or other manipulations of Dialogues.

    :param dialogue: The Dialogue object to be augmented.
    :param topic: The topic to guide the augmentation process (optional).
    """

    def __init__(self):
        super().__init__()

    def invoke(self, dialogue: Dialogue, topic: str = "") -> Dialogue:
        raise NotImplementedError


class GraphAugmentation(BaseAlgorithm):
    """Graph generator that works only with topics."""

    def invoke(self, topic: str, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, topic: str, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError


class TopicGraphGenerator(BaseAlgorithm):
    """Graph generator that works only with topics."""

    def invoke(self, topic: str, model: BaseChatModel) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, topic: str) -> BaseGraph:
        raise NotImplementedError


class GraphGenerator(BaseAlgorithm):
    """Graph generator that works only with topics."""

    def invoke(self, dialogue: Dialogue) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, dialogue: Dialogue) -> BaseGraph:
        raise NotImplementedError


class GraphExtender(BaseAlgorithm):
    """Graph generator that works only with topics."""

    def invoke(self, dialogue: Dialogue, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError

    async def ainvoke(self, dialogue: Dialogue, graph: BaseGraph) -> BaseGraph:
        raise NotImplementedError
