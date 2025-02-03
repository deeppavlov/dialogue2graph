import networkx as nx
from typing import List, Union, Dict
from pydantic import BaseModel, Field, ConfigDict

# from chatsky_llm_autoconfig.dialogue import Dialogue


class DialogueMessage(BaseModel):
    """Represents a single message in a dialogue.

    Attributes:
        text: The content of the message
        participant: The sender of the message (e.g. "user" or "assistant")
    """

    text: str
    participant: str


# class Dialogue(BaseModel):
#     """Represents a complete dialogue consisting of multiple messages.

#     The class provides methods for creating dialogues from different formats
#     and converting dialogues to various representations.
#     """

#     messages: List[DialogueMessage] = Field(default_factory=list)
#     topic: str = ""
#     validate: bool = Field(default=True, description="Whether to validate messages upon initialization")

#     model_config = ConfigDict(
#         arbitrary_types_allowed=True,
#         frozen=False,  # Dialogue needs to be mutable to append messages
#     )

#     def __init__(self, **data):
#         super().__init__(**data)
#         if self.validate:
#             self.__validate(self.messages)

#     @classmethod
#     def from_string(cls, string: str) -> "Dialogue":
#         """Creates a Dialogue from a tab-separated string format.

#         Args:
#             string: Tab-separated string with format: "participant\ttext\n"

#         Returns:
#             Dialogue object with parsed messages
#         """
#         messages: List[DialogueMessage] = [
#             DialogueMessage(participant=line.split("\t")[0], text=line.split("\t")[1]) for line in string.strip().split("\n")
#         ]
#         return cls(messages=messages)

#     @classmethod
#     def from_list(cls, messages: List[Dict[str, str]], validate: bool = True) -> "Dialogue":
#         """Create a Dialogue from a list of dictionaries."""
#         dialogue_messages = [DialogueMessage(**m) for m in messages]
#         return cls(messages=dialogue_messages, validate=validate)

#     @classmethod
#     def from_nodes_ids(cls, graph, node_list, validate: bool = True) -> "Dialogue":
#         utts = []
#         nodes_attributes = nx.get_node_attributes(graph.graph, "utterances")
#         edges_attributes = nx.get_edge_attributes(graph.graph, "utterances")
#         for node in range(len(node_list)):
#             utts.append({"participant": "assistant", "text": nodes_attributes[node_list[node]][0]})
#             if node == len(node_list) - 1:
#                 if graph.graph.has_edge(node_list[node], node_list[0]):
#                     utts.append({"participant": "user", "text": edges_attributes[(node_list[node], node_list[0])][0]})
#             else:
#                 if graph.graph.has_edge(node_list[node], node_list[node + 1]):
#                     utts.append({"participant": "user", "text": edges_attributes[(node_list[node], node_list[node + 1])][0]})

#         return cls(messages=utts, validate=validate)

#     def to_list(self) -> List[Dict[str, str]]:
#         """Converts Dialogue to a list of message dictionaries."""
#         return [msg.model_dump() for msg in self.messages]

#     def __str__(self) -> str:
#         """Returns a readable string representation of the dialogue."""
#         return "\n".join(f"{msg.participant}: {msg.text}" for msg in self.messages).strip()

#     def append(self, text: str, participant: str) -> None:
#         """Adds a new message to the dialogue.

#         Args:
#             text: Content of the message
#             participant: Sender of the message
#         """
#         self.messages.append(DialogueMessage(text=text, participant=participant))

#     def extend(self, messages: List[Union[DialogueMessage, Dict[str, str]]]) -> None:
#         """Adds multiple messages to the dialogue.

#         Args:
#             messages: List of DialogueMessage objects or dicts to add
#         """
#         new_messages = [msg if isinstance(msg, DialogueMessage) else DialogueMessage(**msg) for msg in messages]
#         self.__validate(new_messages)
#         self.messages.extend(new_messages)

#     def __validate(self, messages):
#         """Ensure that messages meets expectations."""
#         if not messages:
#             return

#         # Check if first message is from assistant
#         if messages[0].participant != "assistant":
#             raise ValueError(f"First message must be from assistant, got: {messages[0]}")

#         # Check for consecutive messages from same participant
#         for i in range(len(messages) - 1):
#             if messages[i].participant == messages[i + 1].participant:
#                 raise ValueError(f"Cannot have consecutive messages from the same participant. Messages: {messages[i]}, {messages[i + 1]}")


class Dialogue(BaseModel):
    """Represents a complete dialogue consisting of multiple messages.

    The class provides methods for creating dialogues from different formats
    and converting dialogues to various representations.
    """

    messages: List[DialogueMessage] = Field(default_factory=list)
    topic: str = ""
    validate: bool = Field(default=True, description="Whether to validate messages upon initialization")

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,  # Dialogue needs to be mutable to append messages
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.validate:
            self.__validate(self.messages)

    @classmethod
    def from_string(cls, string: str) -> "Dialogue":
        """Creates a Dialogue from a tab-separated string format.

        Args:
            string: Tab-separated string with format: "participant\ttext\n"

        Returns:
            Dialogue object with parsed messages
        """
        messages: List[DialogueMessage] = [
            DialogueMessage(participant=line.split("\t")[0], text=line.split("\t")[1]) for line in string.strip().split("\n")
        ]
        return cls(messages=messages)

    @classmethod
    def from_list(cls, messages: List[Dict[str, str]], validate: bool = True) -> "Dialogue":
        """Create a Dialogue from a list of dictionaries."""
        dialogue_messages = [DialogueMessage(**m) for m in messages]
        return cls(messages=dialogue_messages, validate=validate)

    @classmethod
    def from_nodes_ids(cls, graph, node_list, validate: bool = True) -> "Dialogue":
        utts = []
        nodes_attributes = nx.get_node_attributes(graph.graph, "utterances")
        edges_attributes = nx.get_edge_attributes(graph.graph, "utterances")
        for node in range(len(node_list)):
            utts.append({"participant": "assistant", "text": nodes_attributes[node_list[node]][0]})
            if node == len(node_list) - 1:
                if graph.graph.has_edge(node_list[node], node_list[0]):
                    utts.append({"participant": "user", "text": edges_attributes[(node_list[node], node_list[0])][0]})
            else:
                if graph.graph.has_edge(node_list[node], node_list[node + 1]):
                    utts.append({"participant": "user", "text": edges_attributes[(node_list[node], node_list[node + 1])][0]})

        return cls(messages=utts, validate=validate)

    def to_list(self) -> List[Dict[str, str]]:
        """Converts Dialogue to a list of message dictionaries."""
        return [msg.model_dump() for msg in self.messages]

    def __str__(self) -> str:
        """Returns a readable string representation of the dialogue."""
        return "\n".join(f"{msg.participant}: {msg.text}" for msg in self.messages).strip()

    def append(self, text: str, participant: str) -> None:
        """Adds a new message to the dialogue.

        Args:
            text: Content of the message
            participant: Sender of the message
        """
        self.messages.append(DialogueMessage(text=text, participant=participant))

    def extend(self, messages: List[Union[DialogueMessage, Dict[str, str]]]) -> None:
        """Adds multiple messages to the dialogue.

        Args:
            messages: List of DialogueMessage objects or dicts to add
        """
        new_messages = [msg if isinstance(msg, DialogueMessage) else DialogueMessage(**msg) for msg in messages]
        self.__validate(new_messages)
        self.messages.extend(new_messages)

    def __validate(self, messages):
        """Ensure that messages meets expectations."""
        if not messages:
            return

        # Check if first message is from assistant
        if messages[0].participant != "assistant":
            raise ValueError(f"First message must be from assistant, got: {messages[0]}")

        # Check for consecutive messages from same participant
        for i in range(len(messages) - 1):
            if messages[i].participant == messages[i + 1].participant:
                raise ValueError(f"Cannot have consecutive messages from the same participant. Messages: {messages[i]}, {messages[i + 1]}")


class Edge(BaseModel):
    source: int = Field(description="ID of the source node")
    target: int = Field(description="ID of the target node")
    utterances: List[str] = Field(description="User's utterances that trigger this transition")


class Node(BaseModel):
    id: int = Field(description="Unique identifier for the node")
    label: str = Field(description="Label describing the node's purpose")
    is_start: bool = Field(description="Whether this is the starting node")
    utterances: List[str] = Field(description="Possible assistant responses at this node")


class DialogueGraph(BaseModel):
    edges: List[Edge] = Field(description="List of transitions between nodes")
    nodes: List[Node] = Field(description="List of nodes representing assistant states")


class GraphGenerationResult(BaseModel):
    """Complete result with graph and dialogues"""

    graph: DialogueGraph
    topic: str
    dialogues: List[Dialogue]
