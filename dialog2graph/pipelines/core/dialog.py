"""
Dialog
--------

The module provides Dialog class that represents complete dialogs with multiple messages.
"""

import uuid
import networkx as nx
from typing import List, Union, Dict
from pydantic import BaseModel, Field, ConfigDict


class DialogMessage(BaseModel):
    """Represents a single message in a dialog.

    Attributes:
        text: The content of the message
        participant: The sender of the message (e.g. "user" or "assistant")
    """

    text: str
    participant: str


class Dialog(BaseModel):
    """Represents a complete dialog consisting of multiple messages.

    The class provides methods for creating dialogs from different formats
    and converting dialogs to various representations.
    """

    messages: List[DialogMessage] = Field(default_factory=list)
    id: str = Field(
        default=str(uuid.uuid1()), description="Unique identifier for the dialog"
    )
    topic: str = ""
    validate: bool = Field(
        default=True, description="Whether to validate messages upon initialization"
    )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=False,  # Dialog needs to be mutable to append messages
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.validate:
            self.__validate(self.messages)

    @classmethod
    def from_string(cls, string: str) -> "Dialog":
        """Creates a Dialog from a tab-separated string format.

        Args:
            string: Tab-separated string with format: "participant\\ttext\\n"

        Returns:
            Dialog object with parsed messages
        """
        messages: List[DialogMessage] = [
            DialogMessage(participant=line.split("\t")[0], text=line.split("\t")[1])
            for line in string.strip().split("\n")
        ]
        return cls(messages=messages)

    @classmethod
    def from_list(
        cls, messages: List[Dict[str, str]], id: str = "", validate: bool = True
    ) -> "Dialog":
        """Create a Dialog from a list of dictionaries."""
        dialog_messages = [DialogMessage(**m) for m in messages]
        return cls(messages=dialog_messages, id=id, validate=validate)

    @classmethod
    def from_nodes_ids(cls, graph, node_list, validate: bool = True) -> "Dialog":
        # TODO: add docs
        utts = []
        nodes_attributes = nx.get_node_attributes(graph.graph, "utterances")
        edges_attributes = nx.get_edge_attributes(graph.graph, "utterances")
        for node in range(len(node_list)):
            utts.append(
                {
                    "participant": "assistant",
                    "text": nodes_attributes[node_list[node]][0],
                }
            )
            if node == len(node_list) - 1:
                if graph.graph.has_edge(node_list[node], node_list[0]):
                    utts.append(
                        {
                            "participant": "user",
                            "text": edges_attributes[(node_list[node], node_list[0])][
                                0
                            ],
                        }
                    )
            else:
                if graph.graph.has_edge(node_list[node], node_list[node + 1]):
                    utts.append(
                        {
                            "participant": "user",
                            "text": edges_attributes[
                                (node_list[node], node_list[node + 1])
                            ][0],
                        }
                    )

        return cls(messages=utts, validate=validate)

    def to_list(self) -> List[Dict[str, str]]:
        """Converts Dialog to a list of message dictionaries."""
        return [msg.model_dump() for msg in self.messages]

    def __str__(self) -> str:
        """Returns a readable string representation of the dialog."""
        return "\n".join(
            f"{msg.participant}: {msg.text}" for msg in self.messages
        ).strip()

    def append(self, text: str, participant: str) -> None:
        """Adds a new message to the dialog.

        Args:
            text: Content of the message
            participant: Sender of the message
        """
        self.messages.append(DialogMessage(text=text, participant=participant))

    def extend(self, messages: List[Union[DialogMessage, Dict[str, str]]]) -> None:
        """Adds multiple messages to the dialog.

        Args:
            messages: List of DialogMessage objects or dicts to add
        """
        new_messages = [
            msg if isinstance(msg, DialogMessage) else DialogMessage(**msg)
            for msg in messages
        ]
        self.__validate(new_messages)
        self.messages.extend(new_messages)

    def __validate(self, messages):
        """Ensure that messages meets expectations."""
        if not messages:
            return

        # Check if first message is from assistant
        if messages[0].participant != "assistant":
            raise ValueError(
                f"First message must be from assistant, got: {messages[0]}"
            )

        # Check for consecutive messages from same participant
        for i in range(len(messages) - 1):
            if messages[i].participant == messages[i + 1].participant:
                raise ValueError(
                    f"Cannot have consecutive messages from the same participant. Messages: {messages[i]}, {messages[i + 1]}"
                )
