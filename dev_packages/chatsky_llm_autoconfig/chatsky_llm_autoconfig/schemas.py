from typing import List
from pydantic import BaseModel, Field


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
    reason: str = Field(description="explanation")

class DialogueNodes(BaseModel):
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="explanation")

class CompareResponse(BaseModel):
    result: bool = Field(description="compare result")
    reason: str = Field(description="explanation")

class DialogueMessage(BaseModel):
    """Represents a single message in a dialogue.

    Attributes:
        text: The content of the message
        participant: The sender of the message (e.g. "user" or "assistant")
    """

    text: str
    participant: str
