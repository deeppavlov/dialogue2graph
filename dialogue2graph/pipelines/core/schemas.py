"""
Schemas
-------

The module contains base classes for graph abstractions.
"""
from typing import List
from pydantic import BaseModel, Field
from dialogue2graph.pipelines.core.dialogue import Dialogue


class Edge(BaseModel):
    # TODO: add docs

    source: int = Field(description="ID of the source node")
    target: int = Field(description="ID of the target node")
    utterances: List[str] = Field(
        description="User's utterances that trigger this transition"
    )


class Node(BaseModel):
    # TODO: add docs

    id: int = Field(description="Unique identifier for the node")
    label: str = Field(description="Label describing the node's purpose")
    is_start: bool = Field(description="Whether this is the starting node")
    utterances: List[str] = Field(
        description="Possible assistant responses at this node"
    )


class DialogueGraph(BaseModel):
    # TODO: add docs
    
    edges: List[Edge] = Field(description="List of transitions between nodes")
    nodes: List[Node] = Field(description="List of nodes representing assistant states")


class ReasonGraph(BaseModel):
    edges: List[Edge] = Field(description="List of transitions between nodes")
    nodes: List[Node] = Field(description="List of nodes representing assistant states")
    reason: str = Field(description="Description of LLM answer")


class GraphGenerationResult(BaseModel):
    """Complete result with graph and dialogues"""

    graph: DialogueGraph
    topic: str
    dialogues: List[Dialogue]


class CompareResponse(BaseModel):
    value: bool = Field(default=True, description="compare result")
    description: str = Field(description="explanation")
