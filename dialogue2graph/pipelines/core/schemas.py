import networkx as nx
from typing import List, Union, Dict
from pydantic import BaseModel, Field, ConfigDict

from dialogue2graph.pipelines.core.dialogue import Dialogue


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
