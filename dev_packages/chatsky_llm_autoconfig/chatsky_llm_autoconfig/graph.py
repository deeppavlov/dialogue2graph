import networkx as nx
from pydantic import BaseModel
from typing import Optional, Any
import matplotlib.pyplot as plt
import abc
import logging

logger = logging.getLogger(__name__)


class BaseGraph(BaseModel, abc.ABC):
    graph_dict: dict
    graph: Optional[nx.Graph] = None
    node_mapping: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    @abc.abstractmethod
    def load_graph(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def visualise(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def nodes_by_utterance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def edges_by_utterance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def node_by_id(self):
        raise NotImplementedError


class Graph(BaseGraph):

    def __init__(self, graph_dict: dict, **kwargs: Any):
        # Pass graph_dict to the parent class
        super().__init__(graph_dict=graph_dict, **kwargs)
        if graph_dict:
            self.load_graph()

    def load_graph(self):
        self.graph = nx.DiGraph()
        nodes = sorted([v["id"] for v in self.graph_dict["nodes"]])
        logging.debug(f"Nodes: {nodes}")

        self.node_mapping = {}
        renumber_flg = nodes != list(range(1, len(nodes) + 1))
        if renumber_flg:
            self.node_mapping = {node_id: idx + 1 for idx, node_id in enumerate(nodes)}
        logging.debug(f"Renumber flag: {renumber_flg}")

        for node in self.graph_dict["nodes"]:
            cur_node_id = node["id"]
            if renumber_flg:
                cur_node_id = self.node_mapping[cur_node_id]

            theme = node.get("theme")
            label = node.get("label")
            if type(node["utterances"]) is list:
                self.graph.add_node(cur_node_id, theme=theme, label=label, utterances=node["utterances"])
            else:
                self.graph.add_node(cur_node_id, theme=theme, label=label, utterances=[node["utterances"]])

        for link in self.graph_dict["edges"]:
            source = self.node_mapping.get(link["source"], link["source"])
            target = self.node_mapping.get(link["target"], link["target"])
            self.graph.add_edges_from([(source, target, {"theme": link.get("theme"), "utterances": link["utterances"]})])

    def visualise(self, *args, **kwargs):
        plt.figure(figsize=(17, 11))  # Make the plot bigger
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False, node_color="lightblue", node_size=500, font_size=8, arrows=True)
        edge_labels = nx.get_edge_attributes(self.graph, "utterances")
        node_labels = nx.get_node_attributes(self.graph, "utterances")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=12)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        plt.title(__name__)
        plt.axis("off")
        plt.show()

    def nodes_by_utterance(self, utterance: str) -> list[dict]:
        return [node for node in self.graph_dict['nodes'] if utterance in node['utterances']]
            
    def edges_by_utterance(self, utterance: str) -> list[dict]:
        return [edge for edge in self.graph_dict['edges'] if utterance in edge['utterances']]
            
    def node_by_id(self, id: int):
        for node in self.graph_dict['nodes']:
            if node['id'] == id:
                return node
    
    def edge_by_source(self, source: int):
        return [edge for edge in self.graph_dict['edges'] if source == edge['source']]