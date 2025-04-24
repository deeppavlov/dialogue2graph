"""
Graph
------

The module contains base class for graphs.
"""

import networkx as nx
from pydantic import BaseModel
from typing import Optional, Any
import matplotlib.pyplot as plt
import abc

from dialogue2graph.utils.logger import Logger

logger = Logger(__file__)


class BaseGraph(BaseModel, abc.ABC):
    """Base abstract class for graph representations of dialogues.

    This class provides the interface for graph operations and manipulations.
    It inherits from both BaseModel for data validation and ABC for abstract methods.

    Attributes:
        graph_dict (dict): Dictionary containing the graph structure with nodes and edges.
        graph (Optional[nx.Graph]): NetworkX graph instance.
        node_mapping (Optional[dict]): Mapping between original node IDs and internal representation.
    """

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
    def find_nodes_by_utterance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def find_edges_by_utterance(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_nodes_by_id(self):
        raise NotImplementedError

    @abc.abstractmethod
    def match_edges_nodes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def check_edges(self):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_duplicated_edges(self):
        raise NotImplementedError

    @abc.abstractmethod
    def remove_duplicated_nodes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def find_paths(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_all_paths(self):
        raise NotImplementedError

    def get_edges_by_source(self):
        raise NotImplementedError

    def get_edges_by_target(self):
        raise NotImplementedError

    def get_nodes_by_source(self):
        raise NotImplementedError

    def get_list_from_nodes(self):
        raise NotImplementedError

    def get_list_from_graph(self):
        raise NotImplementedError


class Graph(BaseGraph):
    """Implementation of BaseGraph for dialogue graph operations.

    This class provides concrete implementations for graph operations including
    loading, visualization, path finding, and graph manipulation methods.

    Attributes:
        Inherits all attributes from BaseGraph.
    """

    def __init__(self, graph_dict: dict, **kwargs: Any):
        """Initialize the Graph instance.

        Args:
            graph_dict (dict): Dictionary containing the graph structure.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        super().__init__(graph_dict=graph_dict, **kwargs)
        if graph_dict:
            self.load_graph()

    def _is_seq_in(self, a: list, b: list) -> bool:
        """Check if sequence a exists within sequence b."""
        return any(map(lambda x: b[x : x + len(a)] == a, range(len(b) - len(a) + 1)))

    def check_edges(self, seq: list[list[int]]) -> bool:
        """Checks whether seq (sequence of pairs (source, target))
        has all the edges of the graph
        """
        graph_dict = self.graph_dict
        edge_set = set((e["source"], e["target"]) for e in graph_dict["edges"])
        seen = set()
        for pair in seq:
            for s, t in zip(pair, pair[1:]):
                if (s, t) in edge_set:
                    seen.add((s, t))
        return seen == edge_set

    def load_graph(self):
        """Load graph from dictionary representation into NetworkX DiGraph.

        Creates a directed graph from the graph_dict, handling node and edge attributes.
        Also creates node mapping if node IDs need renumbering.
        """
        self.graph = nx.DiGraph()
        nodes = sorted([v["id"] for v in self.graph_dict["nodes"]])
        logger.debug(f"Nodes: {nodes}")

        self.node_mapping = {}
        renumber_flg = nodes != list(range(1, len(nodes) + 1))
        if renumber_flg:
            self.node_mapping = {node_id: idx + 1 for idx, node_id in enumerate(nodes)}
        logger.debug(f"Renumber flag: {renumber_flg}")

        for node in self.graph_dict["nodes"]:
            cur_node_id = node["id"]
            if renumber_flg:
                cur_node_id = self.node_mapping[cur_node_id]

            theme = node.get("theme")
            label = node.get("label")
            if type(node["utterances"]) is list:
                self.graph.add_node(
                    cur_node_id, theme=theme, label=label, utterances=node["utterances"]
                )
            else:
                self.graph.add_node(
                    cur_node_id,
                    theme=theme,
                    label=label,
                    utterances=[node["utterances"]],
                )

        for link in self.graph_dict["edges"]:
            source = self.node_mapping.get(link["source"], link["source"])
            target = self.node_mapping.get(link["target"], link["target"])
            self.graph.add_edges_from(
                [
                    (
                        source,
                        target,
                        {"theme": link.get("theme"), "utterances": link["utterances"]},
                    )
                ]
            )

    def visualise(self, *args, **kwargs):
        """Visualize the graph using matplotlib and networkx.

        Creates a visualization of the graph with nodes and edges labeled with utterances.
        Uses pygraphviz layout if available, falls back to kamada_kawai_layout.
        """
        plt.figure(figsize=(17, 11))  # Make the plot bigger
        try:
            pos = nx.nx_agraph.pygraphviz_layout(self.graph)
        except ImportError as e:
            pos = nx.kamada_kawai_layout(self.graph)
            logger.warning(
                f"{e}.\nInstall pygraphviz from http://pygraphviz.github.io/ .\nFalling back to default layout."
            )
        nx.draw(
            self.graph,
            pos,
            with_labels=False,
            node_color="lightblue",
            node_size=500,
            font_size=8,
            arrows=True,
        )
        edge_labels = nx.get_edge_attributes(self.graph, "utterances")
        node_labels = nx.get_node_attributes(self.graph, "utterances")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=12
        )
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        plt.title(__name__)
        plt.axis("off")
        plt.show()

    def visualise_short(self, name, *args, **kwargs):
        """Create a compact visualization of the graph.

        Args:
            name (str): Title for the visualization.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Creates a simplified visualization showing only node IDs and utterance counts.
        """
        try:
            pos = nx.nx_agraph.pygraphviz_layout(self.graph)
        except ImportError as e:
            pos = nx.kamada_kawai_layout(self.graph)
            logger.warning(
                f"{e}.\nInstall pygraphviz from http://pygraphviz.github.io/ .\nFalling back to default layout."
            )
        nx.draw(
            self.graph,
            pos,
            with_labels=False,
            node_color="lightblue",
            node_size=500,
            font_size=8,
            arrows=True,
        )
        edge_attrs = {
            (e["source"], e["target"]): len(e["utterances"])
            for e in self.graph_dict["edges"]
        }
        node_attrs = {
            n["id"]: f"{n['id']}:{len(n['utterances'])}"
            for n in self.graph_dict["nodes"]
        }
        nx.set_edge_attributes(self.graph, edge_attrs, "attrs")
        nx.set_node_attributes(self.graph, node_attrs, "attrs")
        edge_labels = nx.get_edge_attributes(self.graph, "attrs")
        node_labels = nx.get_node_attributes(self.graph, "attrs")
        nx.draw_networkx_edge_labels(
            self.graph, pos, edge_labels=edge_labels, font_size=12
        )
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        plt.title(name)
        plt.axis("off")
        plt.show()

    def find_nodes_by_utterance(self, utterance: str) -> list[dict]:
        """Find nodes containing a specific utterance.

        Args:
            utterance (str): The utterance to search for.

        Returns:
            list[dict]: List of nodes containing the utterance.
        """
        return [
            node for node in self.graph_dict["nodes"] if utterance in node["utterances"]
        ]

    def find_edges_by_utterance(self, utterance: str) -> list[dict]:
        """Find edges containing a specific utterance.

        Args:
            utterance (str): The utterance to search for.

        Returns:
            list[dict]: List of edges containing the utterance.
        """
        return [
            edge for edge in self.graph_dict["edges"] if utterance in edge["utterances"]
        ]

    def get_nodes_by_id(self, id: int):
        """Retrieve a node by its ID.

        Args:
            id (int): The ID of the node to retrieve.

        Returns:
            dict: The node with the specified ID if found, None otherwise.
        """
        for node in self.graph_dict["nodes"]:
            if node["id"] == id:
                return node

    def get_edges_by_source(self, id: int):
        """Get all edges originating from a specific node.

        Args:
            id (int): The ID of the source node.

        Returns:
            list[dict]: List of edges with the specified source node.
        """
        return [edge for edge in self.graph_dict["edges"] if edge["source"] == id]

    def get_edges_by_target(self, id: int):
        """Get all edges targeting a specific node.

        Args:
            id (int): The ID of the target node.

        Returns:
            list[dict]: List of edges with the specified target node.
        """
        return [edge for edge in self.graph_dict["edges"] if edge["target"] == id]

    def match_edges_nodes(self) -> bool:
        """Verify that all edge endpoints correspond to existing nodes.

        Returns:
            bool: True if all edge endpoints match existing nodes, False otherwise.
        """
        graph = self.graph_dict

        nodes_set = set(n["id"] for n in graph["nodes"])
        edges_set = set()
        for e in graph["edges"]:
            if not e["utterances"]:
                return False
            edges_set.add(e["source"])
            edges_set.add(e["target"])

        return nodes_set == edges_set

    def remove_duplicated_edges(self) -> BaseGraph:
        """Remove duplicate edges between the same node pairs.

        Combines utterances from duplicate edges into a single edge.

        Returns:
            BaseGraph: New graph instance with duplicate edges removed.
        """
        graph = self.graph_dict
        edges = graph["edges"]
        node_couples = [(e["source"], e["target"]) for e in edges]
        duplicates = [i for i in set(node_couples) if node_couples.count(i) > 1]
        new_edges = []
        for d in duplicates:
            found = [c for c in edges if c["source"] == d[0] and c["target"] == d[1]]
            new_edge = found[0].copy()
            new_edge["utterances"] = []
            for e in found:
                new_edge["utterances"].extend(e["utterances"])
            new_edge["utterances"] = list(set(new_edge["utterances"]))
            new_edges.append(new_edge)
        self.graph_dict = {
            "edges": [e for e in edges if (e["source"], e["target"]) not in duplicates]
            + new_edges,
            "nodes": graph["nodes"],
        }
        return Graph(self.graph_dict)

    def remove_duplicated_nodes(self) -> BaseGraph | None:
        """Remove duplicate nodes based on their utterances.

        Returns:
            BaseGraph | None: New graph instance with duplicate nodes removed,
                            or None if invalid state is detected.
        """
        graph = self.graph_dict
        nodes = graph["nodes"].copy()
        edges = graph["edges"].copy()
        nodes_utterances = [n["utterances"] for n in nodes]
        map(lambda x: x.sort(), nodes_utterances)
        seen = []
        to_remove = []
        for node in nodes:
            utts = node["utterances"]
            utts.sort()
            if utts not in seen:
                seen_utts = list(set([s for xs in seen for s in xs]))
                if any([utt in seen_utts for utt in utts]):
                    return None
                seen.append(utts)
            else:
                doubled = nodes[nodes_utterances.index(utts)]["id"]
                to_remove.append(node["id"])
                for idx, edge in enumerate(edges):
                    if edge["source"] == node["id"]:
                        edges[idx]["source"] = doubled
                    if edge["target"] == node["id"]:
                        edges[idx]["target"] = doubled
        self.graph_dict = {
            "edges": edges,
            "nodes": [n for n in nodes if n["id"] not in to_remove],
        }
        return self.remove_duplicated_edges()

    def get_all_paths(
        self, start_node_id: int, visited_nodes: list[int], repeats_limit: int
    ) -> list[list[int]]:
        """Find all possible paths in the graph from a starting node.

        Args:
            start_node_id (int): ID of the starting node.
            visited_nodes (list[int]): List of nodes already visited in the current path.
            repeats_limit (int): Maximum number of times a sequence can repeat.

        Returns:
            list[list[int]]: List of all valid paths found.
        """
        if len(visited_nodes) >= repeats_limit and self._is_seq_in(
            visited_nodes[-repeats_limit:] + [start_node_id], visited_nodes
        ):
            return []

        visited_nodes.append(start_node_id)
        visited_paths = [visited_nodes.copy()]

        for edge in self.get_edges_by_source(start_node_id):
            visited_paths.extend(
                self.get_all_paths(edge["target"], visited_nodes, repeats_limit)
            )

        visited_nodes.pop()
        return visited_paths

    def find_paths(
        self, start_node_id: int, end_node_id: int, visited_nodes: list[int]
    ) -> list[list[int]]:
        """Find all paths between two nodes in the graph.

        Args:
            start_node_id (int): ID of the starting node.
            end_node_id (int): ID of the target node.
            visited_nodes (list[int]): List of nodes already visited.

        Returns:
            list[list[int]]: List of all paths found between start and end nodes.
        """
        visited_paths = [[]]

        graph = self.graph_dict
        if (
            len(visited_nodes) <= len(graph["edges"])
            and end_node_id not in visited_paths[-1]
        ):
            visited_nodes.append(start_node_id)
            if end_node_id not in visited_nodes:
                for edge in self.get_edges_by_source(start_node_id):
                    visited_paths += self.find_paths(
                        edge["target"], end_node_id, visited_nodes
                    )
        else:
            visited_nodes.append(start_node_id)
        visited_paths.append(visited_nodes)
        return visited_paths

    def get_ends(self) -> list[int]:
        """Find all terminal nodes in the graph.

        Terminal nodes are those with no outgoing edges.

        Returns:
            list[int]: List of IDs of terminal nodes.
        """
        graph = self.graph_dict
        sources = list(set([g["source"] for g in graph["edges"]]))
        finishes = [g["id"] for g in graph["nodes"] if g["id"] not in sources]
        if not finishes:
            finishes = [[g["id"] for g in graph["nodes"] if not g["is_start"]][-1]]
        visited = set(finishes.copy())
        for f in finishes:
            for n in graph["nodes"]:
                if n["id"] != f:
                    visited_paths = self.find_paths(n["id"], f, [])
                if any([f in v for v in visited_paths]):
                    visited.add(n["id"])
        if len(visited) < len(graph["nodes"]):
            finishes += [v["id"] for v in graph["nodes"] if v["id"] not in visited]
        return finishes

    def get_list_from_nodes(self) -> list[str]:
        """Create a list of concatenated utterances from all nodes.

        Returns:
            list[str]: List where each element is the concatenated utterances of a node.
        """
        graph = self.graph_dict
        result = []

        for node in graph["nodes"]:
            utts = ""
            for node_utt in node["utterances"]:
                utts += node_utt + " "
            result.append(utts)

        return result

    def get_list_from_graph(self) -> tuple[list[str], int]:
        """Create a list of concatenated utterances from nodes and their edges.

        Returns:
            tuple[list[str], int]: Tuple containing:
                - list of concatenated utterances
                - total number of utterances in edges
        """
        graph = self.graph_dict
        res_list = []
        n_edges = 0

        for node in graph["nodes"]:
            utts = " ".join(node["utterances"])
            for edge in self.get_edges_by_source(node["id"]):
                utts += " ".join(edge["utterances"])
                n_edges += len(edge["utterances"])
            res_list.append(utts)
        return res_list, n_edges
