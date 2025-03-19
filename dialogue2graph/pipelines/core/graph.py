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

    @abc.abstractmethod
    def edges_match_nodes(self):
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
    def find_path(self):
        raise NotImplementedError

    @abc.abstractmethod
    def all_paths(self):
        raise NotImplementedError

    @abc.abstractmethod
    def nodes2list(self):
        raise NotImplementedError

    @abc.abstractmethod
    def graph2list(self):
        raise NotImplementedError


class Graph(BaseGraph):

    def __init__(self, graph_dict: dict, **kwargs: Any):
        # Pass graph_dict to the parent class
        super().__init__(graph_dict=graph_dict, **kwargs)
        if graph_dict:
            self.load_graph()

    def _list_in(self, a: list, b: list) -> bool:
        """Check if sequence a exists within sequence b."""
        return any(map(lambda x: b[x : x + len(a)] == a, range(len(b) - len(a) + 1)))

    def check_edges(self, seq: list[list[int]]) -> bool:
        """Checks whether seq (sequence of pairs (source, target)) has all the edges of the graph"""
        graph_dict = self.graph_dict
        edge_seq = {(e["source"], e["target"]) for e in graph_dict["edges"]}
        left = edge_seq.copy()
        for pair in edge_seq:
            for s in seq:
                if self._list_in(list(pair), s):
                    left -= set([pair])
                    if len(left) == 0:
                        return True
        if len(left):

            return False
        return True

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
        pos = nx.nx_agraph.pygraphviz_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False, node_color="lightblue", node_size=500, font_size=8, arrows=True)
        edge_labels = nx.get_edge_attributes(self.graph, "utterances")
        node_labels = nx.get_node_attributes(self.graph, "utterances")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=12)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        plt.title(__name__)
        plt.axis("off")
        plt.show()

    def visualise_short(self, name, *args, **kwargs):
        # pos = nx.nx_agraph.pygraphviz_layout(self.graph)
        pos = nx.kamada_kawai_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=False, node_color="lightblue", node_size=500, font_size=8, arrows=True)
        edge_attrs = {(e["source"], e["target"]): len(e["utterances"]) for e in self.graph_dict["edges"]}
        node_attrs = {n["id"]: f"{n['id']}:{len(n['utterances'])}" for n in self.graph_dict["nodes"]}
        nx.set_edge_attributes(self.graph, edge_attrs, "attrs")
        nx.set_node_attributes(self.graph, node_attrs, "attrs")
        edge_labels = nx.get_edge_attributes(self.graph, "attrs")
        node_labels = nx.get_node_attributes(self.graph, "attrs")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=12)
        nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        plt.title(name)
        plt.axis("off")
        plt.show()

    def nodes_by_utterance(self, utterance: str) -> list[dict]:
        return [node for node in self.graph_dict["nodes"] if utterance in node["utterances"]]

    def edges_by_utterance(self, utterance: str) -> list[dict]:
        return [edge for edge in self.graph_dict["edges"] if utterance in edge["utterances"]]

    def node_by_id(self, id: int):
        for node in self.graph_dict["nodes"]:
            if node["id"] == id:
                return node

    def edge_by_source(self, id: int):
        return [edge for edge in self.graph_dict["edges"] if edge["source"] == id]

    def edge_by_target(self, id: int):
        return [edge for edge in self.graph_dict["edges"] if edge["target"] == id]

    def edges_match_nodes(self) -> bool:
        """Checks whether source and target of all the edges correspond to nodes"""
        graph = self.graph_dict

        for n in graph["nodes"]:
            if not n["utterances"]:
                return False
            src = [e for e in graph["edges"] if e["source"] == n]
            tgt = [e for e in graph["edges"] if e["target"] == n]
            if len(src) == 1 and src == tgt:
                return False
        for e in graph["edges"]:
            if not e["utterances"]:
                return False

        node_ids = set([n["id"] for n in graph["nodes"]])
        node_non_starts = set([n["id"] for n in graph["nodes"] if not n["is_start"]])
        edge_targets = set([e["target"] for e in graph["edges"]])
        edge_sources = set([e["source"] for e in graph["edges"]])
        return node_ids == edge_targets.union(edge_sources) and node_non_starts.issubset(edge_targets)

    def remove_duplicated_edges(self):
        graph = self.graph_dict
        edges = graph["edges"]
        couples = [(e["source"], e["target"]) for e in edges]
        duplicates = [i for i in set(couples) if couples.count(i) > 1]
        new_edges = []
        for d in duplicates:
            found = [c for c in edges if c["source"] == d[0] and c["target"] == d[1]]
            new_edge = found[0].copy()
            new_edge["utterances"] = []
            for e in found:
                new_edge["utterances"].extend(e["utterances"])
            new_edge["utterances"] = list(set(new_edge["utterances"]))
            new_edges.append(new_edge)
        self.graph_dict = {"edges": [e for e in edges if (e["source"], e["target"]) not in duplicates] + new_edges, "nodes": graph["nodes"]}
        return Graph(self.graph_dict)

    def remove_duplicated_nodes(self):
        graph = self.graph_dict
        nodes = graph["nodes"].copy()
        edges = graph["edges"].copy()
        nodes_utterances = [n["utterances"] for n in nodes]
        map(lambda x: x.sort(), nodes_utterances)
        seen = []
        to_remove = []
        for n in nodes:
            utts = n["utterances"]
            utts.sort()
            if utts not in seen:
                seen_utts = list(set([s for xs in seen for s in xs]))
                if any([utt in seen_utts for utt in utts]):
                    return None
                seen.append(utts)
            else:
                doubled = nodes[nodes_utterances.index(utts)]["id"]
                to_remove.append(n["id"])
                for idx, e in enumerate(edges):
                    if e["source"] == n["id"]:
                        edges[idx]["source"] = doubled
                    if e["target"] == n["id"]:
                        edges[idx]["target"] = doubled
        self.graph_dict = {"edges": edges, "nodes": [n for n in nodes if n["id"] not in to_remove]}
        return self.remove_duplicated_edges()

    def all_paths(self, start: int, visited: list[int], repeats: int):
        """Recursion to find all the graph paths with ids of graph nodes
        where node with id=start added to last repeats elements in the visited path do not have any occurance
        visited_list is global variable to store the result"""
        global visited_list
        if len(visited) < repeats or not self._list_in(visited[-repeats:] + [start], visited):
            visited.append(start)
            for edge in self.edge_by_source(start):
                self.all_paths(edge["target"], visited.copy(), repeats)
        visited_list.append(visited)

    def find_path(self, start: int, end: int, visited: list):
        """Recursion to find path from start node id to end node id
        visited is path traveled
        visited_list is global variable to store the result
        """

        global visited_list

        graph = self.graph_dict
        if len(visited) <= len(graph["edges"]) and end not in visited_list[-1]:
            visited.append(start)
            if end not in visited:
                for edge in self.edge_by_source(start):
                    self.find_path(edge["target"], end, visited)
        else:
            visited.append(start)
        visited_list.append(visited)

    def get_ends(self):
        """Find finishing nodes which have no outgoing edges"""

        global visited_list

        graph = self.graph_dict
        sources = list(set([g["source"] for g in graph["edges"]]))
        finishes = [g["id"] for g in graph["nodes"] if g["id"] not in sources]
        if not finishes:
            finishes = [[g["id"] for g in graph["nodes"] if not g["is_start"]][-1]]
        visited = set(finishes.copy())
        for f in finishes:
            for n in graph["nodes"]:
                if n["id"] != f:
                    visited_list = [[]]
                    self.find_path(n["id"], f, [])
                if any([f in v for v in visited_list]):
                    visited.add(n["id"])
        if len(visited) < len(graph["nodes"]):
            finishes += [v["id"] for v in graph["nodes"] if v["id"] not in visited]
        return finishes

    def nodes2list(self) -> list:
        """Returns list of concatenations of all nodes utterances"""
        graph = self.graph_dict
        res = []

        for node in graph["nodes"]:
            utt = ""
            for n_utt in node["utterances"]:
                utt += n_utt + " "
            res.append(utt)

        return res

    def graph2list(self) -> tuple[list[str], int]:
        """Returns:
        res_list - concatenation of utterances of every node and its outgoing edges
        n_edges - total number of utterances in all edges
        """
        graph = self.graph_dict
        res_list = []
        n_edges = 0

        for node in graph["nodes"]:
            edges = [e for e in graph["edges"] if e["source"] == node["id"]]
            utt = ""
            for n_utt in node["utterances"]:
                utt += n_utt + " "
            for edge in edges:
                for e_utt in edge["utterances"]:
                    utt += e_utt + " "
                    n_edges += 1
            res_list.append(utt)
        return res_list, n_edges
