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
    def __init__(self, graph_dict: dict, **kwargs: Any):
        # Pass graph_dict to the parent class
        super().__init__(graph_dict=graph_dict, **kwargs)
        if graph_dict:
            self.load_graph()

    def _is_seq_in(self, a: list, b: list) -> bool:
        """Check if sequence a exists within sequence b."""
        return any(map(lambda x: b[x : x + len(a)] == a, range(len(b) - len(a) + 1)))

    def check_edges(self, seq: list[list[int]]) -> bool:
        """Checks whether seq (sequence of pairs (source, target)) has all the edges of the graph"""
        graph_dict = self.graph_dict
        edge_seq = {(e["source"], e["target"]) for e in graph_dict["edges"]}
        left = edge_seq.copy()
        for pair in edge_seq:
            for s in seq:
                if self._is_seq_in(list(pair), s):
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
        return [node for node in self.graph_dict["nodes"] if utterance in node["utterances"]]

    def find_edges_by_utterance(self, utterance: str) -> list[dict]:
        return [edge for edge in self.graph_dict["edges"] if utterance in edge["utterances"]]

    def get_nodes_by_id(self, id: int):
        for node in self.graph_dict["nodes"]:
            if node["id"] == id:
                return node

    def get_edges_by_source(self, id: int):
        return [edge for edge in self.graph_dict["edges"] if edge["source"] == id]

    def get_edges_by_target(self, id: int):
        return [edge for edge in self.graph_dict["edges"] if edge["target"] == id]

    def match_edges_nodes(self) -> bool:
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
        return node_ids == edge_targets.union(
            edge_sources
        ) and node_non_starts.issubset(edge_targets)

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
        self.graph_dict = {
            "edges": [e for e in edges if (e["source"], e["target"]) not in duplicates]
            + new_edges,
            "nodes": graph["nodes"],
        }
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
        self.graph_dict = {
            "edges": edges,
            "nodes": [n for n in nodes if n["id"] not in to_remove],
        }
        return self.remove_duplicated_edges()

    def get_all_paths(self, start_node_id: int, visited_nodes: list[int], repeats_limit: int):
        """Recursion to find all the graph paths consisting of nodes ids
        which start from node with id=start_node_id
        and do not repeat last repeats_limit elements of the visited_nodes"""

        visited_paths = [[]]
        if len(visited_nodes) < repeats_limit or not self._is_seq_in(visited_nodes[-repeats_limit:] + [start_node_id], visited_nodes):
            visited_nodes.append(start_node_id)
            for edge in self.get_edges_by_source(start_node_id):
                visited_paths += self.get_all_paths(edge["target"], visited_nodes.copy(), repeats_limit)
        visited_paths.append(visited_nodes)
        return visited_paths

    def find_paths(self, start_node_id: int, end_node_id: int, visited_nodes: list):
        """Recursion to find path from start_node_id to end_node_id
        visited_nodes is path traveled so far"""
        visited_paths = [[]]

        graph = self.graph_dict
        if len(visited_nodes) <= len(graph["edges"]) and end_node_id not in visited_paths[-1]:
            visited_nodes.append(start_node_id)
            if end_node_id not in visited_nodes:
                for edge in self.get_edges_by_source(start_node_id):
                    visited_paths += self.find_paths(edge["target"], end_node_id, visited_nodes)
        else:
            visited_nodes.append(start_node_id)
        visited_paths.append(visited_nodes)
        return visited_paths

    def get_ends(self):
        """Find finishing nodes which have no outgoing edges"""

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

    def get_list_from_nodes(self) -> list:
        """Returns list of concatenations of all nodes utterances"""
        graph = self.graph_dict
        res = []

        for node in graph["nodes"]:
            utt = ""
            for n_utt in node["utterances"]:
                utt += n_utt + " "
            res.append(utt)

        return res

    def get_list_from_graph(self) -> tuple[list[str], int]:
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
