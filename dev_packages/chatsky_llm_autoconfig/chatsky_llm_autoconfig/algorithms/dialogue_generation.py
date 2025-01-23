import random
import itertools
import networkx as nx
from chatsky_llm_autoconfig.graph import BaseGraph
from chatsky_llm_autoconfig.algorithms.base import DialogueGenerator
from chatsky_llm_autoconfig.dialogue import Dialogue
from chatsky_llm_autoconfig.autometrics.registry import AlgorithmRegistry


@AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class DialogueSampler(DialogueGenerator):

    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = -1, topic="") -> list[Dialogue]:
        nx_graph = graph.graph
        if end_node == -1:
            end_node = list(nx_graph.nodes)[-1]

        all_dialogues = []
        start_nodes = [n for n, attr in nx_graph.nodes(data=True) if attr.get("is_start", n == start_node)]

        for start in start_nodes:
            # Stack contains: (current_node, path, visited_edges)
            stack = [(start, [], set())]

            while stack:
                current_node, path, visited_edges = stack.pop()

                # Add assistant utterance
                current_utterance = random.choice(nx_graph.nodes[current_node]["utterances"])
                path.append({"text": current_utterance, "participant": "assistant"})

                if current_node == end_node:
                    # Check if the last node has edges and add the last edge utterances
                    edges = list(nx_graph.edges(current_node, data=True))
                    if edges:
                        # Get the last edge's data
                        last_edge_data = edges[-1][2]
                        last_edge_utterance = (
                            random.choice(last_edge_data["utterances"])
                            if isinstance(last_edge_data["utterances"], list)
                            else last_edge_data["utterances"]
                        )
                        path.append({"text": last_edge_utterance, "participant": "user"})

                    all_dialogues.append(Dialogue().from_list(path))
                    path.pop()
                    continue

                # Get all outgoing edges
                edges = list(nx_graph.edges(current_node, data=True))

                # Process each edge
                for source, target, edge_data in edges:
                    edge_key = (source, target)
                    if edge_key in visited_edges:
                        continue

                    # if topic and edge_data.get("theme") != topic:
                    #     continue

                    edge_utterance = random.choice(edge_data["utterances"]) if isinstance(edge_data["utterances"], list) else edge_data["utterances"]

                    # Create new path and visited_edges for this branch
                    new_path = path.copy()
                    new_path.append({"text": edge_utterance, "participant": "user"})

                    new_visited = visited_edges | {edge_key}
                    stack.append((target, new_path, new_visited))

                path.pop()

        return all_dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


@AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class DialoguePathSampler(DialogueGenerator):
    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = -1, topic="") -> list[Dialogue]:
        nx_graph = graph.graph

        # Find all nodes with no outgoing edges (end nodes)
        end_nodes = [node for node in nx_graph.nodes() if nx_graph.out_degree(node) == 0]
        dialogues = []
        # If no end nodes found, return empty list
        if not end_nodes:
            return []

        all_paths = []
        # Get paths from start node to each end node
        for end in end_nodes:
            paths = list(nx.all_simple_paths(nx_graph, source=start_node, target=end))
            all_paths.extend(paths)

        for path in all_paths:
            dialogue_turns = []
            # Process each node and edge in the path
            for i in range(len(path)):
                # Add assistant utterance from current node
                current_node = path[i]
                assistant_utterance = random.choice(nx_graph.nodes[current_node]["utterances"])
                dialogue_turns.append({"text": assistant_utterance, "participant": "assistant"})

                # Add user utterance from edge (if not at last node)
                if i < len(path) - 1:
                    next_node = path[i + 1]
                    edge_data = nx_graph.edges[current_node, next_node]
                    user_utterance = random.choice(edge_data["utterances"]) if isinstance(edge_data["utterances"], list) else edge_data["utterances"]
                    dialogue_turns.append({"text": user_utterance, "participant": "user"})

            dialogues.append(Dialogue().from_list(dialogue_turns))

        return dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)


@AlgorithmRegistry.register(input_type=BaseGraph, output_type=Dialogue)
class RecursiveDialogueSampler(DialogueGenerator):
    def _list_in(self, a: list, b: list) -> bool:
        """Check if sequence a exists within sequence b."""
        return any(map(lambda x: b[x : x + len(a)] == a, range(len(b) - len(a) + 1)))

    def invoke(self, graph: BaseGraph, start_node: int = 1, end_node: int = -1, topic="") -> list[Dialogue]:
        starts = [n for n in graph.graph_dict.get("nodes") if n["is_start"]]
        visitedList = [[]]

        def all_paths(graph, start: int, visited: list):
            # print("start: ", start, len(visitedList))
            if len(visited) < 2 or not self._list_in(visited[-2:] + [start], visited):
                visited.append(start)
                # print("visited:", visited)
                for edge in graph.edge_by_source(start):

                    # if [start,edge['target']] not in visited:
                    all_paths(graph, edge["target"], visited.copy())
            visitedList.append(visited)

        all_paths(graph, starts[0]["id"], [])
        visitedList.sort()
        final = list(k for k, _ in itertools.groupby(visitedList))[1:]
        sources = list(set([g["source"] for g in graph.graph_dict["edges"]]))
        ends = [g["id"] for g in graph.graph_dict["nodes"] if g["id"] not in sources]
        node_paths = [f for f in final if f[-1] in ends]
        full_paths = []
        for p in node_paths:
            # print(p)
            path = []
            for idx, s in enumerate(p[:-1]):
                path.append({"participant": "assistant", "text": graph.node_by_id(s)["utterances"][0]})
                # path.append({"user": list(set(gr.edge_by_source(s)) & set(gr.edge_by_target(p[idx+1])))[0]['utterances']})
                sources = graph.edge_by_source(s)
                targets = graph.edge_by_target(p[idx + 1])
                # print("SOURCES: ", sources, s)
                # print("TARGETS: ", targets, p[idx+1])
                # targets = set([(e['source'],e['target']) for e in gr.edge_by_target(p[idx+1])])
                edge = [e for e in sources if e in targets][0]
                path.append(({"participant": "user", "text": edge["utterances"][0]}))
            path.append({"participant": "assistant", "text": graph.node_by_id(p[-1])["utterances"][0]})
            full_paths.append(path)

        dialogues = [Dialogue().from_list(i) for i in full_paths]

        return dialogues

    async def ainvoke(self, *args, **kwargs):
        return self.invoke(*args, **kwargs)
