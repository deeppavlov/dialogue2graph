import random
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
