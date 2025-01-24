three_1 = """Your input is a dialogue graph from customer chatbot system - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

three_2 = """This is the end of input graph.
**Rules:**
1) is_start field in the node is an entry point to the whole graph.
2) Nodes are assistant's utterances, edges are utterances from the user.
Please consider the graph above, list of dialogues below and do the following:
3) For every user's utterance (let's call it current utterance) not present in input graph
you shall find in the graph the most suitable continuation node of a dialogue flow
and add to the graph new edge with current utterance connecting to this node.
Source field of this new edge shall be node with assistant's utterance previous to current utterance
in any of dialogues. 
4) To find most suitable continuation node:
Firstly try to find in the graph user's utterance most similar
to current utterance and set target node
of an edge with that similar utterance as a target in point 3.
5) It is necessary to choose the most suitable answer.
From two equal candidates choose:
firstly - one with similar phrases, then one closest to the beginning of the graph.
6) If nothing is found, search for a node with current problem elaboration step.
7) Typically it is a clarifying question to current user's utterance.
8) You must use existing nodes only, don't create new nodes.
9) So it is necessary to add new edges to the input graph from utterances which may exist in dialogues but absent in the graph.
10) Never modify existing nodes in the input graph.
11) Never add utterances to existing edges.
12) Never create new or modify existing utterances from list of dialogues.
13) When source and target of one newly added edge are both equal to source and target of another newly edge, these edges shall be combined in one edge
with list of utterances containing utterances from both edges.
14) Also give all the nodes suitable labels.
15) Add reason point to the graph with your explanation which edges you added and why. And why you modified existing edges if any.
I will give a list of dialogues, your return is a fixed version of dialogue graph above according to the rules above.
List of dialogues: """
