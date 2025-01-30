three_1 = """Your input is a dialogue graph from customer chatbot system - it is
a set of nodes with assistant's utterances and a set of edges that are
triggered by user requests: """

three_2 = """This is the end of input graph.
**Rules:**
1) is_start field in the node is an entry point to the whole graph.
2) Nodes are assistant's utterances, edges are utterances from the user.
3) Dialogue flow goes from source to target in every edge.
Please consider the graph above, list of dialogues below and do the following:
4) For every pair (let's call it current pair) of user's utterance (let's call it current utterance)
and assistant's utterance right before it, not present in input graph,
you shall find in the graph the most suitable continuation node of a dialogue flow
and add to the graph new edge with current utterance connecting to this node.
Source field of this new edge shall be node with assistant's utterance previous to current utterance
in any of dialogues.
5) To find most suitable continuation node you must understand that
missing user's utterance shall loop back to one of previous nodes where most suitable reply is located.
6) To choose the most suitable answer from two equal candidates choose:
firstly - one with similar phrases, then one with current problem elaboration step,
then one closest to the beginning of the graph.
7) Typically it is a clarifying question to current utterance or previous assistant's question.
8) You must use existing nodes only, don't create new nodes.
9) So it is necessary to add new edges to the input graph from utterances which may exist in dialogues but absent in the graph.
Source of such an edge shall be node with assitant's utterance in current pair, target will be the continuation node.
10) Never modify existing nodes in the input graph.
11) Never add utterances to existing edges.
12) Never create new or modify existing utterances from list of dialogues.
13) When source and target of one newly added edge are both equal to source and target of another newly edge, these edges shall be combined in one edge
with list of utterances containing utterances from both edges.
14) You must check all the dialogues and add this new edge to the node with assistant's utterance from current pair for all the dialogue.
15) You must take not just all missing user's utterances but their pairs with previous assistant's utterances in all the dialogues.
For example if there is an edge with 'Yes' answer already, but source node differs from previous assistant's utterance for cutrrent utterance, you shall
find right node and create new edge.
16) Also give all the nodes suitable labels.
17) Add reason point to the graph with your explanation which edges you added and why. And why you modified existing edges if any.
I will give a list of dialogues, your return is a fixed version of dialogue graph above according to the rules above.
List of dialogues: """
