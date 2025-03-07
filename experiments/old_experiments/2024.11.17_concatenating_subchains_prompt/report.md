# Appending dialogue paths to existing graphs (prompting approach)

### Issues and goals

LLMs consistently fail in generating multibrach dialogue graphs when given just a dialogues.
We will try to resolve this using branch-by-branch approach:

1. We start with dialogue, a chain graph describing this dialogue and another dialogue acting as another branch of the original dialogue.
2. We prompt model with these data asking to return a graph that has both branches
3. We do this until graph has all of the branches we've created

Prompt:

    You will be given a dialogue and a graph that describes this dialogue: edges' utterances are corresponding to users requests and nodes' utterances are corresponding to assistants' responses. You will also be given with another branch of this dialogue. You tasks are so:
    1. Determine which parts of the dialogues are the same and which are different.
    2. Append the brach defined by second dialogue to the graph -- add corresponding edges and nodes to the graph.
    3. Return graph that describes both dialogues.

    **Rules:**
    1) Responses must acknowledge what the user has already specified
    2) Each node must have clear purpose
    3) Return ONLY the JSON without commentary and codeblocks
    4) All edges must connect to existing nodes
    5) The graph paths should make logical sense
    6) Nodes and edges that bears same meaning must not be duplicated, reuse them when possible
    7) If nodes has same uterances the should be merged into one
    
    Original dialogue: {orig_dial}
    Original graph: {orig_graph}

    Additional branch: {new_dial}
