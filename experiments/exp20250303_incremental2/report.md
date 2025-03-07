# Comparing incrementation

## Issues and future plans

1. One of the incremental methods doesn't work with gpt-3.5-turbo. The other two work but how well remains unclear. Need to experiment with other LLMs.
2. Need to get an example graph and dialogues from generated dataset so as to have a golden graph. In this way we will be able to conduct correct comparisons and scoring.
2. Pipeline is decided to be the following:
    1) take 1 graph
    2) take the longest dialogue of this graph
    3) add other dialogues one by one to the longest one
    4) after adding each dialogue to the graph - compare result graph with golden graph
3. Decided to use one more metric - "compare_graphs" (uses LLM)