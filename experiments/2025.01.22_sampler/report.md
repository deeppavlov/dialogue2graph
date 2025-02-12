# Method to find node finishing cycle in the cyclic graph: find_graph_ends

## Issues and goals

For sampling dialogues in a graph we need to limit set of finishing nodes in a sampled sequence.

## Hypothesises and steps

First hypothesis is to find ends of cycles

## Results

`dialogue_dataset_10_topics_3_dialogue.json` is original data of 10 simple graphs from Dmitry

Metric results with o1-mini:

- find_graph_ends: 1.0

`complex_graphs.json` is data of 5 complex graphs from Andrey modified by Yuri

Metric results with o1-mini:

- find_graph_ends: 1.0

`ready_graphs.json` is data of 5 complex graphs from Dmitry modified by Yuri

Metric results with o1-mini:

- find_graph_ends: 1.0

Feb-12-2025:
generated_plus.json is 182 graphs with full sets of dialogues.
Sampler issues were resolved:
-- all_utterances_present fixed, but still it gives error when utterances are uplicated
-- right ends search for dialogue paths was modified, now it finds nodes from where there is no path to simple ends without outgoing edges
-- graphs with edges wrongly correlated with nodes were removed
-- duplicated edges were removed

## Future plans

All things to be considered by future researchers, plans on next experiments and so on