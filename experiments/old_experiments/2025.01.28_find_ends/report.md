# Method to sample dialogs in a graph: get_full_dialogs

## Issues and goals

To genearate graphs from dialogs first we need dialogs.
Dialogs are sampled from the graphs generated from scratch by LLM.
Dialog sampler used so far and based on networkx gives poor results: not all dialogs are sampled.
So it was decided to try new way.

## Hypotheses and steps

Will use regular recursion.
1. Find finishing nodes with the help of LLM based find_graph_ends().
2. Get starting node and call recursion - all_paths() gives node paths.
3. Limit paths by those where sequence of length n doesn't repeat.
4. Select only paths ending with finishing nodes.
5. Remove paths with duplicates in edges when possible.
6. With all_combinations() recursion (arguments: starting utterance and next index) add user's utterances.
7. Remove paths with duplicated utterances when possible.
8. Increase length of repeated sequence until all utterances present.

## Results

`dialog_dataset_10_topics_3_dialog.json` is original data of 10 simple graphs from Dmitry
`complex_graphs.json` is data of 5 complex graphs from Andrey modified by Yuri
`ready_graphs.json` is data of 5 complex graphs from Dmitry modified by Yuri

100% utterances present with maximum of 6 dialogs

## Future plans

All things to be considered by future researchers, plans on next experiments and so on