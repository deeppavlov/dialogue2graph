# Incrementation with recomended models

## Experiment setup

The incremental approach was tested with the recommended generation models which are: o1-mini, o3-mini, and chatgpt-4o-latest. Test data is example 13 (serial number) from synthetic data subset called "generated_24". This example is one graph with four dialogues sampled from this graph. 
Tested incrementation pipeline:
1. take a graph from the example as a true graph
2. generate the first graph from the first dialogue in the example (same generation method for all models)
3. generate the next 3 graphs subsequently using the selected incrementation method, adding new dialogues one by one (i.e. in the end we get 4 graphs - for dialogue 0, dialogues 0-1, dialogues 0-2, dialogues 0-3)
4. if generation is successful, get metrics for the last incremental graph and the true graph

## Results and observations
1. The best result was obtained using the ThreeStagesGenerator method and o3-mini as a generation model.
2. AppendChain method doesn't work well for this task, regardless of the generation model.
3. chatgpt-4o-latest model has tendency to fail graph generation (returns error with incorrect output format).
4. In the dialogue 0 two assistant's utterances are the same - doesn't matter?

## Future plans and suggestions
1. Test ThreeStagesGenerator with o3-mini on more complex examples (or on all the dataset), get metrics +/- error analysis
2. Test other pipelines: 
    - take the longest dialogue as the first one and continue as before
    - take all the dialogues at once and generate a general graph of them