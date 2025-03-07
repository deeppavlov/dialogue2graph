# Graph based generation and augmentation of the synthetic dialogues

### Issues and goals

Metrics were not satisfying enough and when generating a huge batches of data it is almost impossible to verify generated dialogues and texts by hand.
So we will try to increase robustness of the generated synthetic data using approach similar in some sense to the [Skeleton of Thought](https://arxiv.org/abs/2307.15337).

## Hypothesises and steps

The overview of the approach is so:
- Firstly we create a graph of desired configuration
- Then we send this graph in a networkx notation and ask a LLM to fill the nodes and edges with the utterances
- We iterate through triplets and ask another model if such a triplet can exist (it is logical and sensible)
- If not we mark this triplet and fault and continue
- We sent faulted triplets to the LLM and re-itrerate several times
- When we are satisfied with resulted graph we ask LLM to augment it to other themes and situations
- We move to the next graph

## Results
Created a pipeline for utterance generation for existing graph. Created sampler that returns dialogs from graph. Working on a augmentation.
Is it better to pass a graph for augmentation or just dialog?

For visualisation sake we can use average graph - overlayed multiple graphs where the complementary edges are thicker (also may include average cosine distance between nodes/edges utterances of these graphs as weight)

## Future plans
All things to be considered by future researchers, plans on next experiments and so on
