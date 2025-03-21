# Experiment name
Dia2graph generation with three different approaches

## Hypotheses and steps
1. LLMGenerator: Generating graph from dialogues with dia2graph generator where nodes are grouped via LLM
2. AlgoGenerator: Generating graph from dialogues with dia2graph generator where nodes are grouped via algorithm
3. LLMExtender: Demonstration of incremental approach where first graph generated from 1st dialogue, then next step is generating graph via LLM from previous graph and next dialogue

## Results

Both approaches 1 and 2 show stable results for graphs generated with our TopicGenerator
LLM approach is used with chatgpt-4o-latest and temprerature 0
Approach with LLM Extender is showed just for demonstration. With current data from TopicGenerator, it doesn't make sense to use this incremental approach. 

## Future plans
All things to be considered by future researchers, plans on next experiments and so on
