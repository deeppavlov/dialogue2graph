# Experiment name
Dialogue to graph generation

### Issues and goals
My understanding of reasons for this experiment is:
1. To integrate currently working algorithms of dialogue2graph generation (d2gg) into current project workflow.
2. To measure metrics of d2gg task and to keep them high with dataset evolution.

**The Most noticeable issues.**
1. To make gpt-4o-mini solve d2gg
2. Check current metrics algorithms for d2gg (triplet_match and is_same_structure)
3. Add embedding based comparison for utterances in triplet_match
4. Select embedding model
5. Getting criteria for such selection process

## Hypotheses and steps

Approaches we would try to solve the issues.

*We will try*
1-2. *prompt refinement with gpt-4o-mini, few-shot prompting, additional steps in CoT approach, try different models, fine-tuning*
3-4. *BGE-M3 based*
5. *With dataset evolution will be testing embedders*

## Results
First prompt with gpt-4o-mini
Nov 19: general_graph_generation_prompt gives 3 out of 3 on short.json

To compare augmented graphs with original, BGE-M3 embedder was tried first to compare dialogue utterances.
0.13 threshold fitted most pairs  
BGE-M3 embedder and 0.13 threshold for "pairwise_embedding_distance" in langchain evaluator were used, but it gives 0.21 - problem on this pair:

(Welcome to tech support! How can I assist you today?, Thank you for reaching out to tech support! What seems to be the problem today?)
It is true pair, but doesn't satisfy threshold 

Here and further concatenation of utterances (node+edge) was used, it showed better results

So I tried BAAI/bge-reranker-v2-m3 cross-encoder cosine similarity, it is better but shows problems here:

True: 0.9595928 (Welcome to our car service center! How can I assist you today?, Thank you! Is there anything else I can assist you with today?)
False: 0.978375 (Have you tried restarting your laptop to see if that resolves the flickering? Yes, I've tried restarting it, Could you please elaborate on the problem you're facing with your laptop? The display is flickering.)

Nov 22:
gpt-4o showed strange understanding of arithmetics:

"reason": "The dialogue contains 10 utterances, resulting in 5 nodes and 6 edges. The graph is cyclic with a logical cycle point at node 6, which loops back to node 2 to allow for additional reservations."
"reason": "The dialogue contains 10 utterances, resulting in 6 nodes and 6 edges. The graph is cyclic, with the cycle point logically occurring at the node 'ask_room_type', allowing the user to reserve another room."
"reason": "The graph is cyclic with a logical cycle point at node 2, where the user can start a new reservation process. The number of nodes and edges matches the number of utterances in the dialogue, ensuring all assistant and user utterances are included without duplication."
"reason": "The graph is cyclic with a logical cycle point at node 6, which asks for room type again, allowing the user to start a new reservation. The number of nodes matches the number of assistant utterances, and each utterance is used exactly once."

So it creates extra node in many cases, that's why
gpt-4o for 27 examples: 74% of right answers
gpt-4o-mini for 27 examples: 15% of right answers

Nov 25:
It was decided to take pair of graphs as true one when minimal cosine similarity between (node+edge) of graphs >= 0.99, otherwise
gpt-4o with compare_graphs_prompt is used to compare two graphs. The name of algorithm is llm_match.

Nov 27:
o1-mini for 27 examples: 96% of right answers
And llm_match works 100%, still positives only

Dec 10:
Case with several dialogues as components of complex graph.
o1-mini works well with 3 prepared examples
New prompt to handle all existing 27 examples along with new complex examples is developed.
Prompt handles no-cycles as well.

Dec 27:
three stages algo:
1. Generate nodes with gpt-o1
2. Connect nodes with edges via algo with usage of bge-m3 fo searching typo utterances
3. Add missing edges (loops) with gpt-o1
Results:
ready_graphs.json 5/5
complex_graphs.json 5/5
internet_installation_setup.json 0/1

Are_triplets_valid:
ready_graphs.json 2/5
complex_graphs_original.json 2/5

Are_theme_valid:
ready_graphs.json 5/5
complex_graphs_original.json 5/5

Jan 29:\
`dialogue_dataset_10_topics_3_dialogue.json` 0.87\
`ready_graphs.json` 1.0\
`complex_graphs.json` 0.6 found an error in nodes2graph, abstract purchase graph, 1 edge missed

Jan 30:\
These datasets are in experiments/2025.01.22/sampler\
`dialogue_dataset_10_topics_3_dialogue.json` 0.97\
`ready_graphs.json` 1.0\
`complex_graphs.json` 1.0

This is result of three stages algorithm:
1. Group assistant's utterances with nodes2groups
2. Connect nodes by edges with nodes2graph
3. Add missing edges with LLM

## Future plans
All things to be considered by future researchers, plans on next experiments and so on
