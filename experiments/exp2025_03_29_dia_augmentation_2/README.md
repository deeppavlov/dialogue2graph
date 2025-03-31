# Synthetic dialogue augmentation

## Experiment setup

During the experiment, dialogues from the entire synthetic dataset were augmented.

__Experiment pipeline:__
1. augmentation of remaining dialogues from the synthetic dataset;
2. verification and cleaning of generated data;
3. adding augmented utterances to nodes and edges of the corresponding original graph;
4. combining new augmented dialogues.

__About each step:__
1. Augmentation - before that, a check of the augmented dialogue for compliance with the original dialogue in terms of length and roles was added to the augmentation script.

2. Verification and cleaning - after augmentation, the generated data was checked:
- for any kind of generation errors;
- for the number of augmented variants of each phrase in the dialogue (there should be 3 variations for all cases);
- for duplicates among these variants for one phrase (there shouldn't be any).
Cases that failed at least one of the above checks were removed from the dataset.

3. Adding augmented utterances to nodes and edges - while removing of duplicates in each node or edge.

4. Combining new augmented dialogues - since the number of augmented variations for each phrase in the dialogue is 3, then from each augmented dialogue obtained as a result of the steps above, 3 new augmented dialogues were combined.
These augmented dialogues and their corresponding graphs and original dialogues formed the final dataset.

## Results

A synthetic dataset with augmented dialogs was obtained. The dataset contains 396 examples.

## Issues and future plans

Remaining issues:
1. The final dataset does not contain all the original dialogues, because some of them had errors during augmentation. These dialogues/examples were removed. Error analysis and re-augmentation should be perfomed later on. 
2. The final dataset contains dialogues/graphs with a greeting in the middle. They will be removed or replaced with new ones (without this error) in the next experiments.