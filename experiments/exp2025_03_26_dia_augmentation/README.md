# Synthetic dialogue augmentation

## Experiment setup

The experiment investigated the dialogue augmentation by paraphrasing one original phrase into several augmented phrases. Several prompt variations have been tested. With the best prompt, 100 examples from a synthetic dataset were augmented.
The existing methods for evaluating augmented dialogues have also been updated, and an assessment for duplicates in a single list of augmented phrases (corresponding to one original phrase) has been added.

## Results and observations

Subjectively, the augmentation result is assessed as acceptable. Remaining issues:
1. Incorrect length of the augmented dialog (few cases).
2. Duplicate phrases within one dialog.
3. Duplicate phrases within all augmented dialogues corresponding to the same graph. 

## Suggestions

Possible solutions to the issues listed above are the following:
1. Incorrect length - error analysis; if the patterns that can lead to such an error are not identified, one can add a re-generation of the error dialog to the augmentation script.
2. Duplicate within one dialog - is it possible to avoid this by combining different variants of augmented phrases?
3. Duplicate phrases within all augmented dialogues - may this issue also be solved on the next steps of augmentation? (re-augmentation)