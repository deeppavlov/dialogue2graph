# Getting started with synthetic dialog augmentation

## Experiment setup

The experiment is a continuation of the work on augmentation of the synthetic dataset. 

Pipeline:
1. An analysis of errors in the generation of augmented dialogs was performed (errors of inconsistency of length and roles between the original and augmented dialog, an error of inconsistent generation of variations of the utterance).

2. Dialogs with errors (dialog interruption, loss of context) were removed from the dataset. Dialogs subjectively assessed as having satisfactory quality were re-augmented.

3. After the re-augmentation, the check for various errors was performed again. After that, the final data cleaning and combination of new augmented dialogs from variations of each utterance were performed.

## Results and observations

A new synthetic dataset with augmented dialogs was obtained. The dataset contains 376 examples (graphs), 4104 original dialogs and 11560 augmented dialogs.

## Future plans

In the future, it is planned to upload a dataset to Hugging Face.