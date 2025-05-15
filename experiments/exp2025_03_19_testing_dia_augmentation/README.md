# Getting started with synthetic dialog augmentation

## Experiment setup

The basic approach to dialog augmentation was tested using a prompt tested in earlier experiments (naive augmentation of the type "one original phrase - one augmented phrase"). The gpt-4o-mini model was used for augmentation.

The following checks have been developed to assess the quality of augmentation: 
- checking for duplicates among dialogs related to the same graph;
- checking for duplicates within one dialog;
- checking for matching the roles and length of the original dialog and the augmented dialog.

Augmentation in the one-shot approach was also briefly tested. Rejected due to a large waste of tokens.

## Results and observations

The biggest problem according to the results of the checks is the large number of duplicates of both cross-dialog and intra-dialog types. 

## Future plans

Decided to explore the possibility of augmentation of the type "one original phrase - several variants of the augmented phrase" in order to avoid the problem of duplicates in the future.