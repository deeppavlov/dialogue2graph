# Loading real datasets into new single dataset for testing d2g task 

## Issues and goals

To load datasets, huggingface loading script to be used.
But datasets.download.DownloadManager didn't work so had to code loading datasets and processing them,
with usage of load_dataset from huggingface again.
After loading it's needed to push dataset to huggungface hub.

## Results

1. SMD, MULTIWOZ2_2, META_WOZ, SCHEMA, Frames, TaskMaster3 and WOZ datasets are loaded from internet.
2. Only MSR E2E is not downloaded from https://github.com/xiul-msr/e2e_dialog_challenge/tree/master/data
successfully, so downloaded manually then processed with the loading script.
3. Private dataset pushed to the hub by this [link](https://huggingface.co/datasets/DeepPavlov/d2g_real_dialogs).

## Future plans

All things to be considered by future researchers, plans on next experiments and so on