## Finding out best parameters for start and end loop detection

### Finding start and end nodes.ipynb

* Common start and end phrases were defined by frequency.
* Top 10 start phrases by frequency look too specific and contain specific detail on the dialog topic. To fix it
    - common greetings and following questions were defined among most common openings,
    - combinations of greeting and question were set as general start phrase set.
* Top 10 end phrases being enough general were chosen as general end phrase set.

### Selecting base params for start loop finding.ipynb

**Finding embedder threshold to detect similar utterances**

* Pairwise distance is computed between defined set of START_TURNS (16 utterances chosen earlier) and all utterances in d2g_generated dialogs,
* Mean distance is counted for each pair (start_turn, dialog_utterance) and then the average is calculated across each start turn.
* Start phrases have mean distance around 0.26, whereas non-start have mean distance around 0.43. Thus, threshold can be set as 0.26.

Same process was repeated for END_TURNS. End phrases have mean distance around 0.27, and non-end phrases around 0.47. This threshold can be set a bit higher as 0.27.

**Testing new validators on good and damaged dialogs from graph**

1. The graph examples were chosen - one for good graph, one for graph where opening phrase appears in the middle, one for graph where closing phrase appears in the middle
2. New llm validators were used to eval the dialogs from chosen dialogs (start and end thresholds were lowered to 0.2 as the results were more precise with this thresholds).

### Extra findings

* There appear such phrases as "Certainly! How can I assist you today?" which look very alike to start turns. So a smarter LLM should be used to detect them as non-opening phrase.
* There are too large greeting phrases (above 40 symbols), so they are not very similar to usual opening phrases, even if splited, as they contain very specific info. Need to be analysed.
