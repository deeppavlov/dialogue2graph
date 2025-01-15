# Method for validation of dialogue begin and end: is_dialogue_valid

## Issues and goals

Check if dialgue starts and finished correctly

## Hypothesises and steps

Modifying prompts in are_triplets_valid method

## Results

`new_data.json` is current data of 10 graphs from Andrey

Metric results with gpt-4o-mini:

- are_triplets_valid: 0.6
- is_theme_valid: 1.0
- all_utterances_present: 0.8
- is_dialogue_valid: 0.91

Metric results with gpt-4o:

- are_triplets_valid: 0.7
- is_theme_valid: 1.0
- all_utterances_present: 0.8

`complex_graphs.json` is data of 5 graphs from Andrey modified by Yuri

Metric results with gpt-4o-mini:

- are_triplets_valid: 0.2
- is_theme_valid: 1.0
- all_utterances_present: 1.0
- is_dialogue_valid: 0.74


## Future plans

All things to be considered by future researchers, plans on next experiments and so on