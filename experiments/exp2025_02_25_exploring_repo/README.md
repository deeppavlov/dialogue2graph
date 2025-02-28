# Graph generation review

## Found code issues

1. dialogue2graph\datasets\complex_dialogues\generation.py, line 18 - no `utils` module is provided
2. dialogue2graph\datasets\complex_dialogues\generation.py, line 240 - call function for `GenerationPipeline` class has no `use_cache` argument. `use_cache` is a class attribute. 
3. dialogue2graph\datasets\complex_dialogues\generation.py, line 57 - type mismatch when creating class `Graph`
4. dialogue2graph/pipelines/core/dialogue_sampling.py, line 5 - wrong `Dialogue` class import (it should be imported from another py file)
5. Missing scipy package for `Graph` class

## Found README weaknesses

1. No system requirement found (poetry doesn't work with Python 3.12)
2. Probably it lacks poetry installation step (pip install poetry, smth like that)
3. Need of creating an `.env` file should be expressed clearer
4. Data generation code example is incorrect and copied from source code
5. Same code example - `)` is missing in line 17
6. `deepseek/deepseek-reasoner` is not available
7. For further work it would be great to have each task description, e.g. data generation enables to get dialogue on certain topics using LLMs, etc.
8. Also it would be great to list available models or at least link the source where this info could be found

## Found prompt weaknesses

**Note**: it is related to the initial `create_graph_prompt`

1. Why utterances in the graph example (dialogue2graph\pipelines\cycled_graphs\prompts\prompts.py lines 35-73) are sometimes given as strings, sometimes as a list of strings?
2. dialogue2graph\pipelines\cycled_graphs\prompts\prompts.py lines 35: `[` should probably come last, not first

## Found pipeline weaknesses

1. No model (gpt-4o, gpt-4o-mini, gpt-3.5) could generate dialog graph to be parsed