# Testing examples

## Goals

  Нужны примеры шагов пайплайна работы с данными. 

__Пайплайн работы на текущий момент:__

1. Генерация синтетических графов (dialogue2graph.datasets.complex_dialogues.generation.LoopedGraphGenerator)

2. Сэмплирование графов и генерация диалогов по графу (dialogue2graph.pipelines.core.dialogue_sampling.RecursiveDialogueSampler)

3. Генерация графа по диалогам

4. Оценка алгоритмов генерации графов по диалогам с помощью метрик


__Какие примеры уже есть:__

1. Генерация синтетических графов - пример есть, после решения проблем с кодом (utils, use_cache) возникла ошибка:
```bash
❌ Failed to generate graph for antivirus software problems Error type: ErrorType.GENERATION_FAILED Error message: Unexpected error during generation: 1 validation error for Graph graph_dict Input should be a valid dictionary [type=dict_type, input_value=DialogueGraph(edges=[Edge...he future. Goodbye!'])]), input_type=DialogueGraph] For further information visit [https://errors.pydantic.dev/2.10/v/dict_type](https://errors.pydantic.dev/2.10/v/dict_type)
```
2. Генерация диалогов по графу - отдельного примера нет

3. Генерация графа по диалогам - пример есть (пример в ветке mu_dia2graph в ноутбуке experiments/exp20250207_three_stages_plus/graph_generation.ipynb)

4. Оценка алгоритмов генерации графов по диалогам - один пример есть (examples/evaluation/evaluate_algorithm.ipynb), расчет метрики на готовых данных


##  Suggestions

Что можно добавить:
1. Генерация синтетических графов - визуализация графов в пример генерации графов; генерация разных графов (cycled graphs, simple graphs)
2. Генерация диалогов по графу - отдельный пример генерации диалогов? Например, с выводом в читаемом виде
3. Генерация графа по диалогам - визуализация полученных графов?
4. Оценка алгоритмов генерации графов по диалогам - расчет других метрик
5. Аугментация диалогов?