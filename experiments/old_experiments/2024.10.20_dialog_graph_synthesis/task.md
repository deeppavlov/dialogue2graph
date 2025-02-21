# Cycle Dialog Creation Method

## Algorithm Overview

1. **Cycle Graph Creation**
   - **Input:** A prompt containing an example with one topic (can be applied to various topics).
   - **Output:** A graph `g` representing the dialog structure.

2. **Best Graph Selection**
   - **Input:** A triplet of the form `(node, edge, node)` representing a part of the dialog.
   - **Output:** A true/false evaluation (denoted as `v`) indicating if the selected graph path is valid.

3. **Dialog Sampling**
   - **Input:** The selected graph `g`.
   - **Output:** A sampled dialog `d` generated from the graph.

4. **Dialog Augmentation**
   - **Process:** Rewriting each piece of text in the context to generate variations of the dialog.

---

## Example Data

### Task 1: Graph Examples (g)
#### Graph 1:
- **n1 (start):** "Привет!"
- **e2:** "Привет!"
- **n3:** "Как твои дела?"
- **e4:** "Отлично, как твои!"
- **n5:** "Тоже"
- **e6:** "Пока"
- **n7 (end):** "Пока"

#### Graph 2:
- **n1 (start):** "Привет!"
- **e2:** "Привет!"
- **n3:** "Как твои дела?"
- **e5:** "Плохо, как твои!"
- **n6:** "А мои хорошо"
- **e6:** "Пока"
- **n7 (end):** "Пока"

---

### Task 2: Graph Selection
#### Validation Examples (v):
Validation consists of triplets `(node, edge, node)` that represent transitions in the dialog graph. Some examples include:

- "Привет!, Привет!, Как твои дела?"
- "Привет!, Собака, Как твои дела?"

The second example is invalid since "Собака" (Dog) doesn’t logically follow a greeting in the dialog.

---

### Task 3: Selected Graph
Based on validation, a dialog graph is selected. Example:

- **n1 (start):** "Привет!"
- **e2:** "Привет!"
- **n3:** "Как твои дела?"
- **e5:** "Плохо, как твои!"
- **n6:** "А мои хорошо"
- **e6:** "Пока"
- **n7 (end):** "Пока"

---

### Task 4: Augmented Dialogs
Once the graph is selected, each dialog can be augmented by rephrasing parts of the conversation. Here are some examples of the augmented dialogs:

#### Dialog 1:
- **n1 (start):** "Добрый день!"
- **e2:** "Добрый!"
- **n3:** "Как настроение?"
- **e5:** "Плохо, как твои!"
- **n6:** "А мои хорошо"
- **e6:** "Пока"
- **n7 (end):** "Пока"

#### Dialog 2:
- **n1 (start):** "Привет!"
- **e2:** "Привет!"
- **n3:** "Как твои дела?"
- **e4:** "Отлично, как твои!"
- **n5:** "Тоже"
- **e6:** "Пока"
- **n7 (end):** "Пока"

#### Dialog 3:
- **n1 (start):** "Добрый день!"
- **e2:** "Добрый!"
- **n3:** "Как настроение?"
- **e5:** "Плохо, как твои!"
- **n6:** "А мои хорошо"
- **e6:** "Пока"
- **n7 (end):** "Пока"

---

This process allows for generating and validating dialog graphs and producing a variety of dialogs through augmentation.