:tutorial_name: dialog_augmentation/dialogue_augmentation_example.ipynb

Generate augmented dialogues on one given dialogue
==================================================

Use :py:class:`~dialogue2graph.datasets.augment_dialogues.augmentation.DialogueAugmenter` to augment an original dialogue by paraphrasing 
its lines while maintaining the structure and flow of the conversation.

.. code-block:: python

    from langchain_openai import ChatOpenAI

    from dialogue2graph.datasets.augment_dialogues.augmentation import DialogueAugmenter
    from dialogue2graph.pipelines.model_storage import ModelStorage


1. Create :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` instance and add choosen LLMs for dialogue generation (i.e. dialogue augmentation) 
and formatting LLM's output.
 
.. code-block:: python

    model_storage = ModelStorage()
    model_storage.add(
            key="generation-llm",
            config={"model_name": "gpt-4o-mini-2024-07-18", "temperature": 0.7},
            model_type=ChatOpenAI
        )
    model_storage.add(
            key="formatting-llm",
            config={"model_name": "gpt-3.5-turbo", "temperature": 0.7},
            model_type=ChatOpenAI
        )

2. Create :py:class:`~dialogue2graph.datasets.augment_dialogues.augmentation.DialogueAugmenter` instance and use 
:py:class:`~dialogue2graph.datasets.augment_dialogues.augmentation.DialogueAugmenter.invoke` method to get augmented dialogues.

.. code-block:: python

    augmenter = DialogueAugmenter(
            model_storage=model_storage,
            generation_llm="generation-llm",
            formatting_llm="formatting-llm"
        )
    result = augmenter.invoke(
        dialogue=dialogue, # original dialogue as a Dialogue object
        topic=topic, # topic of the original dialogue as a string
        prompt=augmentation_prompt # prompt for dialogue augmentation as a string
        )