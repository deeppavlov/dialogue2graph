:tutorial_name: dialog_augmentation/dialog_augmentation_example.ipynb

Generate augmented dialogs on one given dialog
==================================================

Use :py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter` to augment an original dialog by paraphrasing 
its lines while maintaining the structure and flow of the conversation.

.. code-block:: python

    from langchain_openai import ChatOpenAI

    from dialog2graph.datasets.augment_dialogs.augmentation import DialogAugmenter
    from dialog2graph.pipelines.model_storage import ModelStorage


1. Create :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` instance and add choosen LLMs for dialog generation (i.e. dialog augmentation) 
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

2. Create :py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter` instance and use 
:py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter.invoke` method to get augmented dialogs.

.. code-block:: python

    augmenter = DialogAugmenter(
            model_storage=model_storage,
            generation_llm="generation-llm",
            formatting_llm="formatting-llm"
        )
    result = augmenter.invoke(
        dialog=dialog, # original dialog as a Dialog object
        topic=topic, # topic of the original dialog as a string
        prompt=augmentation_prompt # prompt for dialog augmentation as a string
        )