:tutorial_name: dialog_augmentation/dialogue_augmentation_example.ipynb

Generate augmented dialogs on one given dialog
==================================================

:py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter` is a class used to augment an original dialog by paraphrasing 
dialog turns while maintaining dialog structure and flow of the conversation.

.. code-block:: python

    from langchain_openai import ChatOpenAI

    from dialog2graph.datasets.augment_dialogs.augmentation import DialogAugmenter
    from dialog2graph.pipelines.model_storage import ModelStorage


First, LLM models should be configured for further use. So, :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` instance 
is created to which choosen LLMs are added for dialog generation (i.e. dialog augmentation) 
and formatting LLM's output. More information on ModelStorage usage may be found :doc:`in this userguide <./generate_graphs>`. 
 
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

Then, :py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter` instance is created to use 
:py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter.invoke` method for getting augmented dialogs. 
:py:class:`~dialog2graph.datasets.augment_dialogs.augmentation.DialogAugmenter` takes previously created 
:py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` instance and the names given to models should be leveraged
for new dialog generation and dialog formatting.

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