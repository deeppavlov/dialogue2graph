:tutorial_name: data_generation/LoopedGraphGenerator_example.ipynb

Generate synthetic graph on certain topic
=========================================

:py:class:`~dialog2graph.datasets.complex_dialogs.generation.LoopedGraphGenerator` is a class to create a validated graph from several 
LLM generated dialogs concerning a given topic. This class represents algorithm to generate dialog graph following several steps: 
graph creation from dialogs, graph validation evaluating theme consistency and cycle validity, dialog generation from graph.

.. code-block:: python

    from langchain_openai import ChatOpenAI

    from dialog2graph.datasets.complex_dialogs.generation import LoopedGraphGenerator
    from dialog2graph.pipelines.model_storage import ModelStorage

First, :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` instance is created. Then, using 
:py:class:`~dialog2graph.pipelines.model_storage.ModelStorage.add` method, choosen LLMs may be added for dialog generation, 
dialog validation, theme validation and cycle end search, important generation steps. You can choose between available models. 
Preferably, the stronger model is used for generation problem and the simpler models are used for other tasks.

.. code-block:: python

    model_storage = ModelStorage()
    model_storage.add(
        "gen_model", # model to use for generation
        config={"model_name": "o1-mini"},
        model_type=ChatOpenAI,
    )
    model_storage.add(
        "help_model", # model to use for other tasks
        config={"model_name": "gpt-3.5-turbo"},
        model_type=ChatOpenAI,
    )

Then, :py:class:`~dialog2graph.datasets.complex_dialogs.generation.LoopedGraphGenerator` is created, passing 
previously defined :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` and specifying model names for 
each task. :py:class:`~dialog2graph.datasets.complex_dialogs.generation.LoopedGraphGenerator.invoke` method is used to get a dialog graph 
on ``restaurant reservation`` topic. The method returns a dictionary (``"graph": {{"edges": [...], "nodes": [...]}, "dialogues": [...]}``) 
that can be passed to :py:class:`~dialog2graph.Graph` (more info you can find :doc:`in this userguide <./basic_usage>`). 

.. code-block:: python

    pipeline = LoopedGraphGenerator(
        model_storage=model_storage,
        generation_llm='gen_model',
        validation_llm='help_model',
        cycle_ends_llm='help_model',
        theme_validation_llm='help_model'
    )

    generated_graph = pipeline.invoke(topic="restaurant reservation")