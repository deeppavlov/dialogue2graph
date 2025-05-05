:tutorial_name: data_generation/LoopedGraphGenerator_example.ipynb

Generate synthetic graph on certain topic
=========================================

Use :py:class:`~dialogue2graph.datasets.complex_dialogues.generation.LoopedGraphGenerator` to create a validated graph from several 
LLM generated dialogues concerning a given topic. 

.. code-block:: python

    from langchain_openai import ChatOpenAI

    from dialog2graph.datasets.complex_dialogs.generation import LoopedGraphGenerator
    from dialog2graph.pipelines.model_storage import ModelStorage

1. Create :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` instance and add choosen LLMs for dialogue generation, 
dialogue validation, theme validation and cycle end search.

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

2. Create :py:class:`~dialogue2graph.datasets.complex_dialogues.generation.LoopedGraphGenerator` and 
use :py:class:`~dialogue2graph.datasets.complex_dialogues.generation.LoopedGraphGenerator.invoke` method to get a dialogue graph

.. code-block:: python

    pipeline = LoopedGraphGenerator(
        model_storage=model_storage,
        generation_llm='gen_model',
        validation_llm='help_model',
        cycle_ends_llm='help_model',
        theme_validation_llm='help_model'
    )

    generated_graph = pipeline.invoke(topic="restaurant reservation")