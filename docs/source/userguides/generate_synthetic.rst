Generate synthetic graph on certain topic
=========================================

Use ``LoopedGraphGenerator`` to create a validated graph from several LLM generated dialogues concerning a given topic. 

.. code-block:: python

    from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
    from langchain_community.chat_models import ChatOpenAI

1. Choose LLMs for dialogue generation and dialogue validation

.. code-block:: python

    gen_model = ChatOpenAI(
        model='gpt-4o',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    val_model = ChatOpenAI(
        model='gpt-3.5-turbo',
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0,
    )

2. Create ``LoopedGraphGenerator`` and use ``invoke`` method to get a dialogue graph

.. code-block:: python

    pipeline = LoopedGraphGenerator(
        generation_model=gen_model,
        validation_model=val_model,
    )

    generated_graph = pipeline.invoke(topic="restaurant reservation")