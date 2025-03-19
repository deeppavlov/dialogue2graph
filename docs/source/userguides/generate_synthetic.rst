Generate synthetic graph on certain topic
=========================================

Choose LLMs for generating and validating dialogue graph and invoke graph generation

.. code-block:: python

    from dialogue2graph.datasets.complex_dialogues.generation import LoopedGraphGenerator
    from langchain_openai import ChatOpenAI


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

    pipeline = LoopedGraphGenerator(
        generation_model=gen_model,
        validation_model=val_model,
    )

    generated_graph = pipeline.invoke(topic="restaurant reservation")