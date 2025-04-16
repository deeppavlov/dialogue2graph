Using Dialog2Graph pipelines
============================

This document describes how to use Dialog2Graph pipelines to process your data. The following example shows how to use the pipeline that generates graph based on the dialogues you provide using both LLM and Embedding models. Also we will dive into details of ``ModelStorage`` usage.

First of all we need to import the ``ModelStorage`` and ``Pipeline`` we will be using.

.. code-block:: python

    from dialogue2graph.pipelines.model_storage import ModelStorage
    from dialogue2graph.pipelines.d2g_llm.pipeline import Pipeline as D2GLLMPipeline


``ModelStorage`` instance is used to store LLM and SentenceTransformer models and use cached variants to avoid multiple instances of the same model being up simultaneously.

Each ``Pipeline`` has it's own default models, but you can override them by passing the key to the model you've added to the ``ModelStorage``.

.. code-block:: python

    ms = ModelStorage()

    ms.add(
        "my_formatting_model",
        config={
            "model": "gpt-3.5-turbo"
        },
        model_type="llm",
    )

    ms.add(
        "my_embedding_model",
        config={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        model_type="emb",
    )


    pipe = D2GLLMPipeline(
        name="d2g_pipeline" # name to save reports for (useful if multiple pipelines are used)
        ms,
        formatting_llm="my_formatting_model",
        sim_model="my_embedding_model"
        )

In this example we are overriding the default "gpt-4o-mini" model with "gpt-3.5-turbo" model for the formatting task in the pipeline and the similarity model to use "all-MiniLM-L6-v2". The rest of the models will be used as default. Don't forget to use correct ``model_type`` when adding the model to the ``ModelStorage``. The available types are ``llm`` for LLMs and ``emb`` for embedders.

Note, that the adding of the models to the ``ModelStorage`` is not necessary if you are using the default models.

.. code-block:: python

    # Define the data, but it can also be a path to the JSON file or a list of Dialogue objects
    data = [{'text': 'Hey there! How can I help you today?',
        'participant': 'assistant'},
    {'text': 'I need to book a ride to the airport.', 'participant': 'user'},
    {'text': 'Sure! I can help with that. When is your flight, and where are you departing from?',
        'participant': 'assistant'},
    {'text': 'Do you have any other options?', 'participant': 'user'},
    {'text': "If you'd prefer, I can send you options for ride-share services instead. Would you like that?",
        'participant': 'assistant'},
    {'text': "No, I'll manage on my own.", 'participant': 'user'},
    {'text': 'No worries! Feel free to reach out anytime.',
        'participant': 'assistant'},
    {'text': 'Alright, thanks anyway.', 'participant': 'user'},
    {'text': "You're welcome! Have a fantastic trip!",
        'participant': 'assistant'}]

    # Invoke the pipeline to get the graph and report objects
    graph, report = pipe.invoke(data)
    report.to_markdown("report.md")

That's it! Now, you have a ``Graph`` object that you can use for further processing.
And you have a ``Report`` object that contains some metrics regarding your data. It can be exported to various formats using built in functions.

If needed you can both save your ``ModelStorage`` and load it later.

.. code-block:: python

    # Save the ModelStorage to a file
    ms.save("models_config.yml")

    # Load the ModelStorage from a file
    ms = ModelStorage.load("models_config.yml")
