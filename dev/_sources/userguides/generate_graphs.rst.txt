Using Dialog2Graph algorithms
=============================

This guide demonstrates usage of ``dialogue2graph`` algorithms to generate a dialogue graph. The following example shows how to use the algorithm that generates graph based on the dialogues you provide using both LLM and Embedding models. Also we will dive into details of :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` usage.

First of all we need to import the :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` and :py:class:`~dialogue2graph.pipelines.d2g_llm.LLMGraphGenerator` we will be using.

.. code-block:: python

    from dialogue2graph import Dialogue
    from dialogue2graph.pipelines.model_storage import ModelStorage
    from dialogue2graph.pipelines.d2g_llm import LLMGraphGenerator
    from dialogue2graph.pipelines.helpers.parse_data import PipelineRawDataType

Now, we need to read the dialogues we want to generate a graph for. In this example we will read the dialogues from a JSON file. The dialogues should be in the following format:

.. code-block:: json

    {
        "dialogs": [
    {"text": "Hey there! How can I help you today?", "participant": "assistant"},
    {"text": "I need to book a ride to the airport.", "participant": "user"},
    {
        "text": "Sure! I can help with that. When is your flight, and where are you departing from?",
        "participant": "assistant"
    },
    {"text": "Do you have any other options?", "participant": "user"},
    {
        "text": "If you'd prefer, I can send you options for ride-share services instead. Would you like that?",
        "participant": "assistant"
    },
    {"text": "No, I'll manage on my own.", "participant": "user"},
    {"text": "No worries! Feel free to reach out anytime.", "participant": "assistant"},
    {"text": "Alright, thanks anyway.", "participant": "user"},
    {"text": "You're welcome! Have a fantastic trip!", "participant": "assistant"}
        ],
    }

Let's read them:

.. code-block:: python

    import json

    with open("example_dialogue.json", "r") as f:
        data = json.load(f)

    dialogues = [Dialogue(**dialogue) for dialogue in data["dialogs"]]
    data = PipelineRawDataType(
        dialogs=dialogues,
        supported_graph=None,
        true_graph=None,
    )

Now we should create a :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` object. This object will be used to store the models we will be using. In this example we will use the LLM model and the Embedding model. The LLM model will be used to generate the graph, and the Embedding model will be used to generate the embeddings for the nodes in the graph.

.. code-block:: python

    model_storage = ModelStorage()
    model_storage.add(
        "my_formatting_model",
        config={
            "model": "gpt-4.1-mini"
        },
        model_type="llm",
    )

    model_storage.add(
        "my_embedding_model",
        config={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        },
        model_type="emb",
    )

Now we can create the :py:class:`~dialogue2graph.pipelines.d2g_llm.LLMGraphGenerator` object. This object will be used to generate the graph. We will pass the :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` object to the constructor of the :py:class:`~dialogue2graph.pipelines.d2g_llm.LLMGraphGenerator` object. Note, that we are overriding the default model on the formatting and similarity tasks with the models we added to the :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage` object. The rest of the models will be used as default. Don't forget to use correct ``model_type`` when adding the model to the :py:class:`~dialogue2graph.pipelines.model_storage.ModelStorage`. The available types are ``llm`` for LLMs and ``emb`` for embedders.

.. code-block:: python

    graph_generator = LLMGraphGenerator(
        model_storage=model_storage,
        formatting_llm="my_formatting_model",
        sim_model="my_embedding_model"
    )

Now we can generate the graph. We will pass the dialogues ``.invoke()`` method of the :py:class:`~dialogue2graph.pipelines.d2g_llm.LLMGraphGenerator` object. The method will return a graph object and a report object. To include the metrics in the report, we need to set the ``enable_evals`` parameter to ``True``. It will run some metrics on the graph during and after the generation process. Keep in mind that this will usually slow down the generation process and rise the token count.

.. code-block:: python

    graph, report = graph_generator.invoke(data, enable_evals=True)
    graph.visualise()

    print(report)
