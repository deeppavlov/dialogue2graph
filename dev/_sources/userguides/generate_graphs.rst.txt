Using Dialog2Graph algorithms
=============================

This guide demonstrates usage of ``dialog2graph`` dialog graph generation algorithms. 
The following example shows how to use the particular algorithm that generates graph from the set of dialogs. It leverages both Embedding and LLM models. 
Also we dive into details of :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` usage.

First of all, we need to import the :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` and 
:py:class:`~dialog2graph.pipelines.d2g_llm.LLMGraphGenerator`.

.. code-block:: python

    from dialog2graph import Dialog
    from dialog2graph.pipelines.model_storage import ModelStorage
    from dialog2graph.pipelines.d2g_llm import LLMGraphGenerator
    from dialog2graph.pipelines.helpers.parse_data import PipelineRawDataType


Now, we need to get the dialogs that are the source for the further graph. In this example we will read the dialogs from a JSON file. 
The dialogs should be presented in the following format:

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

As we got the dialogs from the file, we reformat the initial dictionaries to :py:class:`~dialog2graph.Dialog` objects. Then, the dialogs are 
passed to the data parser :py:class:`~dialog2graph.pipelines.helpers.parse_data.PipelineRawDataType`. So, we've prepared data for the generator:

.. code-block:: python

    import json

    with open("example_dialog.json", "r") as f:
        data = json.load(f)

    dialogs = [Dialog(**dialog) for dialog in data["dialogs"]]
    data = PipelineRawDataType(
        dialogs=dialogs,
        supported_graph=None,
        true_graph=None,
    )

Now we should create a :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` object. This object will be used to store the models for generation. 
In this example we will use the LLM model and the Embedding model. The LLM model will be used to generate the graph, and the Embedding model will be used 
to generate the embeddings for the nodes in the graph.

.. code-block:: python

    model_storage = ModelStorage()
    model_storage.add(
        "my_formatting_model",
        config={
            "model_name": "gpt-4.1-mini"
        },
        model_type=ChatOpenAI,
    )

    model_storage.add(
        "my_embedding_model",
        config={
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {"device": "cpu"}
        },
        model_type=HuggingFaceEmbeddings,
    )

Now we can create the :py:class:`~dialog2graph.pipelines.d2g_llm.LLMGraphGenerator` object. This object will be used to generate the graph. 
We will pass the :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` object to the constructor of the 
:py:class:`~dialog2graph.pipelines.d2g_llm.LLMGraphGenerator` object. Note, that we are overriding the default model on the formatting and 
similarity tasks with the models we added to the :py:class:`~dialog2graph.pipelines.model_storage.ModelStorage` object. 
The rest of the models will be used as default. Don't forget to use correct ``model_type`` when adding the model to the 
:py:class:`~dialog2graph.pipelines.model_storage.ModelStorage`. The available types are ``llm`` for LLMs and ``emb`` for embedders.

.. code-block:: python

    graph_generator = LLMGraphGenerator(
        model_storage=model_storage,
        formatting_llm="my_formatting_model",
        sim_model="my_embedding_model"
    )

Now we can generate the graph. We will pass the dialogs ``.invoke()`` method of the :py:class:`~dialog2graph.pipelines.d2g_llm.LLMGraphGenerator` 
object. The method will return a graph object and a report object. To include the metrics in the report, we need to set the ``enable_evals`` 
parameter to ``True``. It will run some metrics on the graph during and after the generation process. Keep in mind that this will usually slow down 
the generation process and rise the token count.

.. code-block:: python

    graph, report = graph_generator.invoke(data, enable_evals=True)
    graph.visualise()

    print(report)
