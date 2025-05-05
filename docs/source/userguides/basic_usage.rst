:tutorial_name: basics/base_classes_usage.ipynb

Learn base dialog2graph classes
=========================================

:py:class:`~dialogue2graph.pipelines.core.graph.Graph` and :py:class:`~dialogue2graph.pipelines.core.dialogue.Dialogue` are the base structures used in the 
``dialog2graph`` project. They are perfect to store information about your graphs and dialogues in a clear format.

Create dialogue2graph.Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialogue2graph.pipelines.core.graph.Graph` enables to create a directed dialog graph and work with it. 
You can visualise it and sample dialogs from it. 

For experimenting with dialog graphs a set with generated data was created. 
It can be loaded from `HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_generated>`_ and contains 402 dialog graphs on various 
topics concerning custom support and other topics. See more information about available data on :doc:`data collection page <../research/data_collections>`.

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)

First, initialize :py:class:`~dialog2graph.pipelines.core.graph.Graph` by passing a dictionary with graph edges and nodes (``{"edges": [...], "nodes": [...]}``).

.. code-block:: python
    
    from dialogue2graph import Graph

    graph = Graph(graph_dict=dataset['train'][5]["graph"])

Then you can visualise the graph with all the utterances it contains.

.. code-block:: python

    graph.visualise()

Or you can visualise the graph in a more schematic way to see its general structure.

.. code-block:: python

    graph_title = "Schematic graph view"
    graph.visualise_short(graph_title)

Create RecursiveDialogueSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialogue2graph.pipelines.core.dialogue_sampling.RecursiveDialogueSampler` is a class that helps to sample dialogs from existing dialog graphs. 
It uses recursion to get all possible dialogs from the graph.
To sample dialogs from the graph, create :py:class:`~dialogue2graph.pipelines.core.dialogue_sampling.RecursiveDialogueSampler` instance and use 
``invoke`` method to start sampling process.

.. code-block:: python

    from dialog2graph.pipelines.core.dialog_sampling import RecursiveDialogSampler
    from langchain_openai import ChatOpenAI

    sampler = RecursiveDialogueSampler()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    dialogs: list = sampler.invoke(graph=graph, upper_limit=10, cycle_ends_model=model)

The output of :py:class:`~dialogue2graph.pipelines.core.dialogue_sampling.RecursiveDialogueSampler.invoke` method is a list 
of :py:class:`~dialogue2graph.pipelines.core.dialogue.Dialogue` instances. This class is also helpful when working with dialog graphs.

.. code-block:: python
    
    type(dialogs[0])

Use dialog2graph.Dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialog2graph.pipelines.core.dialog.Dialog` is a class that represents a complete dialog and provide method for visualisation and converting. 

.. code-block:: python

    print(dialogs[0])
    dialogs[0].to_list()