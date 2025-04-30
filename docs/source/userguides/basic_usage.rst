Learn base dialog2graph classes
=========================================

Create dialog2graph.Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialog2graph.pipelines.core.graph.Graph`  is a base class in dialog2graph package that enables to create a directed dialog graph and work with it. You can visualise it and sample dialogs from it. 

.. code-block:: python

    from dialog2graph import Graph

For experimenting with dialog graphs a set with generated data was created. It can be loaded from `HuggingFace <https://huggingface.co/datasets/DeepPavlov/d2g_generated>`_ and contains 402 dialog graphs on various topics concerning custom support and other topics.

.. code-block:: python

    from datasets import load_dataset

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)

First, initialize :py:class:`~dialog2graph.pipelines.core.graph.Graph` by passing a dictionary with graph edges and nodes (``{"edges": [...], "nodes": [...]}``).

.. code-block:: python

    graph = Graph(graph_dict=dataset['train'][5]["graph"])

Then you can visualise the graph with all the utterances it contains.

.. code-block:: python

    graph.visualise()

Or you can visualise the graph in a more schematic way to see its general structure.

.. code-block:: python

    graph_title = "Schematic graph view"
    graph.visualise_short(graph_title)

Create dialog2graph.pipelines.core.dialog_sampling.RecursiveDialogSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialog2graph.pipelines.core.dialog_sampling.RecursiveDialogSampler` is a class that helps to sample dialogs from existing dialog graphs. It uses recursion to get all possible dialogs from the graph.

.. code-block:: python

    from dialog2graph.pipelines.core.dialog_sampling import RecursiveDialogSampler
    from langchain_openai import ChatOpenAI

To sample dialogs from the graph, create :py:class:`~dialog2graph.pipelines.core.dialog_sampling.RecursiveDialogSampler` instance and use invoke method to start sampling process.

.. code-block:: python

    sampler = RecursiveDialogSampler()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    dialogs: list = sampler.invoke(graph=graph, upper_limit=10, cycle_ends_model=model)

The output of :py:class:`~dialog2graph.pipelines.core.dialog_sampling.RecursiveDialogSampler.invoke` method is a list of :py:class:`~dialog2graph.pipelines.core.dialog.Dialog` instances. This class is also helpful when working with dialog graphs.

.. code-block:: python
    
    type(dialogs[0])

Use dialog2graph.Dialog
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:class:`~dialog2graph.pipelines.core.dialog.Dialog` is a class that represents a complete dialog and provide method for visualisation and converting. 

.. code-block:: python

    print(dialogs[0])
    dialogs[0].to_list()