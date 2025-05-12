:tutorial_name: evaluation/examples_of_metrics_usage.ipynb

Evaluate graphs and dialogs
============================

``dialog2graph`` offers different evaluation metrics to assess dialog graph completeness and validity. 
Metrics include both LLM and no-LLM metrics and may be used by themselves or received in a pipeline report
by invoking ``evaluate`` method or enabling evaluation with ``enable_evals`` parameter. 
``dialog2graph`` also has validators to assess generated dialog structure.

This userguide demonstrates how to use metrics and validators by themselves. See how to use evaluation within pipeline :doc:`in this userguide <./generate_graphs>`.

Graph evaluation
~~~~~~~~~~~~~~~~~~~~

Graph evaluation encompass metrics to compare 2 graphs on their structure and metrics to assess graph and its dialog correspondence.

Let's try to compare 2 graphs. First, we need to define graphs for comparison. 
Graphs are received from `generated set "d2g_generated" <https://huggingface.co/datasets/DeepPavlov/d2g_generated>`_.

.. code-block:: python

    from datasets import load_dataset
    from dialog2graph import Graph

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)
    graph1_data = data["train"][0]["graph"]
    graph2_data = data["train"][1]["graph"]
    graph1 = Graph(graph_dict=graph1_data)
    graph2 = Graph(graph_dict=graph2_data)

Then, we may choose metric to assess 2 graphs or graph and dialogs. For example, the graphs may be compared on the structure using 
:py:class:`~dialog2graph.metrics.no_llm_metrics.metrics.is_same_structure` method. Other available metrics for 2 graph comparison include 
:py:class:`~dialog2graph.metrics.no_llm_metrics.metrics.match_graph_triplets`, :py:class:`~dialog2graph.metrics.no_llm_metrics.metrics.triplet_match_accuracy`, 
:py:class:`~dialog2graph.metrics.no_llm_metrics.metrics.compute_graph_metrics`.

.. code-block:: python

    from dialog2graph.metrics.no_llm_metrics.metrics import is_same_structure

    is_same_structure(graph1, graph2)

The output is boolean.

Dialog evaluation
~~~~~~~~~~~~~~~~~~~~

Dialog evaluation is aimed to validate dialog formal structure (e.g. role switch) across 2 dialogs and to validate dialogs by themselves on
turn logic.

To evaluate graph dialogs, first, the dialogs are loaded from the same set and :py:class:`~dialog2graph.Dialog` instance is created from messages.

.. code-block:: python

    from datasets import load_dataset
    from dialog2graph import Dialog

    dataset = load_dataset("DeepPavlov/d2g_generated", token=True)
    dialog_template = Dialog()
    good_graph = [
        dialog_template.from_list(dialog["messages"])
        for dialog in dataset["train"][0]["dialogs"]
    ]

Now, we import validators to check whether greeting phrases are repeated.

.. code-block:: python

    from dialog2graph.metrics.no_llm_validators import is_greeting_repeated_regex

    is_greeting_repeated_regex(good_graph)

The output is boolean.